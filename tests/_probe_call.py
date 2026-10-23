# -*- coding: utf-8 -*-
"""一次性投喂多条 JSON-RPC，用 communicate 收全部回包（避开交互式 readline 的坑）。"""
import json
import os
import subprocess
import sys

SRC = r"D:\大肥鱼的实验室\dsh-structchem\src"
ENV = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")

msgs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "probe", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "sc_point_group",
                "arguments": {"atoms": [
                    {"sym": "C", "xyz": [0, 0, 0]},
                    {"sym": "Cl", "xyz": [0, 1.466, -0.992]},
                    {"sym": "Cl", "xyz": [0, -1.466, -0.992]},
                    {"sym": "H", "xyz": [0.895, 0, 0.621]},
                    {"sym": "H", "xyz": [-0.895, 0, 0.621]}]}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "sc_normal_modes",
                "arguments": {"point_group": "Oh", "n_atoms": 7}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
     "params": {"name": "sc_miller_indices",
                "arguments": {"intercepts": ["a/2", "b/3", "c/4"]}}},
]
payload = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n"

p = subprocess.Popen(
    [sys.executable, "-u", "-m", "structchem_mcp.server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=ENV, text=True, encoding="utf-8",
)
try:
    out, err = p.communicate(input=payload, timeout=90)
except subprocess.TimeoutExpired:
    p.kill()
    out, err = p.communicate()
    print("!!! TIMEOUT after 90s")
print("stderr:", (err or "")[:1500] or "(empty)")
print("--- 回包 ---")
for line in (out or "").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        print("  [非 JSON]", line[:200])
        continue
    i = r.get("id")
    if i == 2:
        ts = r["result"]["tools"]
        print(f"  id=2 tools/list: {len(ts)} 个工具")
        # 抽查 schema 是否还有 impl
        leak = [t["name"] for t in ts
                if "impl" in (t.get("inputSchema", {}).get("properties") or {})]
        print(f"        schema 泄漏 impl 的工具数: {len(leak)}")
        pg = next((t for t in ts if t["name"] == "sc_point_group"), None)
        if pg:
            sch = pg["inputSchema"]
            print(f"        sc_point_group 参数: {list(sch.get('properties', {}))} "
                  f"required={sch.get('required')}")
    elif i in (3, 4, 5):
        if "error" in r:
            print(f"  id={i} RPC 错误: {r['error']}")
            continue
        res = r["result"]
        if res.get("isError"):
            print(f"  id={i} 工具报错: {res['content'][0]['text'][:200]}")
            continue
        try:
            d = json.loads(res["content"][0]["text"])
        except Exception:
            print(f"  id={i} 非 JSON 回包: {res['content'][0]['text'][:200]}")
            continue
        if i == 3:
            print(f"  id=3 sc_point_group → {d.get('point_group')} "
                  f"order={d.get('order')} steps={len(d.get('steps') or [])}")
        elif i == 4:
            print(f"  id=4 sc_normal_modes(Oh,7) → Γ_vib={d.get('gamma_vib')} "
                  f"IR={d.get('ir_active')}")
        elif i == 5:
            print(f"  id=5 sc_miller_indices → {d.get('miller')} hkl=({d.get('h')},{d.get('k')},{d.get('l')})")
    else:
        print(f"  id={i}: {json.dumps(r, ensure_ascii=False)[:150]}")
print("进程退出码:", p.returncode)
