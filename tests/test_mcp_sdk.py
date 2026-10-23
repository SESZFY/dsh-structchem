# -*- coding: utf-8 -*-
"""用官方 MCP SDK 客户端做端到端验证（与 DSH 走同一套协议实现）。

自己手搓 JSON-RPC 容易掩盖分帧/并发问题；这里直接用 mcp.client.stdio，
它和 dsh-mcp-client 用的是同一协议栈。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

SRC = r"D:\大肥鱼的实验室\dsh-structchem\src"
os.environ["PYTHONPATH"] = SRC
os.environ["PYTHONIOENCODING"] = "utf-8"

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

PY = sys.executable
ROOT = os.path.dirname(SRC)          # 包根：dsh-structchem
EXPECT_TOOLS = 30

CASES = [
    ("sc_point_group", {"atoms": [
        {"sym": "C", "xyz": [0, 0, 0]},
        {"sym": "Cl", "xyz": [0, 1.466, -0.992]},
        {"sym": "Cl", "xyz": [0, -1.466, -0.992]},
        {"sym": "H", "xyz": [0.895, 0, 0.621]},
        {"sym": "H", "xyz": [-0.895, 0, 0.621]}]},
     lambda d: d.get("point_group") == "C2v"),
    ("sc_point_group", {"atoms": [
        {"sym": "S", "xyz": [0, 0, 0]}] + [
        {"sym": "F", "xyz": v} for v in
        ([1.56, 0, 0], [-1.56, 0, 0], [0, 1.56, 0], [0, -1.56, 0], [0, 0, 1.56], [0, 0, -1.56])]},
     lambda d: d.get("point_group") == "Oh" and d.get("order") == 48),
    ("sc_normal_modes", {"point_group": "Oh", "n_atoms": 7},
     lambda d: (d.get("gamma_vib") or {}) == {"A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1}),
    ("sc_character_table", {"point_group": "D4h"},
     lambda d: d.get("order") == 16 and len(d.get("irreps") or []) == 10),
    ("sc_reduce_representation", {"point_group": "C2v", "chars": [9, -1, 3, 1]},
     lambda d: {x["irrep"]: x["count"] for x in d["decomposition"] if x["count"]}
     == {"A1": 3, "A2": 1, "B1": 3, "B2": 2}),
    ("sc_selection_rules", {"point_group": "D2d"},
     lambda d: d["dipole"]["has"] is False and d["chiral"]["has"] is False),
    ("sc_spectroscopic_terms", {"config": "d2"},
     lambda d: d.get("microstates_total") == 45
     and str(d.get("ground_term", "")).startswith("3F")),
    ("sc_huckel", {"topology": "cyclic", "n_atoms": 6, "n_electrons": 6},
     lambda d: sorted((round(o["E_beta"], 4) for o in d["orbitals"]), reverse=True)
     == [2.0, 1.0, 1.0, -1.0, -1.0, -2.0]),
    ("sc_diatomic_mo", {"n_electrons": 16},
     lambda d: d.get("species") == "O2" and d.get("bond_order") == 2.0
     and d.get("unpaired_electrons") == 2),
    ("sc_vsepr", {"central": "S", "ligands": ["F", "F", "F", "F"], "lone_pairs": 1},
     lambda d: d.get("point_group") == "C2v"),
    ("sc_miller_indices", {"intercepts": ["a/2", "b/3", "c/4"]},
     lambda d: str(d.get("miller", "")).replace(" ", "") == "(234)"),
    ("sc_packing", {"kind": "fcc"},
     lambda d: d.get("atoms_per_cell") == 4 and d.get("coordination_number") == 12),
    ("sc_radius_ratio", {"r_plus_pm": 133, "r_minus_pm": 195},
     lambda d: d.get("coordination_number") == 6),
    ("sc_crystal_field", {"d_n": 6, "geometry": "octahedral",
                          "delta_o_cm1": 10000, "pairing_energy_cm1": 21000},
     lambda d: d.get("unpaired_electrons") == 4),
    ("sc_particle_box", {"n": 3, "L_pm": 100.0, "dim": 1},
     lambda d: abs(d["levels"][2]["E_eV"] / d["levels"][0]["E_eV"] - 9) < 1e-6),
    ("sc_slater", {"config": "Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2", "target": "3d"},
     lambda d: abs(d.get("z_eff", 0) - 6.25) < 0.01),
    ("sc_answer_card", {"query": "点群"}, lambda d: d.get("ok") and d.get("count", 0) >= 1),
    ("sc_rationale", {"query": "偶极矩"}, lambda d: d.get("ok") and d.get("count", 0) >= 1),
    ("sc_notation", {"query": "σ"},
     lambda d: d.get("ok") and len(d.get("symbols") or []) == 1
     and len(d["symbols"][0]["senses"]) == 4),
    ("sc_notation", {"query": "点群"},
     lambda d: d.get("ok") and len(d.get("typography") or []) >= 1),
    ("sc_book_index", {"chapter": "4"}, lambda d: d.get("ok") and d.get("count", 0) >= 1),
    ("sc_style_law", {}, lambda d: d.get("ok") and len(d.get("markdown", "")) > 500),
    ("sc_toolbox", {}, lambda d: d.get("tool_count") == EXPECT_TOOLS
     and not d.get("broken_modules")),
]


async def call(session, name, args):
    """单次调用加硬超时：卡住时要能看到「是哪个工具卡」，而不是整个测试挂死。"""
    try:
        res = await asyncio.wait_for(session.call_tool(name, args), timeout=25)
    except asyncio.TimeoutError:
        return {"_err": "TIMEOUT 25s"}
    if res.isError:
        return {"_err": res.content[0].text[:300]}
    txt = res.content[0].text
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt[:300]}


async def main() -> int:
    ok = fail = 0

    def chk(label, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"[PASS] {label}")
        else:
            fail += 1
            print(f"[FAIL] {label}   {detail}")

    params = StdioServerParameters(
        command=PY, args=["-m", "structchem_mcp.server"],
        env={"PYTHONPATH": SRC, "PYTHONIOENCODING": "utf-8"}, cwd=ROOT,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            chk("initialize", init.serverInfo.name == "structchem",
                f"{init.serverInfo}")

            listed = await session.list_tools()
            tools = listed.tools
            chk(f"tools/list = {EXPECT_TOOLS}", len(tools) == EXPECT_TOOLS,
                f"实得 {len(tools)}")

            leak = [t.name for t in tools
                    if "impl" in (t.inputSchema.get("properties") or {})]
            chk("schema 无 impl 泄漏", not leak, f"{leak[:3]}")
            nodesc = [t.name for t in tools if not (t.description or "").strip()]
            chk("每个工具都有 description", not nodesc, f"{nodesc[:3]}")

            # warmup：先打一发最轻的，让 SDK 侧的响应路由/解码路径就位。
            # 不加这一步时，会话里的**第一次** call_tool 会被 asyncio.wait_for 判超时，
            # 而同一条调用在独立进程里只要 1.6 s（含全部启动）—— 是客户端首次调用路径
            # 的开销，不是插件算得慢。
            await call(session, "sc_toolbox", {})

            # 逐个真调（走完整协议，含并发序列化）
            for name, args, pred in CASES:
                try:
                    d = await call(session, name, args)
                except Exception as exc:  # noqa: BLE001
                    chk(f"{name} 调用", False, f"{type(exc).__name__}: {exc}")
                    continue
                good = pred(d) if "_err" not in d and "_raw" not in d else False
                chk(f"{name}({json.dumps(args, ensure_ascii=False)[:52]})",
                    good, json.dumps(d, ensure_ascii=False)[:160])

            # 非法输入必须优雅
            bad = await call(session, "sc_character_table", {"point_group": "Q9"})
            chk("非法点群 → ok=False", bad.get("ok") is False, str(bad)[:120])
            bad2 = await call(session, "sc_normal_modes",
                              {"point_group": "Oh", "n_atoms": -3})
            chk("负原子数 → ok=False", bad2.get("ok") is False, str(bad2)[:120])

            # 调用完还能继续用（session 未坏）
            again = await call(session, "sc_toolbox", {})
            chk("连打 24 次后 session 仍可用", again.get("ok") is True, str(again)[:120])

    print(f"\n{ok}/{ok + fail} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
