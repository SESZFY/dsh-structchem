# -*- coding: utf-8 -*-
"""真 MCP 协议端到端测试：起子进程，走 stdio JSON-RPC，不绕开协议层。

server --selftest 是直接调 Python 函数，看不到「MCP 客户端装得上吗、工具真的暴露了吗、
参数经 JSON 往返后还对不对」。这个脚本补上那一层。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

SRC = r"D:\大肥鱼的实验室\dsh-structchem\src"
ENV = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")

try:  # 让进度实时可见，否则看起来像卡死
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


class Client:
    def __init__(self) -> None:
        self.p = subprocess.Popen(
            [sys.executable, "-u", "-m", "structchem_mcp.server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=ENV, cwd=os.path.dirname(SRC), text=True, encoding="utf-8", bufsize=1,
        )
        self._id = 0

    def send(self, method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._id += 1
            msg["id"] = self._id
        self.p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                err = self.p.stderr.read()
                raise RuntimeError(f"MCP 进程无响应。stderr:\n{err}")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == self._id:
                return resp

    def close(self) -> None:
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def brief(x, n=120):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s[:n]


def main() -> int:
    ok = 0
    fail = 0

    def chk(label, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"[PASS] {label}")
        else:
            fail += 1
            print(f"[FAIL] {label}   {detail}")

    c = Client()
    try:
        # 1) 握手
        r = c.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "contract-probe", "version": "0"},
        })
        assert "result" in r, f"initialize 失败：{r}"
        info = r["result"].get("serverInfo", {})
        chk("MCP initialize", info.get("name") == "structchem",
            f"serverInfo={info}")

        c.send("notifications/initialized", notify=True)

        # 2) 工具清单
        r = c.send("tools/list", {})
        tools = r["result"]["tools"]
        names = sorted(t["name"] for t in tools)
        chk("tools/list 数量 = 30", len(tools) == 30, f"实得 {len(tools)}")
        must = {"sc_point_group", "sc_character_table", "sc_normal_modes",
                "sc_reduce_representation", "sc_selection_rules", "sc_huckel",
                "sc_spectroscopic_terms", "sc_diatomic_mo", "sc_vsepr",
                "sc_miller_indices", "sc_packing", "sc_radius_ratio",
                "sc_crystal_field", "sc_answer_card", "sc_rationale",
                "sc_book_index", "sc_style_law", "sc_notation", "sc_toolbox"}
        missing = sorted(must - set(names))
        chk("关键工具全部暴露", not missing, f"缺 {missing}")
        # 每个工具必须有非空 description（模型靠它选工具）
        nodesc = [t["name"] for t in tools if not (t.get("description") or "").strip()]
        chk("每个工具都有 description", not nodesc, f"缺 {nodesc}")
        # 输入 schema 必须存在
        noschema = [t["name"] for t in tools if "inputSchema" not in t]
        chk("每个工具都有 inputSchema", not noschema, f"缺 {noschema}")

        # 回归锁：注入用的 impl 绝不能出现在对外 schema 里
        # （曾因 functools.wraps 跟随 __wrapped__，24 个工具的 required 全变成
        #   ["impl", ...]，模型被迫传字符串 → 整个插件在 MCP 层不可用）
        leak = [t["name"] for t in tools
                if "impl" in (t.get("inputSchema", {}).get("properties") or {})]
        chk("schema 里没有泄漏 impl 参数", not leak, f"泄漏 {leak}")
        req_bad = [t["name"] for t in tools
                   if "impl" in (t.get("inputSchema", {}).get("required") or [])]
        chk("required 里没有 impl", not req_bad, f"{req_bad}")

        # 3) 真调工具（参数走 JSON 往返）
        def call(name, args):
            resp = c.send("tools/call", {"name": name, "arguments": args})
            if "error" in resp:
                return {"_rpc_error": resp["error"]}
            res = resp["result"]
            if res.get("isError"):
                return {"_tool_error": brief(res)}
            txt = res["content"][0]["text"]
            return json.loads(txt)

        r1 = call("sc_point_group", {"atoms": [
            {"sym": "C", "xyz": [0, 0, 0]},
            {"sym": "Cl", "xyz": [0, 1.466, -0.992]},
            {"sym": "Cl", "xyz": [0, -1.466, -0.992]},
            {"sym": "H", "xyz": [0.895, 0, 0.621]},
            {"sym": "H", "xyz": [-0.895, 0, 0.621]},
        ]})
        chk("sc_point_group(CH2Cl2) = C2v", r1.get("point_group") == "C2v",
            brief(r1))

        import math
        ben = ([{"sym": "C", "xyz": [1.397 * math.cos(math.pi / 3 * i),
                                     1.397 * math.sin(math.pi / 3 * i), 0]} for i in range(6)] +
               [{"sym": "H", "xyz": [2.481 * math.cos(math.pi / 3 * i),
                                     2.481 * math.sin(math.pi / 3 * i), 0]} for i in range(6)])
        r2 = call("sc_point_group", {"atoms": ben})
        chk("sc_point_group(苯) = D6h", r2.get("point_group") == "D6h", brief(r2))
        chk("D6h 阶 = 24", r2.get("order") == 24, f"order={r2.get('order')}")

        r3 = call("sc_normal_modes", {"point_group": "Oh", "n_atoms": 7})
        gv = r3.get("gamma_vib") or {}
        chk("sc_normal_modes(Oh,7) SF6",
            gv == {"A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1}, brief(gv))
        chk("SF6 IR 活性仅 T1u", r3.get("ir_active") == ["T1u"],
            brief(r3.get("ir_active")))

        r4 = call("sc_huckel", {"topology": "cyclic", "n_atoms": 6, "n_electrons": 6})
        xs = sorted((round(o["E_beta"], 4) for o in r4.get("orbitals", [])), reverse=True)
        chk("sc_huckel(苯) 能级", xs == [2.0, 1.0, 1.0, -1.0, -1.0, -2.0], brief(xs))

        r5 = call("sc_spectroscopic_terms", {"config": "d2"})
        chk("sc_spectroscopic_terms(d2) 基项 3F",
            str(r5.get("ground_term", "")).startswith("3F"), brief(r5.get("ground_term")))
        chk("d2 微观状态 45", r5.get("microstates_total") == 45,
            f"{r5.get('microstates_total')}")

        r6 = call("sc_diatomic_mo", {"n_electrons": 16})
        chk("sc_diatomic_mo(16) = O2 键级2 顺磁",
            r6.get("species") == "O2" and r6.get("bond_order") == 2.0
            and r6.get("unpaired_electrons") == 2, brief(r6))

        r7 = call("sc_miller_indices", {"intercepts": ["a/2", "b/3", "c/4"]})
        chk("sc_miller_indices → (234)", str(r7.get("miller", "")).replace(" ", "") == "(234)",
            brief(r7.get("miller")))

        r8 = call("sc_radius_ratio", {"r_plus_pm": 133, "r_minus_pm": 195})
        chk("sc_radius_ratio(133/195) → CN6 八面体",
            r8.get("coordination_number") == 6, brief(r8.get("coordination_number")))

        r9 = call("sc_crystal_field", {"d_n": 6, "geometry": "octahedral",
                                       "delta_o_cm1": 10000, "pairing_energy_cm1": 21000})
        chk("sc_crystal_field(d6 弱场) 4 单电子", r9.get("unpaired_electrons") == 4,
            brief(r9.get("unpaired_electrons")))

        r10 = call("sc_answer_card", {"query": "点群"})
        chk("sc_answer_card 检索可用", r10.get("ok") and r10.get("count", 0) >= 1,
            brief(r10.get("count")))

        r11 = call("sc_style_law", {})
        chk("sc_style_law 能读到 answer-style.md",
            r11.get("ok") and len(r11.get("markdown", "")) > 500, brief(r11.get("error")))

        r12 = call("sc_toolbox", {})
        chk("sc_toolbox 报告 29 个工具", r12.get("tool_count") == 29,
            f"{r12.get('tool_count')}")
        chk("无损坏模块", not r12.get("broken_modules"), brief(r12.get("broken_modules")))

        # 4) 非法输入必须优雅返回（不许 MCP 层报错或崩）
        r13 = call("sc_point_group", {"atoms": [{"sym": "X", "xyz": [0, 0, 0]}]})
        chk("非法原子输入 → ok=False 且给中文原因",
            r13.get("ok") is False and r13.get("error"), brief(r13))
        r14 = call("sc_character_table", {"point_group": "Q9"})
        chk("不存在的点群 → ok=False", r14.get("ok") is False, brief(r14))
        r15 = call("sc_normal_modes", {"point_group": "Oh", "n_atoms": -3})
        chk("负原子数 → ok=False", r15.get("ok") is False, brief(r15))

        # 5) 进程仍活着（前面 15 次调用没让 server 崩）
        chk("15 次调用后 server 仍存活", c.p.poll() is None, f"poll={c.p.poll()}")
    finally:
        c.close()

    print(f"\n{ok}/{ok + fail} passed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
