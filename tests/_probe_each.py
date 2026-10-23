# -*- coding: utf-8 -*-
"""逐工具隔离定位：每个工具单独起一次 MCP 子进程，找出到底哪个调用会卡。

思路：把「哪个工具在 MCP 环境里不返回」变成可复现的单点结论，
而不是在一个 20 次调用的长会话里赌它卡在哪一步。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

SRC = r"D:\大肥鱼的实验室\dsh-structchem\src"
ROOT = os.path.dirname(SRC)
ENV = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")

CASES = [
    ("sc_toolbox", {}),
    ("sc_answer_card", {"query": "点群"}),
    ("sc_rationale", {"query": "偶极矩"}),
    ("sc_book_index", {"chapter": "4"}),
    ("sc_style_law", {}),
    ("sc_particle_box", {"n": 3, "L_pm": 100.0, "dim": 1}),
    ("sc_hydrogen_orbital", {"n": 3, "l": 1, "m": 0}),
    ("sc_slater", {"config": "Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2", "target": "3d"}),
    ("sc_spectroscopic_terms", {"config": "d2"}),
    ("sc_superposition", {"terms": [{"coef": 0.8, "n": 2, "l": 1, "m": 0},
                                    {"coef": 0.6, "n": 3, "l": 1, "m": 1}]}),
    ("sc_point_group", {"atoms": [
        {"sym": "C", "xyz": [0, 0, 0]},
        {"sym": "Cl", "xyz": [0, 1.466, -0.992]},
        {"sym": "Cl", "xyz": [0, -1.466, -0.992]},
        {"sym": "H", "xyz": [0.895, 0, 0.621]},
        {"sym": "H", "xyz": [-0.895, 0, 0.621]}]}),
    ("sc_symmetry_elements", {"point_group": "C2v"}),
    ("sc_character_table", {"point_group": "D4h"}),
    ("sc_reduce_representation", {"point_group": "C2v", "chars": [9, -1, 3, 1]}),
    ("sc_normal_modes", {"point_group": "C2v", "n_atoms": 3}),
    ("sc_normal_modes", {"point_group": "Oh", "n_atoms": 7}),
    ("sc_selection_rules", {"point_group": "D2d"}),
    ("sc_compose_operations", {"point_group": "D2h", "op1": "C2(z)", "op2": "σ(xy)"}),
    ("sc_huckel", {"topology": "cyclic", "n_atoms": 6, "n_electrons": 6}),
    ("sc_aromaticity", {"n_pi": 6, "n_atoms": 6}),
    ("sc_diatomic_mo", {"n_electrons": 16}),
    ("sc_vsepr", {"central": "S", "ligands": ["F"] * 4, "lone_pairs": 1}),
    ("sc_unit_cell", {"a": 500.0, "b": 500.0, "c": 500.0}),
    ("sc_miller_indices", {"intercepts": ["a/2", "b/3", "c/4"]}),
    ("sc_d_spacing", {"h": 1, "k": 1, "l": 1, "a": 400.0}),
    ("sc_packing", {"kind": "fcc"}),
    ("sc_radius_ratio", {"r_plus_pm": 133, "r_minus_pm": 195}),
    ("sc_cell_content", {"formula": "NaCl", "a_pm": 564.0, "z": 4}),
    ("sc_crystal_field", {"d_n": 6, "geometry": "octahedral",
                          "delta_o_cm1": 10000, "pairing_energy_cm1": 21000}),
    ("sc_spectrochemical_series", {"ligand": "CN-"}),
]


def run_one(name: str, args: dict, timeout: int = 30) -> tuple[str, str]:
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": name, "arguments": args}},
    ]
    payload = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n"
    try:
        p = subprocess.run(
            [sys.executable, "-u", "-m", "structchem_mcp.server"],
            input=payload, capture_output=True, text=True, encoding="utf-8",
            env=ENV, cwd=ROOT, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f">{timeout}s 未返回"
    got = None
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") == 2:
            got = r
            break
    if got is None:
        return "NORESP", f"rc={p.returncode} stderr={(p.stderr or '')[:120]}"
    if "error" in got:
        return "RPCERR", json.dumps(got["error"], ensure_ascii=False)[:120]
    res = got["result"]
    if res.get("isError"):
        return "TOOLERR", res["content"][0]["text"][:120]
    try:
        d = json.loads(res["content"][0]["text"])
    except Exception:
        return "NONJSON", res["content"][0]["text"][:120]
    return ("OK" if d.get("ok") else "OKFALSE"), json.dumps(d, ensure_ascii=False)[:100]


def main() -> int:
    bad = []
    for name, args in CASES:
        status, detail = run_one(name, args)
        flag = "OK  " if status == "OK" else "!!!!"
        print(f"[{flag}] {status:8s} {name:26s} {detail[:90]}", flush=True)
        if status != "OK":
            bad.append((name, status, detail))
    print(f"\n=== {len(CASES) - len(bad)}/{len(CASES)} 正常 ===")
    for n, s, d in bad:
        print(f"  问题: {s}  {n}  {d[:160]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
