# -*- coding: utf-8 -*-
"""dsh-structchem 契约验收测试。

不依赖 pytest，直接跑：
    cd D:\\大肥鱼的实验室\\dsh-structchem
    python tests\\test_contract.py
退出码 0 = 全绿。

设计原则：这里的期望值全部来自教材标准答案 / 群论定理 / 可手算校验的物理量，
**不迁就实现**。实现算不出来就是 FAIL，改实现，不改断言。
"""

from __future__ import annotations

import math
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from structchem_mcp import core  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((label, True, detail or ""))
    except AssertionError as exc:
        RESULTS.append((label, False, f"ASSERT: {exc}"))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((label, False, f"{type(exc).__name__}: {exc}"))
        if os.environ.get("SC_TRACE"):
            traceback.print_exc()


def near(a, b, tol=1e-6, what=""):
    assert abs(float(a) - float(b)) <= tol, f"{what} 期望 {b}, 实得 {a}（容差 {tol}）"
    return f"{what}={a}"


def call(mod: str, fn: str, **kwargs):
    m = core.get_module(mod)
    impl = getattr(m, fn)
    out = impl(**kwargs)
    assert isinstance(out, dict), f"{mod}.{fn} 应返回 dict"
    if out.get("ok") is False:
        raise AssertionError(f"{mod}.{fn} 返回 ok=False: {out.get('error')}")
    return out


def need_steps(out: dict, what: str):
    assert out.get("steps"), f"{what} 缺 steps（本插件的核心价值，不许省）"
    return f"steps×{len(out['steps'])}"


# ---------------------------------------------------------------------------
# 几何夹具
# ---------------------------------------------------------------------------

def _ring(n, r, sym, phase=0.0):
    return [{"sym": sym, "xyz": [r * math.cos(phase + 2 * math.pi * i / n),
                                 r * math.sin(phase + 2 * math.pi * i / n), 0.0]}
            for i in range(n)]


# CH2Cl2：真实 C2v 构型（C2 轴沿 z；两个 Cl 在 yz 平面下方，两个 H 在 xz 平面上方）
# Cl-C-Cl 111.8°、H-C-H 110.5°、C-Cl 1.77 Å、C-H 1.09 Å
CH2CL2 = [
    {"sym": "C", "xyz": [0.0, 0.0, 0.0]},
    {"sym": "Cl", "xyz": [0.0, 1.466, -0.992]},
    {"sym": "Cl", "xyz": [0.0, -1.466, -0.992]},
    {"sym": "H", "xyz": [0.895, 0.0, 0.621]},
    {"sym": "H", "xyz": [-0.895, 0.0, 0.621]},
]
BENZENE = _ring(6, 1.397, "C") + _ring(6, 2.481, "H")
SF6 = [{"sym": "S", "xyz": [0.0, 0.0, 0.0]}] + [
    {"sym": "F", "xyz": v} for v in
    ([1.56, 0, 0], [-1.56, 0, 0], [0, 1.56, 0], [0, -1.56, 0], [0, 0, 1.56], [0, 0, -1.56])
]
CH4 = [{"sym": "C", "xyz": [0.0, 0.0, 0.0]}] + [
    {"sym": "H", "xyz": v} for v in
    ([0.63, 0.63, 0.63], [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63], [-0.63, -0.63, 0.63])
]
BF3 = [{"sym": "B", "xyz": [0.0, 0.0, 0.0]}] + _ring(3, 1.31, "F")
H2O = [{"sym": "O", "xyz": [0.0, 0.0, 0.119]},
       {"sym": "H", "xyz": [0.0, 0.757, -0.477]},
       {"sym": "H", "xyz": [0.0, -0.757, -0.477]}]
TRANS_C2H2CL2 = [
    {"sym": "C", "xyz": [0.68, 0.0, 0.0]}, {"sym": "C", "xyz": [-0.68, 0.0, 0.0]},
    {"sym": "Cl", "xyz": [1.72, 1.60, 0.0]}, {"sym": "Cl", "xyz": [-1.72, -1.60, 0.0]},
    {"sym": "H", "xyz": [1.72, -1.05, 0.0]}, {"sym": "H", "xyz": [-1.72, 1.05, 0.0]},
]
ALLENE = [
    {"sym": "C", "xyz": [0.0, 0.0, 0.0]},
    {"sym": "C", "xyz": [0.0, 0.0, 1.31]}, {"sym": "C", "xyz": [0.0, 0.0, -1.31]},
    {"sym": "H", "xyz": [0.93, 0.0, 1.79]}, {"sym": "H", "xyz": [-0.93, 0.0, 1.79]},
    {"sym": "H", "xyz": [0.0, 0.93, -1.79]}, {"sym": "H", "xyz": [0.0, -0.93, -1.79]},
]
H2O2 = [
    {"sym": "O", "xyz": [0.7375, 0.0, 0.0]}, {"sym": "O", "xyz": [-0.7375, 0.0, 0.0]},
    {"sym": "H", "xyz": [1.0, 0.9, 0.4]}, {"sym": "H", "xyz": [-1.0, -0.9, 0.4]},
]
# 萘 C10H8：两个六元环共用一条边（共用边在 x=0）。环心 (±1.212, 0)，
# 环心到共用边 = 1.40·cos30° = 1.212，C-C 1.40 Å，C-H 1.08 Å 沿径向。
# 全部分子平面 z=0 → D2h。旧坐标把两个环摆成不等价 → 只能测出 C2v（坐标错，非算法错）。
NAPHTHALENE = [
    {"sym": "C", "xyz": [0.0, 0.70, 0.0]}, {"sym": "C", "xyz": [0.0, -0.70, 0.0]},
    {"sym": "C", "xyz": [-1.212, 1.40, 0.0]}, {"sym": "C", "xyz": [-1.212, -1.40, 0.0]},
    {"sym": "C", "xyz": [-2.424, 0.70, 0.0]}, {"sym": "C", "xyz": [-2.424, -0.70, 0.0]},
    {"sym": "C", "xyz": [1.212, 1.40, 0.0]}, {"sym": "C", "xyz": [1.212, -1.40, 0.0]},
    {"sym": "C", "xyz": [2.424, 0.70, 0.0]}, {"sym": "C", "xyz": [2.424, -0.70, 0.0]},
    {"sym": "H", "xyz": [-1.919, 2.216, 0.0]}, {"sym": "H", "xyz": [-1.919, -2.216, 0.0]},
    {"sym": "H", "xyz": [-3.462, 1.000, 0.0]}, {"sym": "H", "xyz": [-3.462, -1.000, 0.0]},
    {"sym": "H", "xyz": [1.919, 2.216, 0.0]}, {"sym": "H", "xyz": [1.919, -2.216, 0.0]},
    {"sym": "H", "xyz": [3.462, 1.000, 0.0]}, {"sym": "H", "xyz": [3.462, -1.000, 0.0]},
]
HCN = [{"sym": "H", "xyz": [0.0, 0.0, -1.06]}, {"sym": "C", "xyz": [0.0, 0.0, 0.0]},
       {"sym": "N", "xyz": [0.0, 0.0, 1.16]}]


# ---------------------------------------------------------------------------
# 1. quantum
# ---------------------------------------------------------------------------

def t_quantum():
    # 势箱：E3/E1 = 9；E1 对 100pm 电子 ≈ 37.6 eV
    r = call("quantum", "particle_box", n=3, L_pm=100.0, dim=1)
    lv = r["levels"]
    assert len(lv) == 3, f"levels 应有 3 项，实得 {len(lv)}"
    e1 = lv[0]["E_eV"]
    e3 = lv[2]["E_eV"]
    near(e3 / e1, 9.0, 1e-6, "E3/E1")
    near(e1, 37.6, 0.5, "E1(eV)")
    near(r["zero_point_eV"], e1, 1e-6, "零点能")
    need_steps(r, "particle_box")

    # 三维简并度：n_x²+n_y²+n_z²=3 → (111) 1 重；=6 → 3 重
    r3 = call("quantum", "particle_box", n=2, L_pm=100.0, dim=3)
    degs = [lv["degeneracy"] for lv in r3["levels"]]
    assert 1 in degs and 3 in degs, f"三维简并度应含 1 与 3，实得 {degs}"

    # 类氢：3p
    h = call("quantum", "hydrogen_orbital", n=3, l=1, m=0)
    assert h["n_radial_nodes"] == 1, "3p 径向节点应为 1"
    assert h["n_angular_nodes"] == 1, "3p 角节面应为 1"
    assert h["n_nodes_total"] == 2, "3p 总节面应为 2"
    near(h["energy_eV"], -1.511, 0.005, "3p 能量")
    # <r> = (n²a0/Z)[1 + ½(1 - l(l+1)/n²)] = 9a0(1+7/18) = 12.5 a0 = 661.5 pm
    near(h["mean_r_pm"], 12.5 * 52.9177, 1.0, "<r> 3p")
    need_steps(h, "hydrogen_orbital")
    h1 = call("quantum", "hydrogen_orbital", n=1, l=0, m=0)
    near(h1["most_probable_r_pm"], 52.92, 0.05, "1s 最可几半径")

    # Slater：Fe 3d → σ=19.75, Z*=6.25
    s = call("quantum", "slater", config="Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2", target="3d")
    near(s["sigma"], 19.75, 0.01, "Fe 3d σ")
    near(s["z_eff"], 6.25, 0.01, "Fe 3d Z*")
    need_steps(s, "slater")

    # 光谱项：d2 → 1G 3F 1D 3P 1S，45 微观状态，基项 3F
    t = call("quantum", "spectroscopic_terms", config="d2")
    names = sorted(x["term"].replace(" ", "") for x in t["terms"])
    assert names == sorted(["1G", "3F", "1D", "3P", "1S"]), f"d2 项集错：{names}"
    assert t["microstates_total"] == 45, f"d2 微观状态应为 45，实得 {t['microstates_total']}"
    assert t.get("check") in (True, "ok", None) or t.get("check"), "微观状态校验未通过"
    assert t["ground_term"].startswith("3F"), f"d2 基项应 3F，实得 {t['ground_term']}"
    need_steps(t, "spectroscopic_terms")

    t6 = call("quantum", "spectroscopic_terms", config="d6")
    assert t6["ground_term"].startswith("5D"), f"d6 基项应 5D，实得 {t6['ground_term']}"
    assert t6["microstates_total"] == 210, "d6 微观状态应为 210"

    p2 = call("quantum", "spectroscopic_terms", config="p2")
    assert p2["microstates_total"] == 15, "p2 微观状态应为 15"
    assert p2["ground_term"].startswith("3P"), "p2 基项应 3P"

    d5 = call("quantum", "spectroscopic_terms", config="d5")
    assert d5["ground_term"].startswith("6S"), f"d5 基项应 6S，实得 {d5['ground_term']}"

    # 叠加：ψ = C1ψ210 + C2ψ311 + C3ψ310
    sup = call("quantum", "superposition", terms=[
        {"coef": 1.0, "n": 2, "l": 1, "m": 0},
        {"coef": 1.0, "n": 3, "l": 1, "m": 1},
        {"coef": 0.0, "n": 3, "l": 1, "m": 0},
    ])
    # 归一化后各占 1/2：E = ½(-3.4) + ½(-1.511) = -2.4555 eV
    near(sup["energy_mean_eV"], -2.4555, 0.01, "叠加能量平均值")
    need_steps(sup, "superposition")
    return "势箱/类氢/Slater/光谱项/叠加 全通过"


# ---------------------------------------------------------------------------
# 2. symmetry
# ---------------------------------------------------------------------------

PG_CASES = [
    ("CH2Cl2", CH2CL2, "C2v"),
    ("苯", BENZENE, "D6h"),
    ("SF6", SF6, "Oh"),
    ("CH4", CH4, "Td"),
    ("BF3", BF3, "D3h"),
    ("H2O", H2O, "C2v"),
    ("反式二氯乙烯", TRANS_C2H2CL2, "C2h"),
    ("丙二烯", ALLENE, "D2d"),
    ("H2O2(扭曲)", H2O2, "C2"),
    ("萘", NAPHTHALENE, "D2h"),
    ("HCN(线形)", HCN, "C*v"),
]


def t_symmetry_point_group():
    fails = []
    for name, atoms, want in PG_CASES:
        r = call("symmetry", "point_group", atoms=atoms)
        got = r["point_group"]
        # 线形点群记号允许 C∞v / Cinfv / C*v 等写法
        if want.startswith("C*"):
            ok = got.replace("∞", "v").replace("inf", "").replace("Cv", "C") in (
                "Cv", "C", "Cv") or "∞" in got or "inf" in got.lower()
        else:
            ok = got == want
        need_steps(r, f"point_group {name}")
        assert r.get("order"), f"{name} 缺 order"
        if not ok:
            fails.append(f"{name}: 期望 {want}, 实得 {got}")
    assert not fails, "；".join(fails)
    return f"{len(PG_CASES)} 个分子点群全对"


def t_symmetry_group_theory():
    # D4h: 16 阶 10 类 10 个不可约表示
    ct = call("symmetry", "character_table", point_group_name="D4h")
    assert ct["order"] == 16, f"D4h 阶应为 16，实得 {ct['order']}"
    assert len(ct["classes"]) == 10, f"D4h 类应为 10，实得 {len(ct['classes'])}"
    assert len(ct["irreps"]) == 10, f"D4h 不可约表示应为 10，实得 {len(ct['irreps'])}"

    # 全点群群论自洽：三条硬判据。
    # 注意**不要求「类数 = 不可约表示数」**——教材把一对共轭一维复表示合并成一个 E
    # （特征标 2cos），此时表示数 < 类数是正确的（S4: 4类3表示、T: 4类3表示、
    # C3h: 6类4表示、Th: 8类6表示、S6/S8 同理）。
    #   ① 全对称表示字符全 1，且 Σ g_c·χ_全对称 = h
    #   ② 不同表示正交：Σ g_c·χ_i·χ_j = 0
    #   ③ 自身归一：非 merged 表示 Σ g_c·χ_i² = h；merged 表示 = 2h
    # 这三条是特征标表唯一的、足够强的验收判据 —— 手抄错一个数必被抓住。
    PGS = ["C1", "Cs", "Ci", "C2", "C3", "C4", "C5", "C6", "C2v", "C3v", "C4v",
           "C5v", "C6v", "C2h", "C3h", "D2", "D3", "D4", "D5", "D6", "D2h",
           "D3h", "D4h", "D5h", "D6h", "D2d", "D3d", "D4d", "D5d", "S4", "S6",
           "S8", "T", "Th", "Td", "O", "Oh", "I", "Ih"]
    checked = 0
    for pg in PGS:
        c = call("symmetry", "character_table", point_group_name=pg)
        h = c["order"]
        cls, irr = c["classes"], c["irreps"]
        sizes = [cl.get("size", 1) for cl in cls]
        assert all(isinstance(s, int) for s in sizes), f"{pg} 类大小应为整数"
        assert sum(sizes) == h, f"{pg} Σ类大小={sum(sizes)} ≠ 阶 {h}"
        tri = irr[0]
        assert all(abs(x - 1) < 1e-9 for x in tri["chars"]), (
            f"{pg} 首个不可约表示 {tri['name']} 不是全对称（应全 1）")
        merged = {i["name"] for i in irr if i.get("merged")}
        n = len(cls)
        for ai in range(len(irr)):
            ca = irr[ai]["chars"]
            assert len(ca) == n, (
                f"{pg}/{irr[ai]['name']} 字符长度 {len(ca)} ≠ 类数 {n}")
            for bi in range(ai, len(irr)):
                cb = irr[bi]["chars"]
                s = sum(sizes[k] * ca[k] * cb[k] for k in range(n))
                if ai == bi:
                    want = 2 * h if irr[ai]["name"] in merged else h
                else:
                    want = 0
                assert abs(s - want) < 1e-5, (
                    f"{pg}: ⟨{irr[ai]['name']}|{irr[bi]['name']}⟩ = {s:g}，应 {want} "
                    f"（差 {s - want:+g} —— 找类大小能整除这个差的那一个类，符号可能反了）")
        checked += 1

    # 约化：C2v 的 Γ = [9,-1,3,1] → 3A1 + A2 + 3B1 + 2B2（总维数 9）
    # 手算核对：(9-1+3+1)/4=3；(9-1-3-1)/4=1；(9+1+3-1)/4=3；(9+1-3+1)/4=2
    r = call("symmetry", "reduce_representation", point_group_name="C2v",
             chars=[9, -1, 3, 1])
    d = {x["irrep"]: x["count"] for x in r["decomposition"] if x["count"]}
    assert d == {"A1": 3, "A2": 1, "B1": 3, "B2": 2}, f"C2v 约化结果错：{d}"
    need_steps(r, "reduce_representation")

    # H2O 振动：3N=9 → Γ_vib = 2A1 + B1
    nm = call("symmetry", "normal_modes", point_group_name="C2v", n_atoms=3)
    v = {x["irrep"]: x["count"] for x in nm["gamma_vib"] if x["count"]} \
        if isinstance(nm["gamma_vib"], list) else nm["gamma_vib"]
    assert v == {"A1": 2, "B1": 1}, f"H2O Γ_vib 应 2A1+B1，实得 {v}"
    need_steps(nm, "normal_modes")

    # SF6：Γ_vib = A1g + Eg + 2T1u + T2g + T2u（15 个振动自由度）
    sf = call("symmetry", "normal_modes", point_group_name="Oh", n_atoms=7)
    vv = {x["irrep"]: x["count"] for x in sf["gamma_vib"] if x["count"]} \
        if isinstance(sf["gamma_vib"], list) else sf["gamma_vib"]
    assert vv == {"A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1}, (
        f"SF6 Γ_vib 错：{vv}")
    # 只有 T1u 是 IR 活性
    assert any("T1u" in str(x) for x in sf["ir_active"]), "SF6 的 IR 活性应含 T1u"

    # 选律：C2v 有偶极矩（沿 C2）、无旋光；D2d 无偶极矩、**也无**旋光；
    # 手性只在纯旋转群（C1 / Cn / Dn / T / O / I）里出现。
    #
    # 勘误：早先这条断言写成「D2d 应有旋光性」是错的 —— D2d 含 2 个 σd 与 S4 映轴，
    # 第二类操作在手，构型必与镜像重合，故 D2d 无旋光性（1,3,5,7-四甲基环辛四烯属 S4，
    # 正是教材用来说明"有 S 轴即无旋光"的经典例子）。
    s1 = call("symmetry", "selection_rules", point_group_name="C2v")
    assert s1["dipole"]["has"] is True, "C2v 应有偶极矩"
    assert s1["chiral"]["has"] is False, "C2v 应无旋光性"
    s2 = call("symmetry", "selection_rules", point_group_name="D2d")
    assert s2["dipole"]["has"] is False, "D2d 应无偶极矩"
    assert s2["chiral"]["has"] is False, "D2d 含 σd 与 S4，应无旋光性"
    s3 = call("symmetry", "selection_rules", point_group_name="Ci")
    assert s3["dipole"]["has"] is False, "Ci 应无偶极矩"
    s3b = call("symmetry", "selection_rules", point_group_name="D3")
    assert s3b["chiral"]["has"] is True, "D3 是纯旋转群，应有旋光性"
    s3c = call("symmetry", "selection_rules", point_group_name="S4")
    assert s3c["chiral"]["has"] is False, "S4 含映轴，应无旋光性"
    # D2 是纯旋转群：无 i、无 σ → 无偶极矩但可有手性（扭曲联苯类）
    s3d = call("symmetry", "selection_rules", point_group_name="D2")
    assert s3d["dipole"]["has"] is False, "D2 应无偶极矩"
    assert s3d["chiral"]["has"] is True, "D2 是纯旋转群，应有旋光性"
    need_steps(s1, "selection_rules")

    # 操作乘法：C2(z)·σ(xy) = i
    # 操作乘法：C2(z)·σ(xy) = i（返回字段是 product_name）
    cp = call("symmetry", "compose_operations", point_group_name="D2h",
              op1="C2(z)", op2="σ(xy)")
    eq = str(cp.get("product_name") or cp.get("equivalent") or "")
    assert eq in ("i", "I", "反演"), f"C2(z)σ(xy) 应等于 i，实得 {eq!r}"
    near(cp["determinant"], -1.0, 1e-9, "乘积行列式")
    assert cp["is_proper"] is False, "i 不是真转动"
    need_steps(cp, "compose_operations")
    return f"特征标表 {checked} 个点群正交性全过 + 约化 + 3N-6 + 选律 + 操作乘法 全对"


# ---------------------------------------------------------------------------
# 3. orbital
# ---------------------------------------------------------------------------

def t_orbital():
    # 苯：α+2β, α+β×2, α-β×2, α-2β；总 π 能 6α+8β；离域能 2β
    b = call("orbital", "huckel", topology="cyclic", n_atoms=6, n_electrons=6)
    xs = sorted((round(o["E_beta"], 4) for o in b["orbitals"]), reverse=True)
    want = sorted([2.0, 1.0, 1.0, -1.0, -1.0, -2.0], reverse=True)
    assert all(abs(a - c) < 1e-3 for a, c in zip(xs, want)), f"苯能级错：{xs}"
    assert abs(abs(b["delocalization_energy"]) - 2.0) < 1e-3, (
        f"苯离域能应 2β，实得 {b['delocalization_energy']}")
    # 电荷密度全 1.0、键级全 2/3（bond_orders 是 [{i,j,order,order_str}]）
    assert all(abs(q - 1.0) < 1e-3 for q in b["charge_densities"]), "苯电荷密度应全 1"
    bo = [x["order"] if isinstance(x, dict) else x for x in b["bond_orders"]]
    assert all(abs(p - 2 / 3) < 1e-3 for p in bo), f"苯键级应全 0.667，实得 {bo}"
    need_steps(b, "huckel")

    # 丁二烯：±1.618, ±0.618
    d = call("orbital", "huckel", topology="linear", n_atoms=4, n_electrons=4)
    xs = sorted((round(o["E_beta"], 3) for o in d["orbitals"]), reverse=True)
    assert all(abs(a - c) < 2e-3 for a, c in
               zip(xs, [1.618, 0.618, -0.618, -1.618])), f"丁二烯能级错：{xs}"
    # 离域能 = (2×1.618 + 2×0.618) - 4 = 4.472-4 = 0.472β
    assert abs(abs(d["delocalization_energy"]) - 0.472) < 5e-3, (
        f"丁二烯离域能应 0.472β，实得 {d['delocalization_energy']}")

    # 环丙烯正离子：α+2β, α-β, α-β，离域能 2β
    c3 = call("orbital", "huckel", topology="cyclic", n_atoms=3, n_electrons=2)
    xs = sorted((round(o["E_beta"], 3) for o in c3["orbitals"]), reverse=True)
    assert all(abs(a - b2) < 1e-3 for a, b2 in zip(xs, [2.0, -1.0, -1.0])), (
        f"环丙烯正离子能级错：{xs}")
    need_steps(c3, "huckel 环丙烯")

    # 芳香性
    a1 = call("orbital", "aromaticity", n_pi=6, n_atoms=6, ring=True)
    assert a1["aromatic"] is True, "苯应芳香"
    a2 = call("orbital", "aromaticity", n_pi=4, n_atoms=4, ring=True)
    assert a2["aromatic"] is False, "环丁二烯应反芳香"

    # 双原子 MO
    o2 = call("orbital", "diatomic_mo", n_electrons=16)
    near(o2["bond_order"], 2.0, 1e-9, "O2 键级")
    assert o2["unpaired_electrons"] == 2, f"O2 应有 2 个单电子，实得 {o2['unpaired_electrons']}"
    assert "顺磁" in str(o2["magnetism"]), "O2 应顺磁"
    n2 = call("orbital", "diatomic_mo", n_electrons=14)
    near(n2["bond_order"], 3.0, 1e-9, "N2 键级")
    assert n2["unpaired_electrons"] == 0, "N2 应无单电子"
    no = call("orbital", "diatomic_mo", n_electrons=15)
    near(no["bond_order"], 2.5, 1e-9, "NO 键级")
    need_steps(o2, "diatomic_mo")

    # VSEPR
    sf4 = call("orbital", "vsepr", central="S",
               ligands=["F", "F", "F", "F"], lone_pairs=1)
    assert "AX4E" in str(sf4["vsepr_type"]), f"SF4 应 AX4E，实得 {sf4['vsepr_type']}"
    assert sf4["point_group"] == "C2v", f"SF4 应 C2v，实得 {sf4['point_group']}"
    xef4 = call("orbital", "vsepr", central="Xe",
                ligands=["F", "F", "F", "F"], lone_pairs=2)
    assert xef4["point_group"] == "D4h", f"XeF4 应 D4h，实得 {xef4['point_group']}"
    need_steps(sf4, "vsepr")
    return "HMO(苯/丁二烯/环丙烯) + 芳香性 + 双原子 MO + VSEPR 全对"


# ---------------------------------------------------------------------------
# 4. crystal
# ---------------------------------------------------------------------------

def t_crystal():
    u = call("crystal", "unit_cell", a=500.0, b=500.0, c=500.0)
    assert "立方" in str(u["crystal_system"]), f"立方判错：{u['crystal_system']}"
    near(u["volume_A3"], 125.0, 1e-6, "立方晶胞体积")
    need_steps(u, "unit_cell")
    h = call("crystal", "unit_cell", a=500.0, b=500.0, c=400.0, gamma=120.0)
    assert "六方" in str(h["crystal_system"]), f"六方判错：{h['crystal_system']}"

    m = call("crystal", "miller_indices", intercepts=["a/2", "b/3", "c/4"])
    near(m["h"], 2, 1e-9, "h")
    near(m["k"], 3, 1e-9, "k")
    near(m["l"], 4, 1e-9, "l")
    assert str(m["miller"]).replace(" ", "") == "(234)", f"晶面指标错：{m['miller']}"
    need_steps(m, "miller_indices")

    d = call("crystal", "d_spacing", h=1, k=0, l=0, a=400.0)
    near(d["d_pm"], 400.0, 1e-6, "立方 d(100)")
    d111 = call("crystal", "d_spacing", h=1, k=1, l=1, a=400.0)
    near(d111["d_pm"], 400.0 / math.sqrt(3), 1e-6, "立方 d(111)")

    f = call("crystal", "packing", kind="fcc")
    assert f["atoms_per_cell"] == 4, "fcc 每胞 4 原子"
    assert f["coordination_number"] == 12, "fcc 配位数 12"
    near(float(str(f["packing_fraction"]).rstrip("%")) / 100, 0.7405, 1e-3, "fcc 空间利用率")
    b = call("crystal", "packing", kind="bcc")
    near(float(str(b["packing_fraction"]).rstrip("%")) / 100, 0.6802, 1e-3, "bcc 空间利用率")

    # 半径比：KBr 经典陷阱 —— 题目只给 r(K+)=133pm 与键长 328pm，
    # 必须自己减出 r(Br-) = 328 - 133 = 195pm，再算 133/195 = 0.682 → 八面体 CN=6。
    # （若误把 328 当 r-，得 0.405 → 会错判成四面体。南开真题第 14 题答案就是八面体。）
    r = call("crystal", "radius_ratio", r_plus_pm=133.0, r_minus_pm=195.0)
    near(r["ratio"], 0.682, 1e-3, "KBr 半径比")
    assert r["coordination_number"] == 6, (
        f"r+/r-=0.682 应 CN=6，实得 {r['coordination_number']}")
    assert "八面体" in str(r["polyhedron"]), f"配位多面体应八面体，实得 {r['polyhedron']}"
    need_steps(r, "radius_ratio")
    # 边界正确性：0.4055 落在 0.225–0.414 → 四面体
    r2 = call("crystal", "radius_ratio", r_plus_pm=133.0, r_minus_pm=328.0)
    assert r2["coordination_number"] == 4, (
        f"r+/r-=0.4055 应 CN=4，实得 {r2['coordination_number']}")
    return "晶系/晶面指标/晶面间距/堆积/半径比 全对"


# ---------------------------------------------------------------------------
# 5. coordination
# ---------------------------------------------------------------------------

def t_coordination():
    d6w = call("coordination", "crystal_field", d_n=6, geometry="octahedral",
               pairing_energy_cm1=21000.0, delta_o_cm1=10000.0)
    assert d6w["unpaired_electrons"] == 4, f"d6 弱场应 4 单电子，实得 {d6w['unpaired_electrons']}"
    near(d6w["magnetic_moment_BM"], math.sqrt(24), 0.01, "d6 弱场磁矩")
    need_steps(d6w, "crystal_field")

    d6s = call("coordination", "crystal_field", d_n=6, geometry="octahedral",
               pairing_energy_cm1=15000.0, delta_o_cm1=30000.0)
    assert d6s["unpaired_electrons"] == 0, "d6 强场应 0 单电子"
    assert abs(d6s["magnetic_moment_BM"]) < 1e-6, "d6 强场应抗磁"
    assert "低自旋" in str(d6s["spin_state"]), "d6 强场应低自旋"

    d5 = call("coordination", "crystal_field", d_n=5, geometry="octahedral",
              pairing_energy_cm1=30000.0, delta_o_cm1=10000.0)
    assert d5["unpaired_electrons"] == 5, "d5 弱场应 5 单电子"
    near(d5["magnetic_moment_BM"], math.sqrt(35), 0.01, "d5 磁矩")

    s = call("coordination", "spectrochemical_series", ligand="CN-")
    assert s.get("ok", True), "光谱化学序列查询失败"
    return "晶体场 CFSE/自旋态/磁矩 + 光谱化学序列 全对"


# ---------------------------------------------------------------------------
# 6. 数据层
# ---------------------------------------------------------------------------

def t_data_layer():
    import json
    ddir = os.path.join(_ROOT, "src", "structchem_mcp", "data")
    problems = []
    for name, key, minimum in [
        ("answer_cards.json", "cards", 20),
        ("rationale.json", "dims", 15),
        ("book_index.json", "chapters", 10),
        ("character_tables.json", None, 25),
    ]:
        path = os.path.join(ddir, name)
        assert os.path.exists(path), f"缺数据文件 {name}"
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if key is None:
            n = len(data) if isinstance(data, dict) else len(data)
        else:
            n = len(data.get(key, []))
            if n < minimum:
                problems.append(f"{name}: {key} 只有 {n} 条（要求 ≥{minimum}）")
        if key is not None and n < minimum:
            continue
    assert not problems, "；".join(problems)

    # 检索通路
    card = _tool("sc_answer_card", "点群")
    assert card["ok"] and card["count"] >= 1, "answer_card 检索不到「点群」"
    rat = _tool("sc_rationale", "偶极矩")
    assert rat["ok"] and rat["count"] >= 1, "rationale 检索不到「偶极矩」"
    bi = _tool("sc_book_index", "4")
    assert bi["ok"] and bi["count"] >= 1, "book_index 检索不到第 4 章"
    return "数据文件齐备 + 检索通路通"


def _tool(name, *args):
    """直接调 server 层的工具函数（不走 MCP 协议）。"""
    import importlib
    srv = importlib.import_module("structchem_mcp.server")
    fn = getattr(srv, name)
    return fn(*args)


# ---------------------------------------------------------------------------
# 7. skill 包完整性
# ---------------------------------------------------------------------------

def t_skill():
    skdir = os.path.join(_ROOT, "skills", "structural-chem-answering")
    main = os.path.join(skdir, "SKILL.md")
    assert os.path.exists(main), "缺 SKILL.md"
    with open(main, encoding="utf-8") as fh:
        text = fh.read()
    assert text.startswith("---"), "SKILL.md 应有 YAML frontmatter"
    assert "name:" in text.split("---")[1], "frontmatter 缺 name"
    assert "description:" in text.split("---")[1], "frontmatter 缺 description"
    assert len(text.splitlines()) >= 250, f"SKILL.md 太薄：{len(text.splitlines())} 行"

    # 16 个工具名必须都被 skill 提到，否则模型调不到
    tools = ["sc_particle_box", "sc_hydrogen_orbital", "sc_slater",
             "sc_spectroscopic_terms", "sc_superposition", "sc_point_group",
             "sc_symmetry_elements", "sc_character_table", "sc_reduce_representation",
             "sc_normal_modes", "sc_selection_rules", "sc_compose_operations",
             "sc_huckel", "sc_aromaticity", "sc_diatomic_mo", "sc_vsepr",
             "sc_unit_cell", "sc_miller_indices", "sc_d_spacing", "sc_packing",
             "sc_radius_ratio", "sc_cell_content", "sc_crystal_field",
             "sc_spectrochemical_series", "sc_notation"]
    missing = [t for t in tools if t not in text]
    assert not missing, f"SKILL.md 未提及工具：{missing}"

    refs = os.path.join(skdir, "references")
    assert os.path.isdir(refs), "缺 references/ 目录"
    md = [f for f in os.listdir(refs) if f.endswith(".md")]
    assert len(md) >= 5, f"references 只有 {len(md)} 册（建议 ≥6）"
    total = 0
    for f in md:
        with open(os.path.join(refs, f), encoding="utf-8") as fh:
            total += len(fh.read().splitlines())
    assert total >= 600, f"references 总行数只有 {total}（要求 ≥600）"
    assert os.path.exists(os.path.join(skdir, "meta.yaml")), "缺 meta.yaml"
    return f"SKILL.md {len(text.splitlines())} 行 + references {len(md)} 册 / {total} 行"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("dsh-structchem 契约验收")
    print("=" * 72)

    avail = core.availability()
    print("\n-- 模块健康 --")
    for name, info in avail.items():
        flag = "OK  " if info["ok"] else "FAIL"
        extra = "" if info["ok"] else f"   {info['error']}"
        print(f"  [{flag}] {name:14s} {len(info['functions'])} funcs{extra}")

    print("\n-- 断言 --")
    check("quantum 全组", t_quantum)
    check("symmetry 点群判定", t_symmetry_point_group)
    check("symmetry 群论", t_symmetry_group_theory)
    check("orbital 全组", t_orbital)
    check("crystal 全组", t_crystal)
    check("coordination 全组", t_coordination)
    check("数据层", t_data_layer)
    check("skill 包", t_skill)

    print()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for label, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"\n        {detail}" if detail else ""))
    print(f"\n{passed}/{len(RESULTS)} 通过")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
