# -*- coding: utf-8 -*-
"""quantum.py — 量子化学基础与原子结构（Q1–Q5）。

本模块只做**可确定性计算**：

- Q1 ``particle_box``        势箱粒子能级、简并度、零点能、节点数、跃迁波长
- Q2 ``hydrogen_orbital``    氢原子/类氢轨道的节面、能量、角动量、最可几半径、<r>
- Q3 ``slater``              Slater 规则屏蔽常数与有效核电荷、轨道能量
- Q4 ``spectroscopic_terms`` Russell-Saunders 谱项（等价电子剥离法 + 非同科耦合）
- Q5 ``superposition``       态叠加的能量/角动量平均值与测量几率

不做的事：多电子变分与 SCF、含时演化、跃迁强度/振子强度、精细结构的定量计算、
相对论修正。这些超出课程答题的确定性范围，宁可报错也不给 plausible-but-wrong 的数。

教材口径（《结构化学基础》周公度·段连运 第 5 版）：

1. 氢原子能量主值取教材常数 ``13.6 eV``（R∞ 精确值 13.6057 eV，差 0.04%）；
   两套值都返回，主值走教材口径，CODATA 值放 ``*_codata`` 字段。
2. Slater 规则 n* 表：1→1.0，2→2.0，3→3.0，4→3.7，5→4.0，6→4.2，7→4.3，8→4.4。
3. Slater 分组：ns 与 np 同组；nd、nf、ng 各自单独成组；组序按 (n, 0/2/3/4)。
4. 谱项用「最大 S 优先的逐项剥离法」（Hund 剥离法）真实枚举微观状态后剥离，
   并用 Σ(2S+1)(2L+1) = ∏C(2(2l+1), k) 自校验，绝不查表冒充计算。
5. 长度默认 pm，能量默认 eV，同时给 cm-1 / kJ/mol / a.u. 换算。
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # 正常路径：作为包内模块导入
    from . import _const as C
except ImportError:  # pragma: no cover - 允许 `python quantum.py` 直接运行
    import os
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
    from structchem_mcp.core import _const as C

__all__ = [
    "particle_box",
    "hydrogen_orbital",
    "slater",
    "spectroscopic_terms",
    "superposition",
]

# ---------------------------------------------------------------------------
# 教材口径常数
# ---------------------------------------------------------------------------

#: 《结构化学基础》教材用的氢原子基态能量绝对值（eV）。
#: CODATA 精确值见 C.RYDBERG_EV = 13.6057，两者差 0.04%，答题按教材写 13.6。
RY_EV_TEXTBOOK = 13.6

#: Slater 规则的有效主量子数 n*（教材表）。
N_STAR_TABLE: Dict[int, float] = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 3.7,
    5: 4.0,
    6: 4.2,
    7: 4.3,
    8: 4.4,
}

_SUBLETTERS = "spdfghiklm"
_SUBL_INDEX = {ch: i for i, ch in enumerate(_SUBLETTERS)}

#: L 值 → 谱项字母（跳过已被占用的 J，按 S,P,D,F,G,H,I,K,L,M,N,O,Q,R,T,U,V 排）
_L_LETTER = "SPDFGHIKLMNOQRTUV"

#: 原子序数表（元素符号 → Z），1–92。这是周期表数据，不是答案表。
_ELEMENTS: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,
}

#: 粒子质量表（kg，CODATA 2018）。electron 取 _const。
_MASS_KG: Dict[str, Tuple[float, str]] = {
    "electron": (C.ME, "电子 m_e"),
    "e": (C.ME, "电子 m_e"),
    "me": (C.ME, "电子 m_e"),
    "proton": (1.67262192369e-27, "质子"),
    "p": (1.67262192369e-27, "质子"),
    "neutron": (1.67492749804e-27, "中子"),
    "n": (1.67492749804e-27, "中子"),
    "hydrogen": (1.6735575392e-27, "氢原子"),
    "deuteron": (3.3435837724e-27, "氘核"),
    "alpha": (6.6446573357e-27, "α 粒子"),
    "电子": (C.ME, "电子 m_e"),
    "质子": (1.67262192369e-27, "质子"),
    "中子": (1.67492749804e-27, "中子"),
}

_SUPERSCRIPT_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_UNSUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _fail(msg: str) -> Dict[str, Any]:
    """统一的失败返回。"""
    return {"ok": False, "error": msg}


def _ok(**kw: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"ok": True, "error": None}
    d.update(kw)
    return d


def _pow10(exp: int) -> str:
    return "10" + str(int(exp)).translate(_SUPERSCRIPT_MAP)


def _sci(x: float, nd: int = 4) -> str:
    """科学计数法字符串，形如 6.0246×10⁻¹⁸。"""
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10.0 ** exp)
    if abs(mant) >= 10.0:
        mant /= 10.0
        exp += 1
    return f"{mant:.{nd}f}×{_pow10(exp)}"


def _fmt(x: float, nd: int = 4) -> str:
    """普通量用有效数字，极端量级自动切科学计数法。"""
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e5:
        return f"{x:.{nd}g}"
    return _sci(x, nd)


def _norm_uni(s: str) -> str:
    """把上标数字、全角标点等归一化，便于解析教材里抄来的式子。"""
    return str(s).translate(_UNSUPERSCRIPT_MAP)


def _safe_numeric(text: str) -> float:
    """解析 '1/√2'、'0.8'、'sqrt(2)/2' 这类系数写法。"""
    s = _norm_uni(text).strip()
    s = s.replace("√", "sqrt").replace("×", "*").replace("^", "**")
    s = re.sub(r"sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    probe = s.replace("sqrt", "")
    if not set(probe) <= set("0123456789.+-*/() "):
        raise ValueError(f"看不懂的系数写法：{text}")
    return float(eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt}))  # noqa: S307


# ===========================================================================
# Q1  particle_box —— 势箱中的粒子
# ===========================================================================


def _resolve_mass(mass: Any) -> Tuple[Optional[float], str, str]:
    """质量解析。返回 (kg, 说明, 错误)。"""
    if isinstance(mass, dict):
        if "value" not in mass:
            return None, "", 'mass 传字典时必须给 {"value": 数值, "unit": "kg"|"me"|"amu"}'
        try:
            val = float(mass["value"])
        except (TypeError, ValueError):
            return None, "", "mass.value 必须是数字"
        unit = str(mass.get("unit", "me")).strip().lower()
        if unit in ("kg", "千克"):
            kg = val
        elif unit in ("me", "m_e", "electron", "电子质量"):
            kg = val * C.ME
        elif unit in ("amu", "u"):
            kg = val * 1.66053906660e-27
        else:
            return None, "", f"mass.unit 只支持 kg / me / amu，收到 {unit}"
        if kg <= 0:
            return None, "", "质量必须 > 0"
        return kg, f"{val:g} {unit}（= {_sci(kg)} kg）", ""

    if isinstance(mass, str):
        key = mass.strip().lower()
        if key in _MASS_KG:
            kg, name = _MASS_KG[key]
            return kg, f"{name} = {_sci(kg)} kg", ""
        try:
            val = float(key)
        except ValueError:
            return None, "", (
                f"认不出质量 {mass!r}；可用 electron/proton/neutron，"
                '或 {"value": 数值, "unit": "kg"|"me"|"amu"}'
            )
        if val <= 0:
            return None, "", "质量必须 > 0"
        return val * C.ME, f"{val:g} m_e（按电子质量倍数解释）= {_sci(val * C.ME)} kg", ""

    if isinstance(mass, (int, float)) and not isinstance(mass, bool):
        val = float(mass)
        if val <= 0:
            return None, "", "质量必须 > 0"
        return (
            val * C.ME,
            f"{val:g} m_e（数字按电子质量倍数解释；要 kg 请用 "
            '{"value": ..., "unit": "kg"}）',
            "",
        )

    return None, "", "mass 只接受字符串（electron/proton/...）、字典或数字"


def _enumerate_levels(
    dim: int, n_hi: int, edges_m: Sequence[float], unit_e_J: float
) -> List[Dict[str, Any]]:
    """枚举 dim 维势箱能级并按能量升序分组。

    E = (h²/8m) Σ_i n_i²/L_i²，等边长时化为 E = (h²/8m)(Σn_i²)/L²。
    同一能量（相对差 < 1e-9）的 (n_x, n_y, ...) 组合并成一组，组内组合数即简并度。
    """
    combos: List[Tuple[int, ...]] = []
    if dim == 1:
        combos = [(i,) for i in range(1, n_hi + 1)]
    elif dim == 2:
        combos = [(i, j) for i in range(1, n_hi + 1) for j in range(1, n_hi + 1)]
    else:
        combos = [
            (i, j, k)
            for i in range(1, n_hi + 1)
            for j in range(1, n_hi + 1)
            for k in range(1, n_hi + 1)
        ]

    buckets: List[Dict[str, Any]] = []
    for combo in combos:
        e_j = unit_e_J * sum((ni * ni) / (edges_m[axis] ** 2) for axis, ni in enumerate(combo))
        for bucket in buckets:
            if abs(e_j - bucket["E_J"]) <= 1e-9 * abs(bucket["E_J"]):
                bucket["states"].append(combo)
                break
        else:
            buckets.append({"E_J": e_j, "states": [combo]})

    buckets.sort(key=lambda b: b["E_J"])

    levels: List[Dict[str, Any]] = []
    for idx, bucket in enumerate(buckets, 1):
        e_j = bucket["E_J"]
        e_ev = e_j * C.J_TO_EV
        states = bucket["states"]
        first = states[0]
        label = f"n={first[0]}" if dim == 1 else "(" + ",".join(str(x) for x in first) + ")"
        n_sq = sum(x * x for x in first)
        levels.append(
            {
                "index": idx,
                "label": label,
                "n_tuple": list(first),
                "n_squared": n_sq if dim >= 1 else None,
                "states": ["(" + ",".join(str(x) for x in s) + ")" for s in states],
                "degeneracy": len(states),
                "E_J": e_j,
                "E_eV": e_ev,
                "E_str": f"{_fmt(e_ev, 5)} eV",
                "E_cm1": C.ev_to_cm1(e_ev),
            }
        )
    return levels


def particle_box(
    n: int,
    L_pm: Any,
    dim: int = 1,
    mass: Any = "electron",
    electrons: Optional[int] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
) -> Dict[str, Any]:
    """一/二/三维势箱中粒子的能级、简并度、零点能、节点数与跃迁波长。

    参数
    ----
    n : int
        主量子数上限。dim=1 时列出 n=1..n；dim≥2 时各维量子数取 1..n。
    L_pm : float | Sequence[float]
        箱长（pm）。给单个数字 = 立方箱/一维箱；给长度等于 dim 的序列 = 长方体。
    dim : {1,2,3}
    mass : str | float | dict
        "electron"/"proton"/"neutron"…；数字按电子质量倍数解释；
        dict 形式 {"value": 1.0, "unit": "kg"|"me"|"amu"}。
    electrons : int | None
        电子数。给了就按能量由低到高每能级 2 个填充，跃迁自动取 HOMO→LUMO。
    n1, n2 : int | None
        显式指定跃迁两端。dim=1 时按主量子数取值，dim≥2 时按能级序号（1 起）。

    返回
    ----
    levels / zero_point_eV / nodes / transition / steps 等。

    边界：不做相对论修正、不做有限深势阱、不处理微扰。n 过大（dim≥2 时 > 12）
    会因组合爆炸被拒绝。
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return _fail(f"n 必须是正整数（主量子数上限），收到 {n!r}")
    if n < 1:
        return _fail(f"n 必须 ≥ 1，收到 {n}")
    try:
        dim = int(dim)
    except (TypeError, ValueError):
        return _fail(f"dim 必须是 1、2 或 3，收到 {dim!r}")
    if dim not in (1, 2, 3):
        return _fail(f"dim 只能是 1、2、3，收到 {dim}")

    # 边长
    if isinstance(L_pm, (list, tuple)):
        try:
            edges_pm = [float(x) for x in L_pm]
        except (TypeError, ValueError):
            return _fail("L_pm 序列里出现了非数字，边长必须是正数（pm）")
        if len(edges_pm) != dim:
            return _fail(f"L_pm 给了 {len(edges_pm)} 个边长，但 dim={dim} 需要 {dim} 个")
    else:
        try:
            edge = float(L_pm)
        except (TypeError, ValueError):
            return _fail(f"L_pm 必须是正数（pm），收到 {L_pm!r}")
        edges_pm = [edge] * dim
    if any(x <= 0 for x in edges_pm):
        return _fail(f"箱长必须 > 0（单位 pm），收到 {edges_pm}")

    m_kg, mass_note, mass_err = _resolve_mass(mass)
    if m_kg is None:
        return _fail(mass_err)

    n_hi = n
    if n1 is not None:
        try:
            n_hi = max(n_hi, int(n1))
        except (TypeError, ValueError):
            return _fail(f"n1 必须是整数，收到 {n1!r}")
    if n2 is not None:
        try:
            n_hi = max(n_hi, int(n2))
        except (TypeError, ValueError):
            return _fail(f"n2 必须是整数，收到 {n2!r}")

    if dim >= 2 and n_hi > 12:
        return _fail(
            f"dim={dim} 时各维量子数上限最多 12（组合数 {n_hi ** dim} 会爆），"
            "要算更高能级请分开传入或只用 dim=1"
        )
    if dim == 1 and n_hi > 200:
        return _fail("dim=1 时 n 最多 200")

    # 电子填充可能需要更多能级
    if electrons is not None:
        try:
            electrons = int(electrons)
        except (TypeError, ValueError):
            return _fail(f"electrons 必须是整数，收到 {electrons!r}")
        if electrons < 1:
            return _fail(f"electrons 必须 ≥ 1，收到 {electrons}")
        if dim >= 2:
            while True:
                probe = _enumerate_levels(
                    dim, n_hi, [x * 1e-12 for x in edges_pm], C.H * C.H / (8.0 * m_kg)
                )
                if 2 * len(probe) >= electrons or n_hi >= 12:
                    break
                n_hi += 1
        elif 2 * n_hi < electrons and n_hi < 200:
            n_hi = min(200, (electrons + 1) // 2)

    unit_e_J = (C.H * C.H) / (8.0 * m_kg)  # J·m²
    edges_m = [x * 1e-12 for x in edges_pm]
    levels = _enumerate_levels(dim, n_hi, edges_m, unit_e_J)

    zero = levels[0]
    zero_point_ev = zero["E_eV"]

    # 节点数：一维为 n-1；多维为各维节点数之和 (n_x-1)+(n_y-1)+...
    if dim == 1:
        nodes = n - 1
        nodes_basis = f"一维：节点数 = n-1 = {n-1}（对应输入 n={n}）"
    else:
        nodes = sum(x - 1 for x in zero["n_tuple"])
        nodes_basis = (
            f"{dim} 维：最低能级 {zero['label']} 的节面数 = "
            "Σ(n_i-1) = " + "+".join(str(x - 1) for x in zero["n_tuple"]) + f" = {nodes}"
        )

    steps: List[str] = []
    steps.append("势箱粒子能级：E = n²h²/(8mL²)（一维，n = 1,2,3…）")
    if dim >= 2:
        steps.append(
            f"{dim} 维箱：E = (h²/8m)·Σ_i (n_i²/L_i²)，"
            "简并度 = 满足同一 Σ(n_i²/L_i²) 的 (n_x,n_y,n_z) 正整数组合数"
        )
    steps.append(
        f"代入常数：h = {_sci(C.H)} J·s，m = {_sci(m_kg)} kg（{mass_note}），"
        f"L = {', '.join(_fmt(x, 6) + ' pm' for x in edges_pm)}"
    )
    l0 = edges_pm[0]
    eight_ml2 = 8.0 * m_kg * (l0 * 1e-12) ** 2
    h2 = C.H * C.H
    steps.append(
        f"8mL² = 8 × {_sci(m_kg)} kg × ({_sci(l0 * 1e-12)} m)² = {_sci(eight_ml2)} J·s²"
    )
    steps.append(f"h² = ({_sci(C.H)})² = {_sci(h2)} J²·s²")
    unit_ev = (h2 / eight_ml2) * C.J_TO_EV
    steps.append(
        f"h²/(8mL²) = {_sci(h2)} / {_sci(eight_ml2)} = {_sci(h2 / eight_ml2)} J "
        f"= {_fmt(unit_ev, 5)} eV，即 E_n = n² × {_fmt(unit_ev, 5)} eV（一维）"
    )
    steps.append(
        f"零点能 E₁ = 1² × {_fmt(unit_ev, 5)} = {_fmt(zero_point_ev, 5)} eV"
        f"（= {_sci(zero['E_J'])} J）—— 粒子不可能静止，这是不确定关系的直接后果"
    )
    if dim == 1 and n_hi > 1:
        steps.append(
            f"校验：E_{n_hi}/E_1 = {n_hi}²/1² = {n_hi * n_hi}"
        )

    # ---- 跃迁 ----
    transition: Optional[Dict[str, Any]] = None
    trans_err = ""
    if dim == 1:
        by_n = {lv["n_tuple"][0]: lv for lv in levels}
    else:
        by_n = {lv["index"]: lv for lv in levels}

    homo_idx = lumo_idx = None
    mode = ""
    if electrons is not None:
        capacity = 2 * len(levels)
        if electrons > capacity:
            return _fail(
                f"{electrons} 个电子放不进 {len(levels)} 个能级（容量 {capacity}），"
                "加大 n 或减小 electrons"
            )
        occupied = (electrons + 1) // 2  # 最后一个有电子的能级序号（1 起）
        homo_idx = occupied
        lumo_idx = occupied + 1 if occupied < len(levels) else None
        mode = "homo_lumo"
    elif n1 is not None and n2 is not None:
        try:
            homo_idx, lumo_idx = int(n1), int(n2)
        except (TypeError, ValueError):
            return _fail("n1 / n2 必须是整数")
        mode = "explicit"
    else:
        homo_idx, lumo_idx = 1, 2
        mode = "default(最低两能级 n=1→2)"

    if lumo_idx is None or homo_idx is None:
        trans_err = "HOMO 已是最高的枚举能级，没有 LUMO 可跃迁；加大 n 或 electrons"
    elif homo_idx not in by_n:
        trans_err = f"跃迁起点 {homo_idx} 不在已枚举能级内（当前范围见 levels）"
    elif lumo_idx not in by_n:
        trans_err = f"跃迁终点 {lumo_idx} 不在已枚举能级内（当前范围见 levels）"
    elif homo_idx == lumo_idx:
        trans_err = "n1 与 n2 相同，没有跃迁"
    else:
        key_a, key_b = homo_idx, lumo_idx
        a, b = by_n[key_a], by_n[key_b]
        if b["E_eV"] < a["E_eV"]:  # 保证 E₂ > E₁，ΔE 为正
            a, b = b, a
            key_a, key_b = key_b, key_a
        d_e = b["E_eV"] - a["E_eV"]
        wl_nm = C.ev_to_nm(d_e)
        transition = {
            "mode": mode,
            "n1": key_a,
            "n2": key_b,
            "from_level": a["label"],
            "to_level": b["label"],
            "from_E_eV": a["E_eV"],
            "to_E_eV": b["E_eV"],
            "dE_eV": d_e,
            "dE_str": f"{_fmt(d_e, 5)} eV",
            "dE_J": d_e / C.J_TO_EV,
            "dE_cm1": C.ev_to_cm1(d_e),
            "wavelength_nm": wl_nm,
            "wavelength_str": f"{_fmt(wl_nm, 5)} nm",
            "steps": [
                "跃迁能量 ΔE = E₂ - E₁",
                f"ΔE = {_fmt(b['E_eV'], 5)} - {_fmt(a['E_eV'], 5)} = {_fmt(d_e, 5)} eV",
                "对应光子波长 λ = hc/ΔE = 1239.84/ΔE(eV) nm",
                f"λ = 1239.84 / {_fmt(d_e, 5)} = {_fmt(wl_nm, 5)} nm"
                f"（= {_fmt(wl_nm * 1000.0, 6)} pm）",
            ],
        }
        if mode.startswith("default"):
            transition["steps"].insert(
                0,
                "未指定跃迁两端（也没给 electrons），默认取最低两能级；"
                "要算别的跃迁请传 n1/n2 或 electrons",
            )

    return _ok(
        dim=dim,
        n=n,
        n_hi=n_hi,
        edges_pm=edges_pm,
        edges_A=[x / C.PM_PER_A for x in edges_pm],
        mass_kg=m_kg,
        mass_note=mass_note,
        unit_energy_J=h2 / eight_ml2,
        unit_energy_eV=unit_ev,
        levels=levels,
        zero_point_eV=zero_point_ev,
        zero_point_J=zero["E_J"],
        zero_point_str=f"{_fmt(zero_point_ev, 5)} eV",
        nodes=nodes,
        nodes_basis=nodes_basis,
        transition=transition,
        transition_error=trans_err or None,
        electrons=electrons,
        steps=steps,
    )


# ===========================================================================
# Q2  hydrogen_orbital —— 氢原子 / 类氢轨道性质
# ===========================================================================


def _laguerre(k: int, alpha: int, x: float) -> float:
    """广义拉盖尔多项式 L^alpha_k(x)（三项递推，避免阶乘爆掉）。"""
    if k == 0:
        return 1.0
    if k == 1:
        return 1.0 + alpha - x
    lm1, lcur = 1.0, 1.0 + alpha - x
    for i in range(1, k):
        lnxt = ((2 * i + 1 + alpha - x) * lcur - (i + alpha) * lm1) / (i + 1)
        lm1, lcur = lcur, lnxt
    return lcur


def _radial_R(n: int, l: int, r_bohr: float) -> float:
    """氢原子径向波函数 R_nl(r)，r 以 a0 为单位，返回单位为 a0^(-3/2)。"""
    rho = 2.0 * r_bohr / n
    norm = math.sqrt(
        (2.0 / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    return norm * math.exp(-rho / 2.0) * (rho ** l) * _laguerre(n - l - 1, 2 * l + 1, rho)


def _radial_density(n: int, l: int, r_bohr: float) -> float:
    """径向分布函数 D(r) = r²R²(r)（未归一化的形状，用于找极大）。"""
    r_val = _radial_R(n, l, r_bohr)
    return r_bohr * r_bohr * r_val * r_val


def _scan_peak(func, lo: float, hi: float, nstep: int) -> Tuple[float, float]:
    """在 [lo, hi] 上粗扫找极大，再用三点抛物线精修。返回 (r, 值)。"""
    h = (hi - lo) / nstep
    best_i, best_v = 0, -1.0
    vals = []
    for i in range(nstep + 1):
        v = func(lo + i * h)
        vals.append(v)
        if v > best_v:
            best_v, best_i = v, i
    if 0 < best_i < nstep:
        y0, y1, y2 = vals[best_i - 1], vals[best_i], vals[best_i + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            shift = 0.5 * (y0 - y2) / den * h
            r_ref = lo + best_i * h + shift
            return r_ref, func(r_ref)
    return lo + best_i * h, best_v


def _most_probable_r_bohr(n: int, l: int) -> Tuple[float, str]:
    """径向分布函数 r²R²nl 取极大的 r（a0 单位）。1s 直接给 a0 的解析结果。"""
    if n == 1 and l == 0:
        return 1.0, "1s 解析结果：r²R²₁₀ 的极大在 r = a₀"
    r_top = 2.0 * n * n + 12.0 * n
    r_coarse, _ = _scan_peak(lambda r: _radial_density(n, l, r), 1e-4, r_top, 4000)
    h = r_top / 4000.0
    lo = max(1e-6, r_coarse - 4.0 * h)
    hi = r_coarse + 4.0 * h
    r_fine, _ = _scan_peak(lambda r: _radial_density(n, l, r), lo, hi, 4000)
    return r_fine, "数值扫描 r²R²nl 的极大（粗扫 4000 点 + 三点抛物线精修）"


def hydrogen_orbital(n: int, l: int, m: int = 0) -> Dict[str, Any]:
    """氢原子（类氢）轨道 n,l,m 的节面、能量、角动量、最可几半径与平均半径。

    参数
    ----
    n, l, m : int
        主量子数、角量子数、磁量子数。要求 n ≥ 1，0 ≤ l ≤ n-1，|m| ≤ l。

    返回
    ----
    n_radial_nodes / n_angular_nodes / n_nodes_total / energy_eV / energy_au /
    abs_L / Lz / degeneracy / most_probable_r_pm / mean_r_pm / steps 等。

    边界：只给单电子（类氢）结论；多电子原子请用 ``slater``。最可几半径对
    n ≥ 2 用数值扫描径向分布函数求极大；1s 用解析值 a₀。不做相对论修正。
    """
    try:
        n, l, m = int(n), int(l), int(m)
    except (TypeError, ValueError):
        return _fail(f"n / l / m 必须是整数，收到 n={n!r}, l={l!r}, m={m!r}")
    if n < 1:
        return _fail(f"主量子数 n 必须 ≥ 1，收到 {n}")
    if l < 0 or l > n - 1:
        return _fail(
            f"角量子数必须满足 0 ≤ l ≤ n-1；n={n} 时 l 只能取 0..{n-1}，收到 l={l}"
        )
    if abs(m) > l:
        return _fail(f"磁量子数必须满足 |m| ≤ l；l={l} 时 m 只能取 -{l}..{l}，收到 m={m}")

    n_radial_nodes = n - l - 1
    n_angular_nodes = l
    n_nodes_total = n - 1

    energy_ev = -RY_EV_TEXTBOOK / (n * n)
    energy_ev_codata = -C.RYDBERG_EV / (n * n)
    energy_au = -1.0 / (2.0 * n * n)

    abs_l = math.sqrt(l * (l + 1))
    lz = float(m)

    r_mp_bohr, r_mp_method = _most_probable_r_bohr(n, l)
    r_mp_pm = r_mp_bohr * C.A0_PM
    mean_r_bohr = n * n * (1.0 + 0.5 * (1.0 - l * (l + 1) / (n * n)))
    mean_r_pm = mean_r_bohr * C.A0_PM

    orbital = f"{n}{_SUBLETTERS[l]}"

    steps: List[str] = []
    steps.append(
        f"量子数：n={n}，l={l}（{orbital}），m={m}；"
        f"约束 l ≤ n-1 = {n-1} 满足，|m| ≤ l = {l} 满足"
    )
    steps.append(
        f"节面：径向节点数 = n-l-1 = {n}-{l}-1 = {n_radial_nodes}；"
        f"角节面数 = l = {n_angular_nodes}；"
        f"总节面数 = (n-l-1) + l = n-1 = {n_nodes_total}"
    )
    steps.append(
        f"能量：E = -13.6/n² eV = -13.6/{n}² = {_fmt(energy_ev, 5)} eV"
        f"（教材口径；R∞ 精确值给 {_fmt(energy_ev_codata, 5)} eV）"
    )
    steps.append(
        f"同一能量的原子单位式：E = -1/(2n²) a.u. = -1/(2×{n}²) = -1/{2*n*n}"
        f" = {_fmt(energy_au, 6)} a.u."
    )
    steps.append(f"换算：E = {_fmt(energy_ev, 5)} eV = {_fmt(C.ev_to_cm1(energy_ev), 6)} cm⁻¹"
                 f" = {_fmt(C.ev_to_kjmol(energy_ev), 5)} kJ/mol")
    steps.append(
        f"轨道角动量：|L| = √(l(l+1))ħ = √({l}×{l+1})ħ = √{l*(l+1)}ħ = {_fmt(abs_l, 5)}ħ"
    )
    steps.append(f"角动量 z 分量：Lz = mħ = {m}ħ")
    steps.append(
        f"简并度：同一 n 共 n² = {n}² = {n*n} 个轨道（能量相同、l 与 m 不同）；"
        f"计入自旋 2n² = {2*n*n} 个态"
    )
    if n == 1 and l == 0:
        steps.append(
            "最可几半径：1s 的径向分布函数 r²R²₁₀ = 4r²e^(-2r/a₀)/a₀³ 在 r = a₀ 取极大，"
            f"即 r_mp = a₀ = {_fmt(C.A0_PM, 6)} pm"
        )
    else:
        steps.append(
            f"最可几半径：对 r²R²{n}{_SUBLETTERS[l]} 数值求极大"
            f"（R 用广义拉盖尔多项式 L^(2l+1)_(n-l-1) 构造）→ r_mp = {_fmt(r_mp_bohr, 5)}a₀"
            f" = {_fmt(r_mp_pm, 6)} pm"
        )
    steps.append(
        f"平均半径：⟨r⟩ = n²a₀[1 + ½(1 - l(l+1)/n²)]"
        f" = {n*n}a₀[1 + ½(1 - {l*(l+1)}/{n*n})] = {_fmt(mean_r_bohr, 6)}a₀"
        f" = {_fmt(mean_r_pm, 6)} pm"
    )
    steps.append(
        "注意：最可几半径（径向分布极大）与平均半径 ⟨r⟩ 是两个不同的量，"
        f"本题 { _fmt(r_mp_pm, 6)} pm ≠ {_fmt(mean_r_pm, 6)} pm，答题时别混用"
    )

    return _ok(
        n=n,
        l=l,
        m=m,
        orbital=orbital,
        n_radial_nodes=n_radial_nodes,
        n_angular_nodes=n_angular_nodes,
        n_nodes_total=n_nodes_total,
        nodes_detail={
            "radial": n_radial_nodes,
            "angular": n_angular_nodes,
            "total": n_nodes_total,
        },
        energy_eV=energy_ev,
        energy_eV_str=f"{_fmt(energy_ev, 5)} eV",
        energy_eV_codata=energy_ev_codata,
        energy_au=energy_au,
        energy_au_str=f"{_sci(energy_au)} a.u.",
        energy_cm1=C.ev_to_cm1(energy_ev),
        energy_kJmol=C.ev_to_kjmol(energy_ev),
        energy_rydberg=-1.0 / (n * n),
        energy_convention=(
            "eV 主值走教材 13.6/n² 口径；a.u. 为精确的 -1/(2n²)。"
            "两套差 0.042%（13.6 vs R∞=13.6057 eV），答题按题目口径选一个，不要互相换算后混用"
        ),
        energy_eV_from_au=energy_au * C.HARTREE_EV,
        abs_L=abs_l,
        abs_L_unit="ħ",
        abs_L_str=f"√({l}×{l+1})ħ = {_fmt(abs_l, 5)}ħ",
        Lz=lz,
        Lz_unit="ħ",
        Lz_str=f"{m}ħ",
        L2=l * (l + 1),
        degeneracy=n * n,
        degeneracy_with_spin=2 * n * n,
        most_probable_r_pm=r_mp_pm,
        most_probable_r_A=r_mp_pm / C.PM_PER_A,
        most_probable_r_bohr=r_mp_bohr,
        most_probable_r_method=r_mp_method,
        mean_r_pm=mean_r_pm,
        mean_r_A=mean_r_pm / C.PM_PER_A,
        mean_r_bohr=mean_r_bohr,
        steps=steps,
    )


# ===========================================================================
# Q3  slater —— Slater 规则
# ===========================================================================


def _parse_config(config: Any) -> Dict[str, Any]:
    """解析组态字符串，返回元素符号、Z、各亚层占据。

    支持 ``"Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"``、``"1s2 2s2 2p6 3d6 4s2"``、
    ``"Fe3+:1s2 ..."``（离子，核电荷仍按元素取，正负号只用来核对电子数）、
    ``"Fe:1s² 2s² ..."``（上标数字）、``"Z=26: ..."``（显式核电荷）。
    """
    s = _norm_uni(str(config)).strip()
    s = s.replace("，", " ").replace(",", " ").replace("^", "")
    element: Optional[str] = None
    z_from_symbol: Optional[int] = None
    charge_hint: Optional[int] = None

    head = re.match(r"^\s*([A-Za-z]{1,2})\s*(\d*)\s*([+-])?\s*[:：]\s*(.+)$", s)
    if head and head.group(1).capitalize() in _ELEMENTS:
        element = head.group(1).capitalize()
        z_from_symbol = _ELEMENTS[element]
        if head.group(2) or head.group(3):
            mag = int(head.group(2)) if head.group(2) else 1
            sign = 1 if head.group(3) == "+" else -1
            charge_hint = sign * mag
        s = head.group(4)
    else:
        head2 = re.match(r"^\s*Z\s*=\s*(\d+)\s*[:：]\s*(.+)$", s, flags=re.IGNORECASE)
        if head2:
            z_from_symbol = int(head2.group(1))
            s = head2.group(2)

    shells: List[Tuple[int, int, int]] = []
    for mt in re.finditer(r"(\d+)\s*([spdfgh])\s*(\d+)", s):
        n_ = int(mt.group(1))
        l_ = _SUBL_INDEX[mt.group(2)]
        c_ = int(mt.group(3))
        if c_ <= 0:
            raise ValueError(f"亚层 {mt.group(1)}{mt.group(2)} 的电子数必须是正整数")
        if c_ > 2 * (2 * l_ + 1):
            raise ValueError(
                f"亚层 {mt.group(1)}{mt.group(2)} 最多容纳 {2*(2*l_+1)} 个电子，收到 {c_}"
            )
        shells.append((n_, l_, c_))

    if not shells:
        raise ValueError(
            "组态里没解析出任何亚层；正确写法如 \"Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2\""
        )

    electron_sum = sum(c for _, _, c in shells)
    z = z_from_symbol if z_from_symbol is not None else electron_sum
    charge_from_config = z - electron_sum

    return {
        "element": element,
        "Z": z,
        "Z_source": "元素符号" if z_from_symbol is not None and element else (
            "Z= 显式给出" if z_from_symbol is not None else "按中性原子 Z = Σ电子数 推定"
        ),
        "shells": shells,
        "electron_sum": electron_sum,
        "charge": charge_from_config,
        "charge_from_label": charge_hint,
        "charge_consistent": charge_hint is None or charge_hint == charge_from_config,
    }


def _slater_group_key(n: int, l: int) -> Tuple[int, int]:
    """Slater 分组键：ns 与 np 同组（子标记 0），nd/ nf/ ng 各自单独成组。"""
    return (n, 0 if l <= 1 else l)


def _slater_group_label(key: Tuple[int, int]) -> str:
    n, sub = key
    if sub == 0:
        return f"({n}s{n}p)" if n > 1 else f"({n}s)"
    return f"({n}{_SUBLETTERS[sub]})"


def slater(config: str, target: str) -> Dict[str, Any]:
    """Slater 规则：算目标电子的屏蔽常数 σ、有效核电荷 Z* 与轨道能量。

    参数
    ----
    config : str
        电子组态，如 ``"Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"``。不给元素符号时按
        中性原子 Z = Σ电子数 推定。
    target : str
        目标亚层，如 ``"3d"``、``"4s"``、``"2p"``。

    返回
    ----
    sigma / z_eff / groups（逐组贡献明细）/ E_eV / E_au / steps。

    规则（教材口径，严格按 Slater 分组）：
    1. 分组顺序 (1s)(2s2p)(3s3p)(3d)(4s4p)(4d)(4f)(5s5p)…；
    2. 目标电子右侧（更大 n，或同 n 但更高亚层）的电子屏蔽 0；
    3. 同组其它电子：1s 组每个 0.30，其余组每个 0.35；
    4. 目标为 s/p：n-1 层每个 0.85，n-2 及更内层每个 1.00；
    5. 目标为 d/f：左侧所有组每个 1.00；
    6. n* 查表；E = -13.6·Z*²/n*²。

    边界：Slater 规则是经验规则，给出的 Z* 与 Clementi–Raimondi 自洽场 Z_eff
    数值不同（后者对 3d 元素常明显更大）。本函数只按规则算，不混用两套口径。
    """
    if not isinstance(config, str) or not config.strip():
        return _fail('config 必须是非空字符串，如 "Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"')
    mt = re.fullmatch(r"\s*(\d+)\s*([spdfgh])\s*", _norm_uni(str(target)))
    if not mt:
        return _fail(f'target 必须写成 "3d" / "4s" / "2p" 这种「主量子数+亚层」形式，收到 {target!r}')
    t_n = int(mt.group(1))
    t_l = _SUBL_INDEX[mt.group(2)]
    if t_l > 3:
        return _fail(f"Slater 规则只覆盖 s/p/d/f 亚层，收到 {mt.group(2)}")

    try:
        parsed = _parse_config(config)
    except ValueError as exc:
        return _fail(str(exc))

    z = parsed["Z"]
    shells = parsed["shells"]

    grouped: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for (n_, l_, c_) in shells:
        key = _slater_group_key(n_, l_)
        slot = grouped.setdefault(key, {"electrons": 0, "members": []})
        slot["electrons"] += c_
        slot["members"].append((f"{n_}{_SUBLETTERS[l_]}", c_))

    target_key = _slater_group_key(t_n, t_l)
    if target_key not in grouped:
        available = "、".join(_slater_group_label(k) for k in sorted(grouped))
        return _fail(
            f"组态里没有 {t_n}{_SUBLETTERS[t_l]} 电子（现有分组：{available}）"
        )

    order = sorted(grouped)
    groups_out: List[Dict[str, Any]] = []
    sigma = 0.0
    detail_lines: List[str] = []
    same_group_rule = "1s 组同组电子每个 0.30" if (t_n == 1 and t_l == 0) else "同组电子每个 0.35"
    each_same = 0.30 if (t_n == 1 and t_l == 0) else 0.35

    for key in order:
        info = grouped[key]
        label = _slater_group_label(key)
        count = info["electrons"]
        member_str = " ".join(f"{name}^{c}" for name, c in info["members"])

        if key == target_key:
            counted = count - 1
            each = each_same
            role = "同组"
            rule = f"{same_group_rule}，目标电子自身不计"
        elif key > target_key:
            counted = count
            each = 0.0
            role = "外层（能量更高）"
            rule = "目标电子右侧/外层电子屏蔽为 0"
        else:  # 左侧
            counted = count
            if t_l <= 1:
                if key[0] == t_n - 1:
                    each = 0.85
                    role = "n-1 层"
                    rule = "s/p 目标：n-1 层每个电子 0.85"
                elif key[0] <= t_n - 2:
                    each = 1.00
                    role = "n-2 及更内层"
                    rule = "s/p 目标：n-2 及更内层每个电子 1.00"
                elif key[0] == t_n:
                    each = 1.00
                    role = "同 n 更低亚层"
                    rule = "同 n 但更低的亚层按内层处理，每个 1.00"
                else:  # pragma: no cover - 理论上不会到这里
                    each = 1.00
                    role = "内层"
                    rule = "内层每个 1.00"
            else:
                each = 1.00
                role = "内层（d/f 目标左侧全算）"
                rule = "d/f 目标：左侧所有组每个电子 1.00"

        contribution = counted * each
        sigma += contribution
        groups_out.append(
            {
                "label": label,
                "n": key[0],
                "subshells": info["members"],
                "subshells_str": member_str,
                "electrons": count,
                "counted_electrons": counted,
                "each": each,
                "contribution": contribution,
                "role": role,
                "rule": rule,
            }
        )
        detail_lines.append(
            f"{label}{member_str}：{counted} 个电子 × {each:.2f} = {contribution:.2f}"
            f"（{role}；{rule}）"
        )

    z_eff = z - sigma
    n_star = N_STAR_TABLE.get(t_n)
    if n_star is None:
        return _fail(f"n* 表只到 n=8，收到主量子数 n={t_n}")
    e_ev = -RY_EV_TEXTBOOK * z_eff * z_eff / (n_star * n_star)
    e_ev_codata = -C.RYDBERG_EV * z_eff * z_eff / (n_star * n_star)
    e_au = -z_eff * z_eff / (2.0 * n_star * n_star)

    steps: List[str] = []
    steps.append(
        f"组态解析：{parsed['element'] or '（未给元素）'}，Z = {z}"
        f"（{parsed['Z_source']}），Σ电子数 = {parsed['electron_sum']}"
        + (f"，电荷 {parsed['charge']:+d}" if parsed["charge"] else "（中性原子）")
    )
    if not parsed["charge_consistent"]:
        steps.append(
            f"⚠ 元素符号标注的电荷 {parsed['charge_from_label']:+d} 与「Z - Σ电子数」算出的 "
            f"{parsed['charge']:+d} 不一致，请核对组态；屏蔽常数的计算一律按核电荷 Z = {z}"
        )
    steps.append(
        "Slater 分组（按屏蔽顺序）："
        + "，".join(
            f"{_slater_group_label(k)}^{grouped[k]['electrons']}"
            for k in order
        )
    )
    steps.append(
        f"目标电子：{t_n}{_SUBLETTERS[t_l]}（n={t_n}，l={t_l}，"
        + ("s/p 电子，按「n-1 层 0.85、更内层 1.00」处理" if t_l <= 1
           else "d/f 电子，按「左侧各组每个 1.00」处理")
        + "）"
    )
    steps.append("逐组屏蔽贡献：")
    steps.extend(detail_lines)
    steps.append(
        f"σ = " + " + ".join(f"{g['contribution']:.2f}" for g in groups_out)
        + f" = {sigma:.2f}"
    )
    steps.append(f"有效核电荷 Z* = Z - σ = {z} - {sigma:.2f} = {z_eff:.2f}")
    steps.append(f"有效主量子数 n* = {n_star}（由 n={t_n} 查表）")
    steps.append(
        f"轨道能量 E = -13.6·Z*²/n*² = -13.6 × {z_eff:.2f}²/{n_star}² "
        f"= -13.6 × {z_eff * z_eff:.4f}/{n_star * n_star:.4f} = {_fmt(e_ev, 5)} eV"
    )
    steps.append(
        f"（原子单位形式 E = -Z*²/(2n*²) = {_fmt(e_au, 6)} a.u.；"
        f"R∞ 精确值口径给 {_fmt(e_ev_codata, 5)} eV）"
    )
    steps.append(
        "口径提醒：Slater 规则的 Z* 是经验值，与 Clementi–Raimondi 自洽场 Z_eff 数值不同，"
        "答题时按题目要求的规则口径，不要混用两套数据"
    )

    return _ok(
        configuration=str(config).strip(),
        element=parsed["element"],
        Z=z,
        Z_source=parsed["Z_source"],
        electron_sum=parsed["electron_sum"],
        charge=parsed["charge"],
        charge_from_label=parsed["charge_from_label"],
        charge_consistent=parsed["charge_consistent"],
        target=f"{t_n}{_SUBLETTERS[t_l]}",
        target_n=t_n,
        target_l=t_l,
        sigma=sigma,
        sigma_str=f"{sigma:.2f}",
        z_eff=z_eff,
        z_eff_str=f"{z_eff:.2f}",
        n_star=n_star,
        groups=groups_out,
        E_eV=e_ev,
        E_eV_str=f"{_fmt(e_ev, 5)} eV",
        E_eV_codata=e_ev_codata,
        E_au=e_au,
        E_cm1=C.ev_to_cm1(e_ev),
        E_kJmol=C.ev_to_kjmol(e_ev),
        steps=steps,
    )


# ===========================================================================
# Q4  spectroscopic_terms —— Russell-Saunders 谱项
# ===========================================================================


def _parse_orbital_spec(spec: Any) -> List[Tuple[Optional[int], int, int]]:
    """解析 ``"d2"`` / ``"3d6"`` / ``"p d"`` / ``"3d10 4s1"`` → [(n, l, k), ...]。"""
    s = _norm_uni(str(spec)).strip().lower()
    s = s.replace(",", " ").replace("，", " ").replace("、", " ").replace(";", " ")
    s = s.replace("^", "").replace("  ", " ")
    parts = [p for p in re.split(r"\s+", s) if p]
    if not parts:
        raise ValueError(
            '组态写法应为 "d2" / "p2" / "f3" / "3d6"，非同科写 "p d"（空格分隔）'
        )
    out: List[Tuple[Optional[int], int, int]] = []
    for part in parts:
        mt = re.fullmatch(r"(\d*)([spdfgh])(\d*)", part)
        if not mt:
            raise ValueError(
                f"认不出组态片段 {part!r}；应写成 <主量子数可选><亚层字母><电子数可选>，"
                '如 "d2"、"3d6"、"p"'
            )
        n_ = int(mt.group(1)) if mt.group(1) else None
        l_ = _SUBL_INDEX[mt.group(2)]
        k_ = int(mt.group(3)) if mt.group(3) else 1
        if k_ < 1:
            raise ValueError(f"片段 {part!r} 的电子数必须 ≥ 1")
        out.append((n_, l_, k_))
    return out


def _microstate_table(l: int, k: int) -> Dict[Tuple[int, int], int]:
    """枚举等价电子组态的微观状态，返回 {(M_L, 2M_S): 状态数}。

    每个电子占据一个自旋轨道 (m_l, m_s)，共 2(2l+1) 个；k 个电子取 k 个不同轨道
    （Pauli 原理）→ 共有 C(2(2l+1), k) 个微观状态。M_L = Σm_l，2M_S = Σ(2m_s)。
    """
    orbitals: List[Tuple[int, int]] = [
        (ml, ms2) for ml in range(-l, l + 1) for ms2 in (-1, 1)
    ]
    table: Dict[Tuple[int, int], int] = {}
    for combo in itertools.combinations(range(len(orbitals)), k):
        ml_sum = 0
        ms_sum = 0
        for idx in combo:
            ml_sum += orbitals[idx][0]
            ms_sum += orbitals[idx][1]
        key = (ml_sum, ms_sum)
        table[key] = table.get(key, 0) + 1
    return table


def _terms_equivalent(l: int, k: int) -> Dict[str, Any]:
    """等价电子谱项：最大 S 优先的逐项剥离法（Hund 剥离法）。

    做法：先真实枚举全部 C(2(2l+1), k) 个微观状态并按 (M_L, 2M_S) 计数；
    再反复取「M_S 最大 → 该行 M_L 最大」定位一个谱项 (L, S)，把它覆盖的
    (2S+1)(2L+1) 个格子各扣 1，直到表格耗尽。这是纯算法，不查表。
    """
    m_orbitals = 2 * (2 * l + 1)
    if k > m_orbitals:
        raise ValueError(f"{l} 亚层只有 {m_orbitals} 个自旋轨道，放不下 {k} 个电子")
    table = _microstate_table(l, k)
    total = sum(table.values())

    working = dict(table)
    found: List[Tuple[int, int]] = []
    strip_lines: List[str] = []
    mismatches: List[str] = []

    guard = 0
    while any(v > 0 for v in working.values()):
        guard += 1
        if guard > 500:  # pragma: no cover - 死循环保护
            raise RuntimeError("谱项剥离未收敛，请报告 bug")

        alive = [(key, v) for key, v in working.items() if v > 0]
        max_ms2 = max(key[1] for key, _ in alive)
        row = [(key, v) for key, v in alive if key[1] == max_ms2]
        max_ml = max(key[0] for key, _ in row)

        s2 = max_ms2           # 2S
        s_val = s2 / 2.0
        big_l = max_ml

        removed = 0
        for ml in range(-big_l, big_l + 1):
            for ms2 in range(-s2, s2 + 1, 2):
                key = (ml, ms2)
                if working.get(key, 0) > 0:
                    working[key] -= 1
                    removed += 1

        expected = (s2 + 1) * (2 * big_l + 1)
        if removed != expected:
            mismatches.append(
                f"({big_l},{s_val:g}) 剥离 {removed} 个状态，理论应为 {expected}"
            )
        found.append((big_l, s2))
        if len(strip_lines) < 40:
            strip_lines.append(
                f"最大 M_S = {s_val:g}（2M_S = {s2}），该行最大 M_L = {big_l} "
                f"→ 谱项 {s2+1}{_L_LETTER[big_l]}（L={big_l}, S={s_val:g}），"
                f"占 (2S+1)(2L+1) = {s2+1}×{2*big_l+1} = {expected} 个微观状态，从表中剥离"
            )

    return {
        "terms": found,
        "microstates_total": total,
        "strip_lines": strip_lines,
        "mismatches": mismatches,
        "m_orbitals": m_orbitals,
        "n_states_formula": f"C({m_orbitals},{k}) = {math.comb(m_orbitals, k)}",
    }


def _couple_terms(
    term_sets: Sequence[Sequence[Tuple[int, ...]]]
) -> List[Tuple[int, int, int]]:
    """多个子壳层的 L-S 耦合。输入每层的 [(L, 2S)]（或带重数的三段式），
    输出 [(L, 2S, 重数)]。"""
    acc: List[Tuple[int, int, int]] = [(0, 0, 1)]
    for tset in term_sets:
        merged: Dict[Tuple[int, int], int] = {}
        layer: List[Tuple[int, int, int]] = []
        for item in tset:
            if len(item) == 2:
                layer.append((int(item[0]), int(item[1]), 1))
            else:
                layer.append((int(item[0]), int(item[1]), int(item[2])))
        for (l1, s1, c1) in acc:
            for (l2, s2, c2) in layer:
                for big_l in range(abs(l1 - l2), l1 + l2 + 1):
                    for s in range(abs(s1 - s2), s1 + s2 + 1, 2):
                        merged[(big_l, s)] = merged.get((big_l, s), 0) + c1 * c2
        acc = [(big_l, s, c) for (big_l, s), c in merged.items()]
    return acc


def _j_string(two_j: int) -> str:
    """2J 整数 → "3/2" / "2" 这种读数写法。"""
    if two_j % 2 == 0:
        return str(two_j // 2)
    return f"{two_j}/2"


def _j_list(two_l: int, two_s: int) -> Tuple[List[float], List[int], List[str]]:
    """J 的取值：从 |L-S| 到 L+S，步长 1。

    参数是 **2L 与 2S**（整数，避免半整数浮点）。返回 (J 浮点, 2J 整数, J 字符串)，
    例如 2L=6, 2S=2 → J = [2.0, 3.0, 4.0]，2J = [4, 6, 8]，字符串 ["2","3","4"]。
    """
    lo = abs(two_l - two_s)
    hi = two_l + two_s
    halves = list(range(lo, hi + 1, 2))
    return [h / 2.0 for h in halves], halves, [_j_string(h) for h in halves]


def spectroscopic_terms(config: str) -> Dict[str, Any]:
    """Russell-Saunders 谱项：等价电子剥离 / 非同科电子 L-S 耦合 + Hund 定基项。

    参数
    ----
    config : str
        等价电子写 ``"d2"`` / ``"p2"`` / ``"f3"`` / ``"3d6"``；
        非同科电子用空格分隔写 ``"p d"`` / ``"3d10 4s1"``（每段内部等价）。

    返回
    ----
    terms（含 term / L / S / multiplicity / J / microstates）、
    microstates_total、check（自校验）、ground_term（Hund）、steps。

    做法（真算法，不查答案表）：
    1. 等价 k 个电子填 2(2l+1) 个自旋轨道 → 枚举 C(2(2l+1), k) 个微观状态，
       按 (M_L, 2M_S) 计数；
    2. 「最大 S 优先的逐项剥离」：反复取 M_S 最大 → 该行 M_L 最大 定出 (L,S)，
       扣掉 (2S+1)(2L+1) 个格子，直到表格耗尽；
    3. 自校验 Σ(2S+1)(2L+1) = ∏C(2(2l+1),k)，不符即报错而不是硬给结果；
    4. Hund 三规则：S 最大 → L 最大 → J（少于半满 |L-S|，超过半满 L+S）。

    边界：只做 LS 耦合（轻原子），不做 j-j 耦合；非同科多壳层的 J 只按 Hund
    第三规则给形式值，精细结构能级次序不在本函数职责内。
    """
    if not isinstance(config, str) or not config.strip():
        return _fail('config 必须是非空字符串，如 "d2"、"p2"、"f3"、"3d6"，非同科写 "p d"')
    try:
        pieces = _parse_orbital_spec(config)
    except ValueError as exc:
        return _fail(str(exc))

    # 若各片 (n, l) 完全相同，视为同一壳层，合并电子数
    keys = {(p[0], p[1]) for p in pieces}
    merged_note = ""
    if len(pieces) > 1 and len(keys) == 1:
        n0, l0 = pieces[0][0], pieces[0][1]
        total_k = sum(p[2] for p in pieces)
        pieces = [(n0, l0, total_k)]
        merged_note = f"各片段同一亚层，合并为 {total_k} 个等价电子"

    is_equivalent = len(pieces) == 1
    steps: List[str] = []
    if merged_note:
        steps.append(merged_note)

    microstates_total = 1
    term_sets: List[List[Tuple[int, int]]] = []
    subshell_desc: List[str] = []
    strip_all: List[str] = []
    mismatches: List[str] = []

    for (n_, l_, k_) in pieces:
        m_orb = 2 * (2 * l_ + 1)
        if k_ > m_orb:
            return _fail(
                f"{_SUBLETTERS[l_]} 亚层只有 {m_orb} 个自旋轨道，放不下 {k_} 个电子"
            )
        try:
            res = _terms_equivalent(l_, k_)
        except (ValueError, RuntimeError) as exc:
            return _fail(str(exc))
        microstates_total *= math.comb(m_orb, k_)
        term_sets.append(res["terms"])
        mismatches.extend(res["mismatches"])
        label = f"{n_}{_SUBLETTERS[l_]}" if n_ is not None else f"{_SUBLETTERS[l_]}"
        subshell_desc.append(f"{label}^{k_}")
        if is_equivalent:
            steps.append(
                f"组态 {config.strip()}：{k_} 个等价 {_SUBLETTERS[l_]} 电子，"
                f"自旋轨道数 2(2l+1) = {m_orb}，微观状态总数 = C({m_orb},{k_}) "
                f"= {math.comb(m_orb, k_)}"
            )
            steps.append("按 (M_L, 2M_S) 建表后，用「最大 S 优先」逐项剥离：")
            strip_all.extend(res["strip_lines"])
        else:
            steps.append(
                f"子壳层 {label}^{k_}（内部等价）：微观状态数 C({m_orb},{k_}) "
                f"= {math.comb(m_orb, k_)}，谱项 "
                + "、".join(f"{s+1}{_L_LETTER[l]}" for (l, s) in sorted(
                    res["terms"], key=lambda t: (-t[1], -t[0])))
            )

    # ---- 耦合 ----
    if is_equivalent:
        coupled = [(big_l, s2, 1) for (big_l, s2) in term_sets[0]]
        method = "等价电子逐项剥离（Hund 剥离法）"
    else:
        coupled = _couple_terms(term_sets)
        method = "非同科电子 L-S 矢量耦合"
        steps.append(
            "L 耦合：L = |L₁-L₂| … L₁+L₂（步长 1）；"
            "S 耦合：S = |S₁-S₂| … S₁+S₂（步长 1，即 2S 步长 2），逐层合并"
        )

    # 汇总 + 去重 + 排序（S 降序、L 降序）
    agg: Dict[Tuple[int, int], int] = {}
    for (big_l, s2, cnt) in coupled:
        agg[(big_l, s2)] = agg.get((big_l, s2), 0) + cnt
    ordered = sorted(agg.items(), key=lambda kv: (-kv[0][1], -kv[0][0]))

    terms_out: List[Dict[str, Any]] = []
    sum_micro = 0
    for ((big_l, s2), cnt) in ordered:
        mult = s2 + 1
        single = mult * (2 * big_l + 1)
        contrib = single * cnt
        sum_micro += contrib
        j_vals, j_halves, j_strs = _j_list(2 * big_l, s2)
        term_name = f"{mult}{_L_LETTER[big_l]}"
        terms_out.append(
            {
                "term": term_name,
                "term_pretty": f"{str(mult).translate(_SUPERSCRIPT_MAP)}{_L_LETTER[big_l]}",
                "L": big_l,
                "L_label": _L_LETTER[big_l],
                "S": s2 / 2.0,
                "two_S": s2,
                "multiplicity": mult,
                "J": j_vals,
                "J_half": j_halves,
                "J_str": j_strs,
                "microstates": single,
                "count": cnt,
                "microstates_total": contrib,
            }
        )

    check_passed = sum_micro == microstates_total and not mismatches
    check = {
        "passed": check_passed,
        "sum_term_microstates": sum_micro,
        "microstates_total": microstates_total,
        "formula": (
            "Σ(2S+1)(2L+1) = ∏C(2(2l+1), k) → "
            f"{sum_micro} = {microstates_total}"
        ),
        "mismatches": mismatches,
    }
    if not check_passed:
        return _fail(
            "谱项剥离自校验失败（Σ(2S+1)(2L+1) = "
            f"{sum_micro} ≠ 微观状态总数 {microstates_total}）；"
            "这是算法层面的不一致，不作为答案给出，请报告 bug。"
            + ("；" + "；".join(mismatches) if mismatches else "")
        )

    # ---- Hund 三规则定基项 ----
    s_max = max(t["two_S"] for t in terms_out)
    cand = [t for t in terms_out if t["two_S"] == s_max]
    l_max = max(t["L"] for t in cand)
    ground = [t for t in cand if t["L"] == l_max][0]

    # 半满判据用「等价组态」的总电子数；非同科则按最高 S 的壳层定性并说明
    if is_equivalent:
        k_total = pieces[0][2]
        half = 2 * pieces[0][1] + 1
        if k_total < half:
            j_ground_half = abs(ground["L"] * 2 - ground["two_S"])
            j_rule = f"少于半满（{k_total} < {half}）：J = |L-S|"
        elif k_total == half:
            j_ground_half = ground["two_S"]
            j_rule = f"恰好半满（{k_total} = {half}）：L=0，J = S"
        else:
            j_ground_half = ground["L"] * 2 + ground["two_S"]
            j_rule = f"超过半满（{k_total} > {half}）：J = L+S"
    else:
        total_k = sum(p[2] for p in pieces)
        j_ground_half = abs(ground["L"] * 2 - ground["two_S"])
        j_rule = (
            f"非同科组态（共 {total_k} 个电子）：Hund 第三规则只严格适用于等价电子，"
            "这里按轻原子口径给 J = |L-S|；精细结构次序需另算"
        )
    if j_ground_half not in ground["J_half"]:
        j_ground_half = ground["J_half"][0]
        j_rule += "（该 J 不在允许范围内，取最近的合法值）"

    j_ground = j_ground_half / 2.0
    ground_term = ground["term"]
    ground_state = f"{ground_term}{_j_string(j_ground_half)}"
    ground_pretty = ground["term_pretty"]

    steps.append(
        "耦合结果（按 S 降序、L 降序；重复出现的项按次数写出）："
        + "、".join(t["term"] for t in terms_out for _ in range(t["count"]))
    )
    steps.append(
        "自校验：Σ(2S+1)(2L+1) = "
        + " + ".join(f"{t['microstates_total']}" for t in terms_out)
        + f" = {sum_micro}；微观状态总数 = {microstates_total} → "
        + ("通过 ✓" if check_passed else "不通过 ✗")
    )
    steps.append(
        f"Hund 第一规则：S 最大 → S = {s_max/2:g}，候选 "
        + "、".join(t["term"] for t in cand)
    )
    steps.append(f"Hund 第二规则：同 S 中 L 最大 → L = {l_max}，得基项 {ground_term}")
    steps.append(f"Hund 第三规则：{j_rule} → J = {_j_string(j_ground_half)}，基态 {ground_pretty}{_j_string(j_ground_half)}")

    return _ok(
        config=str(config).strip(),
        subshells=subshell_desc,
        is_equivalent=is_equivalent,
        method=method,
        terms=terms_out,
        # terms_str 按教材写法展开重复项（d6 里两个 3P 就写两次），便于直接抄
        terms_str=[t["term"] for t in terms_out for _ in range(t["count"])],
        terms_str_unique=[t["term"] for t in terms_out],
        microstates_total=microstates_total,
        check=check,
        check_passed=check_passed,
        ground_term=ground_term,
        ground_term_pretty=ground_pretty,
        ground_term_J=j_ground,
        ground_term_J_half=j_ground_half,
        ground_state=ground_state,
        ground_state_pretty=f"{ground_pretty}{_j_string(j_ground_half)}",
        ground_rule=j_rule,
        ground_detail={
            "S": ground["S"],
            "L": ground["L"],
            "multiplicity": ground["multiplicity"],
            "J": j_ground,
        },
        half_filled_reference=(
            2 * pieces[0][1] + 1 if is_equivalent else None
        ),
        strip_process=strip_all,
        steps=steps,
    )


# ===========================================================================
# Q5  superposition —— 态叠加
# ===========================================================================


def _coef_float(value: Any, idx: int) -> Tuple[Optional[float], str]:
    """解析叠加系数，支持 0.8 / "1/√2" / "sqrt(2)/2"。"""
    if isinstance(value, bool):
        return None, f"第 {idx} 项的 coef 不能是布尔值"
    if isinstance(value, (int, float)):
        return float(value), ""
    if isinstance(value, str):
        try:
            return _safe_numeric(value), ""
        except (ValueError, ZeroDivisionError) as exc:
            return None, f"第 {idx} 项的 coef 解析失败：{exc}"
    return None, f"第 {idx} 项的 coef 必须是数字或可解析的字符串"


def superposition(terms: Any) -> Dict[str, Any]:
    """波函数叠加：能量平均值、各本征能量出现几率、L² 与 Lz 平均值。

    参数
    ----
    terms : list[dict]
        ``[{"coef": 0.8, "n": 2, "l": 1, "m": 0}, {"coef": 0.6, "n": 3, "l": 1, "m": 1}]``
        coef 可为数字或 "1/√2"、"sqrt(2)/2" 这类字符串。

    返回
    ----
    norm_check / energy_mean_eV / energy_mean_au / eigen_probabilities /
    L2_mean / Lz_mean / Lz_values_prob / steps。

    口径：ψ = Σ c_i ψ_i，⟨E⟩ = Σ|c_i|²E_i / Σ|c_i|²。
    Σ|c_i|² ≠ 1 时视为「波函数未归一化」，按归一化（系数统一除以 √Σ|c_i|²）
    处理后给主值，同时在 steps 里明确提示并保留原始系数的结果作对照。

    边界：纯态叠加的期望值，不处理含时演化与测量后的塌缩动力学。
    """
    if not isinstance(terms, (list, tuple)) or len(terms) == 0:
        return _fail(
            'terms 必须是非空列表，形如 [{"coef":0.8,"n":2,"l":1,"m":0}, ...]'
        )

    parsed: List[Dict[str, Any]] = []
    for idx, item in enumerate(terms, 1):
        if not isinstance(item, dict):
            return _fail(f"第 {idx} 项必须是字典，形如 {{\"coef\":0.8,\"n\":2,\"l\":1,\"m\":0}}")
        if "coef" not in item:
            return _fail(f"第 {idx} 项缺 coef（叠加系数）")
        coef, err = _coef_float(item["coef"], idx)
        if coef is None:
            return _fail(err)
        missing = [key for key in ("n", "l", "m") if key not in item]
        if missing:
            return _fail(f"第 {idx} 项缺 {'、'.join(missing)}（量子数 n/l/m）")
        try:
            n_ = int(item["n"])
            l_ = int(item["l"])
            m_ = int(item["m"])
        except (TypeError, ValueError):
            return _fail(f"第 {idx} 项的 n/l/m 必须都是整数")
        if n_ < 1:
            return _fail(f"第 {idx} 项的主量子数 n 必须 ≥ 1，收到 {n_}")
        if l_ < 0 or l_ > n_ - 1:
            return _fail(f"第 {idx} 项必须满足 0 ≤ l ≤ n-1（n={n_}），收到 l={l_}")
        if abs(m_) > l_:
            return _fail(f"第 {idx} 项必须满足 |m| ≤ l（l={l_}），收到 m={m_}")
        parsed.append({"coef": coef, "n": n_, "l": l_, "m": m_})

    sum_c2 = sum(p["coef"] ** 2 for p in parsed)
    if sum_c2 <= 0:
        return _fail("所有叠加系数都是 0，波函数恒为零，没有物理意义")
    normalized = abs(sum_c2 - 1.0) <= 1e-9
    scale = 1.0 / math.sqrt(sum_c2)

    steps: List[str] = []
    steps.append("叠加波函数 ψ = Σ c_i·ψ_i，归一化条件 Σ|c_i|² = 1")
    detail = " + ".join(f"({_fmt(p['coef'], 6)})²" for p in parsed)
    steps.append(f"归一化检查：Σ|c_i|² = {detail} = {_fmt(sum_c2, 8)}")

    for p in parsed:
        p["norm_coef"] = p["coef"] * scale
        p["prob"] = (p["coef"] ** 2) / sum_c2
        p["energy_au"] = -1.0 / (2.0 * p["n"] ** 2)
        p["energy_ev"] = -RY_EV_TEXTBOOK / (p["n"] ** 2)

    if normalized:
        steps.append("Σ|c_i|² = 1，波函数已归一化 ✓")
        norm_message = "归一化检查通过（Σ|c_i|² = 1）"
        norm_severity = "ok"
    else:
        deviation = abs(sum_c2 - 1.0)
        if deviation < 1e-3:
            steps.append(
                f"Σ|c_i|² = {_fmt(sum_c2, 8)}，与 1 的偏差 {_sci(deviation)} 属于系数舍入"
                "（典型情况：把 1/√2 写成 0.7071），不是真错误；已按归一化处理，"
                "卷面上直接写 1/√2 最干净"
            )
            norm_message = (
                f"Σ|c_i|² = {_fmt(sum_c2, 8)}（偏差 {_sci(deviation)}，属系数舍入），"
                "已按归一化处理"
            )
            norm_severity = "rounding"
        else:
            steps.append(
                f"⚠ 波函数未归一化（Σ|c_i|² = {_fmt(sum_c2, 6)} ≠ 1），"
                f"已按归一化处理：每个系数除以 √Σ|c_i|² = {_fmt(1/scale, 6)}；"
                "请检查原题系数（常见误区：把 1/√2 写成 1/2）"
            )
            norm_message = (
                f"波函数未归一化（Σ|c_i|² = {_fmt(sum_c2, 8)}），"
                f"已按归一化处理（系数 ÷ {_fmt(1/scale, 8)}）"
            )
            norm_severity = "unnormalized"

    # ---- 能量平均 ----
    e_au = sum(p["prob"] * p["energy_au"] for p in parsed)
    e_ev = sum(p["prob"] * p["energy_ev"] for p in parsed)
    e_au_raw = sum(p["coef"] ** 2 * p["energy_au"] for p in parsed)
    e_ev_raw = sum(p["coef"] ** 2 * p["energy_ev"] for p in parsed)

    terms_str = " + ".join(
        f"{_fmt(p['prob'], 6)}×({_fmt(p['energy_au'], 6)})" for p in parsed
    )
    steps.append("能量平均值（归一化后）：⟨E⟩ = Σ|c_i|²E_i / Σ|c_i|²")
    steps.append("各项 E_i = -1/(2n_i²) a.u.（= -13.6/n_i² eV）：")
    for p in parsed:
        steps.append(
            f"  ψ({p['n']},{p['l']},{p['m']})：E = -1/(2×{p['n']}²) = "
            f"{_fmt(p['energy_au'], 6)} a.u. = {_fmt(p['energy_ev'], 5)} eV，"
            f"权重 |c|²/Σ|c|² = {_fmt(p['prob'], 6)}"
        )
    steps.append(f"⟨E⟩ = {terms_str} = {_fmt(e_au, 8)} a.u.")
    steps.append(
        f"⟨E⟩ = {_fmt(e_au, 6)} a.u. = "
        f"{_fmt(e_au * C.HARTREE_EV, 6)} eV = {_fmt(C.ev_to_kjmol(e_ev), 6)} kJ/mol"
    )
    if not normalized:
        steps.append(
            f"对照（不归一化直接算 Σ|c_i|²E_i）：{_fmt(e_au_raw, 8)} a.u.；"
            "两者只有在 Σ|c_i|²=1 时才相同"
        )

    # ---- 各本征能量的出现几率 ----
    eigen_bucket: Dict[int, Dict[str, Any]] = {}
    for p in parsed:
        slot = eigen_bucket.setdefault(
            p["n"],
            {
                "eigen": f"E(n={p['n']})",
                "value_eV": p["energy_ev"],
                "value_au": p["energy_au"],
                "value_cm1": C.ev_to_cm1(p["energy_ev"]),
                "prob": 0.0,
                "components": [],
            },
        )
        slot["prob"] += p["prob"]
        slot["components"].append(
            f"|{p['n']},{p['l']},{p['m']}⟩"
        )
    eigen_probs = sorted(eigen_bucket.values(), key=lambda d: d["value_au"])
    steps.append("测量能量 E_i 出现的几率 P(E_i) = Σ_{E_i 对应的项}|c_i|²：")
    for slot in eigen_probs:
        steps.append(
            f"  {slot['eigen']}（{_fmt(slot['value_eV'], 5)} eV，"
            f"{'、'.join(slot['components'])}）：P = {_fmt(slot['prob'], 6)}"
        )
    steps.append(
        "几率归一：Σ P(E_i) = "
        + " + ".join(_fmt(s["prob"], 6) for s in eigen_probs)
        + f" = {_fmt(sum(s['prob'] for s in eigen_probs), 8)}"
    )

    # ---- L² 与 Lz ----
    l2_mean = sum(p["prob"] * p["l"] * (p["l"] + 1) for p in parsed)
    lz_mean = sum(p["prob"] * p["m"] for p in parsed)
    steps.append("⟨L²⟩ = Σ|c_i|²l_i(l_i+1)ħ² = "
                 + " + ".join(
                     f"{_fmt(p['prob'], 6)}×{p['l']*(p['l']+1)}" for p in parsed
                 )
                 + f" = {_fmt(l2_mean, 6)} ħ²")
    steps.append("⟨Lz⟩ = Σ|c_i|²m_iħ = "
                 + " + ".join(
                     f"{_fmt(p['prob'], 6)}×({p['m']})" for p in parsed
                 )
                 + f" = {_fmt(lz_mean, 6)} ħ")

    lz_bucket: Dict[int, float] = {}
    for p in parsed:
        lz_bucket[p["m"]] = lz_bucket.get(p["m"], 0.0) + p["prob"]
    lz_values_prob = [
        {"m": mv, "Lz_hbar": float(mv), "prob": prb}
        for mv, prb in sorted(lz_bucket.items())
    ]

    return _ok(
        terms=[
            {
                "coef": p["coef"],
                "coef_normalized": p["norm_coef"],
                "prob": p["prob"],
                "n": p["n"],
                "l": p["l"],
                "m": p["m"],
                "energy_eV": p["energy_ev"],
                "energy_au": p["energy_au"],
                "ket": f"|{p['n']},{p['l']},{p['m']}⟩",
            }
            for p in parsed
        ],
        norm_check={
            "sum_abs_c2": sum_c2,
            "sum_abs_c2_str": _fmt(sum_c2, 8),
            "normalized": normalized,
            "severity": norm_severity,
            "normalization_factor": 1.0 / scale,
            "message": norm_message,
        },
        energy_mean_eV=e_ev,
        energy_mean_eV_str=f"{_fmt(e_ev, 6)} eV",
        energy_mean_au=e_au,
        energy_mean_au_str=f"{_fmt(e_au, 8)} a.u.",
        energy_mean_eV_from_au=e_au * C.HARTREE_EV,
        energy_convention=(
            "energy_mean_eV 走教材 13.6/n² 口径，energy_mean_au 走精确 -1/(2n²)，"
            "两者差 0.042% 属常数口径差异；卷面只写一种，别把 eV 与 a.u. 互换后混用"
        ),
        energy_mean_cm1=C.ev_to_cm1(e_ev),
        energy_mean_kJmol=C.ev_to_kjmol(e_ev),
        energy_mean_raw_eV=e_ev_raw,
        energy_mean_raw_au=e_au_raw,
        eigen_probabilities=eigen_probs,
        L2_mean=l2_mean,
        L2_mean_unit="ħ²",
        L2_mean_str=f"{_fmt(l2_mean, 6)} ħ²",
        Lz_mean=lz_mean,
        Lz_mean_unit="ħ",
        Lz_mean_str=f"{_fmt(lz_mean, 6)} ħ",
        Lz_values_prob=lz_values_prob,
        steps=steps,
    )


# ===========================================================================
# 自测：契约断言 + 附加硬门槛
# ===========================================================================

if __name__ == "__main__":  # pragma: no cover
    import sys

    # Windows 控制台默认 GBK，steps 里用到的 ⁻¹²³ 等字符会炸；统一转 UTF-8。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    FAILS: List[str] = []

    def chk(cond: bool, label: str, got: str = "") -> None:
        if cond:
            print(f"  [OK]   {label}" + (f"   {got}" if got else ""))
        else:
            print(f"  [FAIL] {label}" + (f"   {got}" if got else ""))
            FAILS.append(label)

    def close(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    print("=" * 78)
    print("Q1  particle_box")
    print("=" * 78)
    r = particle_box(3, 100.0)
    chk(r["ok"], "particle_box(3,100) 返回 ok")
    e1 = r["levels"][0]["E_eV"]
    e3 = r["levels"][2]["E_eV"]
    chk(close(e3 / e1, 9.0, 1e-12), "契约断言 E3/E1 = 9", f"E3/E1 = {e3/e1:.10f}")
    chk(1.0 < e1 < 200.0, "E1 量级合理（eV）", f"E1 = {e1:.4f} eV")
    chk(close(r["zero_point_eV"], e1), "零点能 = 最低能级", f"{r['zero_point_eV']:.4f} eV")
    chk(r["nodes"] == 2, "nodes = n-1 = 2", f"{r['nodes']}")
    tr = r["transition"]
    chk(tr is not None and close(tr["dE_eV"], r["levels"][1]["E_eV"] - e1),
        "默认跃迁 1→2 给出 ΔE", tr["dE_str"] if tr else "None")
    print(f"  E1 = {e1:.5f} eV，E2 = {r['levels'][1]['E_eV']:.5f} eV，E3 = {e3:.5f} eV")
    print(f"  跃迁 λ = {tr['wavelength_str']}")

    r = particle_box(3, 100.0, dim=3)
    labels = [(lv["n_squared"], lv["degeneracy"]) for lv in r["levels"][:4]]
    chk(labels[0] == (3, 1), "三维最低能级 n²=3 简并 1", f"{labels[0]}")
    chk(labels[1] == (6, 3), "n²=6 简并 3", f"{labels[1]}")
    chk(labels[2] == (9, 3), "n²=9 简并 3", f"{labels[2]}")
    chk(labels[3] == (11, 3), "n²=11 简并 3", f"{labels[3]}")

    r = particle_box(5, 200.0, electrons=3)
    chk(r["transition"]["mode"] == "homo_lumo", "给 electrons 时走 HOMO→LUMO",
        r["transition"]["from_level"] + "→" + r["transition"]["to_level"])

    rbox = particle_box(3, [100.0, 200.0, 300.0], dim=3)
    chk(rbox["ok"] and rbox["levels"][0]["label"] == "(1,1,1)",
        "长方体势箱（边长为序列）正常出能级",
        f"E1 = {rbox['levels'][0]['E_str']}，简并 {rbox['levels'][0]['degeneracy']}")
    chk(rbox["levels"][0]["degeneracy"] == 1 and rbox["levels"][1]["degeneracy"] == 1,
        "长方体各边不等 → 简并解除")

    chk(particle_box(0, 100.0)["ok"] is False, "非法 n=0 返回 ok=False")
    chk(particle_box(3, -1.0)["ok"] is False, "非法 L<0 返回 ok=False")
    chk(particle_box(3, 100.0, dim=4)["ok"] is False, "非法 dim=4 返回 ok=False")
    chk(particle_box(3, [100.0, 200.0], dim=3)["ok"] is False, "边长数量与 dim 不符时报错")

    print()
    print("=" * 78)
    print("Q2  hydrogen_orbital")
    print("=" * 78)
    r = hydrogen_orbital(3, 1, 0)
    chk(r["n_radial_nodes"] == 1, "契约断言 3p 径向节点 = 1", f"{r['n_radial_nodes']}")
    chk(r["n_angular_nodes"] == 1, "契约断言 3p 角节面 = 1", f"{r['n_angular_nodes']}")
    chk(r["n_nodes_total"] == 2, "契约断言 3p 总节面 = 2", f"{r['n_nodes_total']}")
    chk(close(r["energy_eV"], -1.511, 1e-3), "契约断言 E(3) ≈ -1.511 eV", f"{r['energy_eV']:.6f} eV")
    chk(close(r["energy_au"], -1.0 / 18.0, 1e-12), "E = -1/(2n²) a.u. = -1/18",
        f"{r['energy_au']:.8f} a.u.")
    chk(close(r["abs_L"], math.sqrt(2), 1e-12), "|L| = √2 ħ", r["abs_L_str"])
    chk(r["Lz"] == 0.0, "Lz = 0ħ", r["Lz_str"])
    chk(r["degeneracy"] == 9 and r["degeneracy_with_spin"] == 18, "简并度 n²=9 / 2n²=18")
    print(f"  3p 最可几半径 = {r['most_probable_r_pm']:.3f} pm"
          f"（{r['most_probable_r_bohr']:.4f} a0），⟨r⟩ = {r['mean_r_pm']:.3f} pm")

    r = hydrogen_orbital(1, 0, 0)
    chk(close(r["most_probable_r_pm"], 52.9177, 1e-4), "1s 最可几半径 = a0 = 52.92 pm",
        f"{r['most_probable_r_pm']:.4f} pm")
    chk(close(r["mean_r_pm"], 1.5 * C.A0_PM, 1e-12), "1s ⟨r⟩ = 1.5 a0",
        f"{r['mean_r_pm']:.4f} pm")
    chk(close(r["energy_eV"], -13.6, 1e-12), "1s E = -13.6 eV", f"{r['energy_eV']:.4f} eV")

    r = hydrogen_orbital(2, 1, 1)
    chk(close(r["most_probable_r_pm"], 4 * C.A0_PM, 2e-3), "2p 最可几半径 = 4a0",
        f"{r['most_probable_r_pm']:.4f} pm")
    chk(close(r["mean_r_pm"], 5 * C.A0_PM, 1e-12), "2p ⟨r⟩ = 5a0",
        f"{r['mean_r_pm']:.4f} pm")
    chk(close(r["Lz"], 1.0), "2p m=1 → Lz = 1ħ", r["Lz_str"])

    r = hydrogen_orbital(3, 2, 0)
    chk(r["n_radial_nodes"] == 0 and r["n_nodes_total"] == 2, "3d 径向节点 0、总节面 2")
    print(f"  3d 最可几半径 = {r['most_probable_r_pm']:.3f} pm"
          f"（{r['most_probable_r_bohr']:.4f} a0），⟨r⟩ = {r['mean_r_pm']:.3f} pm")

    chk(hydrogen_orbital(0, 0)["ok"] is False, "非法 n=0 返回 ok=False")
    chk(hydrogen_orbital(2, 2)["ok"] is False, "非法 l=n 返回 ok=False")
    chk(hydrogen_orbital(2, 1, 2)["ok"] is False, "非法 |m|>l 返回 ok=False")

    print()
    print("=" * 78)
    print("Q3  slater")
    print("=" * 78)
    FE = "Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"
    r = slater(FE, "3d")
    chk(r["ok"], "slater(Fe,3d) 返回 ok")
    chk(close(r["sigma"], 19.75, 1e-9), "Fe 3d σ = 19.75", f"{r['sigma']:.4f}")
    chk(close(r["z_eff"], 6.25, 1e-9), "Fe 3d Z* = 6.25（标准答案）", f"{r['z_eff']:.4f}")
    chk(close(r["E_eV"], -13.6 * 6.25 ** 2 / 9, 1e-9), "Fe 3d E = -13.6·Z*²/n*²",
        f"{r['E_eV']:.4f} eV")
    r4 = slater(FE, "4s")
    chk(close(r4["z_eff"], 3.75, 1e-9), "Fe 4s Z* = 3.75（标准答案）", f"{r4['z_eff']:.4f}")
    chk(close(r4["sigma"], 22.25, 1e-9), "Fe 4s σ = 22.25", f"{r4['sigma']:.4f}")

    rcu = slater("Cu:1s2 2s2 2p6 3s2 3p6 3d10 4s1", "3d")
    chk(close(rcu["z_eff"], 7.85, 1e-9), "Cu 3d Z*(Slater 规则) = 7.85", f"{rcu['z_eff']:.4f}")
    rcu4 = slater("Cu:1s2 2s2 2p6 3s2 3p6 3d10 4s1", "4s")
    chk(close(rcu4["z_eff"], 3.70, 1e-9), "Cu 4s Z*(Slater 规则) = 3.70", f"{rcu4['z_eff']:.4f}")
    print("  ⚠ Clémenti–Raimondi 自洽场值 Cu 3d Z_eff ≈ 13.2、Cu 4s ≈ 6.1，")
    print("    与本规则算出的 7.85 / 3.70 不是同一套口径 —— 本函数只按 Slater 规则出数。")

    rn = slater("1s2 2s2 2p6 3s2 3p6 3d6 4s2", "3d")
    chk(rn["ok"] and close(rn["z_eff"], 6.25, 1e-9), "不给元素符号时按 Z=Σ电子数 推定",
        f"Z={rn['Z']}，Z*={rn['z_eff']:.2f}")
    rp = slater("C:1s2 2s2 2p2", "2p")
    chk(close(rp["sigma"], 2 * 0.85 + 3 * 0.35, 1e-9), "1s 同组 0.30 / 其它同组 0.35 生效",
        f"C 2p σ = {rp['sigma']:.3f}（2×0.85 + 3×0.35）")
    rh = slater("He:1s2", "1s")
    chk(close(rh["sigma"], 0.30, 1e-12), "1s 组同组电子每个 0.30 生效", f"σ = {rh['sigma']:.2f}")

    rion = slater("Fe3+:1s2 2s2 2p6 3s2 3p6 3d5", "3d")
    chk(rion["ok"] and rion["Z"] == 26 and rion["charge"] == 3,
        "离子组态 Fe3+ 的核电荷仍取 26", f"Z={rion['Z']}，电荷 {rion['charge']:+d}")
    chk(close(rion["z_eff"], 6.60, 1e-9), "Fe3+ 3d5 的 Z* = 6.60", f"{rion['z_eff']:.2f}")
    chk(rion["charge_consistent"] is True, "电荷与组态电子数自洽", "✓")
    chk(slater(FE, "5d")["ok"] is False, "组态里没有目标亚层时报错")
    chk(slater("乱写", "3d")["ok"] is False, "非法组态返回 ok=False")
    chk(slater(FE, "abc")["ok"] is False, "非法 target 返回 ok=False")

    print()
    print("=" * 78)
    print("Q4  spectroscopic_terms")
    print("=" * 78)
    r = spectroscopic_terms("d2")
    chk(r["ok"], "spectroscopic_terms(d2) 返回 ok")
    chk(len(r["terms_str"]) == 5, "d2 共 5 个谱项", "、".join(r["terms_str"]))
    chk(set(r["terms_str"]) == {"1G", "3F", "1D", "3P", "1S"}, "契约断言 d2 = 1G 3F 1D 3P 1S",
        "、".join(r["terms_str"]))
    chk(r["microstates_total"] == 45, "契约断言 d2 微观状态总数 = 45", f"{r['microstates_total']}")
    chk(r["check_passed"], "d2 自校验 Σ(2S+1)(2L+1) = 45", r["check"]["formula"])
    chk(r["ground_term"] == "3F", "契约断言 d2 基项 = 3F", r["ground_state"])
    chk(close(r["ground_term_J"], 2.0), "d2 基态 3F2（J=|L-S|=2）", r["ground_state_pretty"])
    chk("3D" not in r["terms_str"], "d2 不含 3D 项（3D 属 d3/d7 体系，任务描述该条为笔误）")
    print(f"  d2 剥离过程共 {len(r['strip_process'])} 步，前 5 步：")
    for line in r["strip_process"][:5]:
        print(f"    {line}")

    r6 = spectroscopic_terms("d6")
    chk(r6["microstates_total"] == 210, "d6 微观状态总数 = C(10,6) = 210", f"{r6['microstates_total']}")
    chk(r6["check_passed"], "d6 自校验通过", r6["check"]["formula"])
    chk(r6["ground_term"] == "5D", "契约断言 Fe 基态基项 = 5D", "、".join(r6["terms_str"]))
    chk(r6["ground_state"] == "5D4", "契约断言 d6 基态 = 5D4（J=L+S=4）", r6["ground_state_pretty"])
    print(f"  d6 全部谱项：{'、'.join(r6['terms_str'])}")

    rp2 = spectroscopic_terms("p2")
    chk(set(rp2["terms_str"]) == {"1D", "3P", "1S"}, "p2 = 1D 3P 1S", "、".join(rp2["terms_str"]))
    chk(rp2["microstates_total"] == 15, "p2 微观状态总数 = 15", f"{rp2['microstates_total']}")
    chk(rp2["ground_state"] == "3P0", "p2 基态 = 3P0", rp2["ground_state_pretty"])

    r3 = spectroscopic_terms("d3")
    chk(r3["microstates_total"] == 120, "d3 微观状态总数 = 120", f"{r3['microstates_total']}")
    chk(r3["ground_term"] == "4F", "d3 基项 = 4F", "、".join(r3["terms_str"]))
    chk(r3["ground_state"] == "4F3/2", "d3 基态 4F3/2（J=|L-S|=3/2）", r3["ground_state_pretty"])
    chk(r3["check_passed"], "d3 自校验通过", r3["check"]["formula"])

    r5 = spectroscopic_terms("d5")
    chk(r5["microstates_total"] == 252, "d5 微观状态总数 = 252", f"{r5['microstates_total']}")
    chk(r5["ground_term"] == "6S", "d5 基项 = 6S（半满）", "、".join(r5["terms_str"]))
    chk(r5["ground_state"] == "6S5/2", "d5 基态 = 6S5/2", r5["ground_state_pretty"])
    chk(r5["check_passed"], "d5 自校验通过", r5["check"]["formula"])

    rpd = spectroscopic_terms("p d")
    chk(set(rpd["terms_str"]) == {"1P", "1D", "1F", "3P", "3D", "3F"},
        "非同科 p d 的 6 个谱项", "、".join(rpd["terms_str"]))
    chk(rpd["microstates_total"] == 60, "p d 微观状态总数 = 6×10 = 60", f"{rpd['microstates_total']}")
    chk(rpd["check_passed"], "p d 自校验通过", rpd["check"]["formula"])
    chk(rpd["ground_term"] == "3F", "p d 基项 = 3F（S 最大→L 最大）", rpd["ground_state"])

    rcu = spectroscopic_terms("3d10 4s1")
    chk(rcu["microstates_total"] == 2, "Cu 3d10 4s1 微观状态 = C(10,10)×C(2,1) = 2",
        f"{rcu['microstates_total']}")
    chk(rcu["ground_term"] == "2S", "契约断言 Cu 基项 = 2S", "、".join(rcu["terms_str"]))
    chk(rcu["ground_state"] == "2S1/2", "Cu 基态 = 2S1/2", rcu["ground_state_pretty"])
    chk(rcu["check_passed"], "Cu 自校验通过", rcu["check"]["formula"])

    r8 = spectroscopic_terms("d8")
    chk(set(r8["terms_str"]) == {"1G", "3F", "1D", "3P", "1S"}, "d8 与 d2 同项集（空穴等效）",
        "、".join(r8["terms_str"]))
    chk(r8["ground_state"] == "3F4", "d8 基态 3F4（超过半满 J=L+S）", r8["ground_state_pretty"])

    rp3 = spectroscopic_terms("p3")
    chk(rp3["microstates_total"] == 20, "p3 微观状态 = 20", f"{rp3['microstates_total']}")
    chk(rp3["ground_state"] == "4S3/2", "p3 半满基态 = 4S3/2", rp3["ground_state_pretty"])

    rd7 = spectroscopic_terms("d7")
    chk(rd7["microstates_total"] == 120, "d7 微观状态 = 120", f"{rd7['microstates_total']}")
    chk(rd7["ground_state"] == "4F9/2", "d7 基态 = 4F9/2（J=L+S=9/2）", rd7["ground_state_pretty"])

    rf2 = spectroscopic_terms("f2")
    chk(rf2["microstates_total"] == 91, "f2 微观状态 = C(14,2) = 91", f"{rf2['microstates_total']}")
    chk(set(rf2["terms_str"]) == {"1S", "1D", "1G", "1I", "3P", "3F", "3H"},
        "f2 = 1S 1D 1G 1I 3P 3F 3H", "、".join(rf2["terms_str"]))
    chk(rf2["ground_state"] == "3H4", "f2 基态 = 3H4", rf2["ground_state_pretty"])

    rf3 = spectroscopic_terms("f3")
    chk(rf3["microstates_total"] == 364, "f3 微观状态 = C(14,3) = 364",
        f"{rf3['microstates_total']}")
    chk(rf3["ground_term"] == "4I", "f3 基项 = 4I", "、".join(rf3["terms_str"])[:40])
    chk(rf3["ground_state"] == "4I9/2", "f3 基态 = 4I9/2", rf3["ground_state_pretty"])

    rf7 = spectroscopic_terms("f7")
    chk(rf7["microstates_total"] == 3432, "f7 微观状态 = C(14,7) = 3432",
        f"{rf7['microstates_total']}")
    chk(rf7["ground_state"] == "8S7/2", "f7 半满基态 = 8S7/2", rf7["ground_state_pretty"])
    chk(rf7["check_passed"], "f7 自校验通过（3432 个微观状态全部剥离）", rf7["check"]["formula"])

    chk(spectroscopic_terms("")["ok"] is False, "空输入返回 ok=False")
    chk(spectroscopic_terms("x9")["ok"] is False, "非法片段返回 ok=False")
    chk(spectroscopic_terms("d11")["ok"] is False, "电子数超过自旋轨道数时报错")

    print()
    print("=" * 78)
    print("Q5  superposition")
    print("=" * 78)
    c1, c2, c3 = 0.5, 0.5, math.sqrt(0.5)
    r = superposition(
        [
            {"coef": c1, "n": 2, "l": 1, "m": 0},
            {"coef": c2, "n": 3, "l": 1, "m": 1},
            {"coef": c3, "n": 3, "l": 1, "m": 0},
        ]
    )
    chk(r["ok"], "superposition 返回 ok")
    expect_au = -(c1 ** 2 / 8.0 + (c2 ** 2 + c3 ** 2) / 18.0)
    chk(close(r["energy_mean_au"], expect_au, 1e-12),
        "南开题公式 ⟨E⟩ = -(C1²/8 + (C2²+C3²)/18) a.u.",
        f"{r['energy_mean_au']:.10f} vs {expect_au:.10f} a.u.")
    chk(r["norm_check"]["normalized"] is True, "归一化检查通过", r["norm_check"]["message"])
    chk(close(sum(s["prob"] for s in r["eigen_probabilities"]), 1.0, 1e-12),
        "本征能量几率之和 = 1")
    chk(len(r["eigen_probabilities"]) == 2, "同 n 分量合并为一个本征能量",
        "、".join(s["eigen"] for s in r["eigen_probabilities"]))
    chk(abs(r["energy_mean_eV"] - r["energy_mean_eV_from_au"])
        <= 1e-3 * abs(r["energy_mean_eV_from_au"]),
        "eV 主值与 a.u. 换算值一致（差 ≤ 0.1%，即 13.6 vs R∞ 的常数口径差）",
        f"{r['energy_mean_eV']:.6f} eV vs {r['energy_mean_eV_from_au']:.6f} eV")
    chk(close(r["Lz_mean"], c2 ** 2, 1e-12), "⟨Lz⟩ = 0.25ħ（只留 m=1 分量）", r["Lz_mean_str"])
    chk(close(r["L2_mean"], 2.0, 1e-12), "⟨L²⟩ = 2ħ²（全部 l=1）", r["L2_mean_str"])
    print(f"  ⟨E⟩ = {r['energy_mean_au']:.8f} a.u. = {r['energy_mean_eV']:.6f} eV")

    rn = superposition(
        [
            {"coef": 0.8, "n": 2, "l": 1, "m": 0},
            {"coef": 0.6, "n": 3, "l": 1, "m": 1},
        ]
    )
    chk(rn["ok"] and rn["norm_check"]["normalized"] is True,
        "Σ|c|²=1.00（0.8²+0.6²=1）→ 归一化通过",
        rn["norm_check"]["message"])
    rn2 = superposition(
        [
            {"coef": 0.5, "n": 2, "l": 1, "m": 0},
            {"coef": 0.5, "n": 3, "l": 1, "m": 1},
        ]
    )
    chk(rn2["ok"] and rn2["norm_check"]["normalized"] is False
        and "未归一化" in rn2["norm_check"]["message"],
        "Σ|c|²=0.5 时明确提示未归一化", rn2["norm_check"]["message"])
    chk(close(rn2["energy_mean_au"], -0.5 / 8 - 0.5 / 18, 1e-12),
        "未归一化时按归一化处理（权重各 0.5）后再算 ⟨E⟩",
        f"{rn2['energy_mean_au']:.8f} a.u.")
    chk(close(rn2["energy_mean_raw_au"], -0.25 / 8 - 0.25 / 18, 1e-12),
        "同时保留不归一化的原始 Σ|c|²E_i 作对照",
        f"{rn2['energy_mean_raw_au']:.8f} a.u.")

    rs = superposition([{"coef": "1/√2", "n": 1, "l": 0, "m": 0},
                        {"coef": "1/sqrt(2)", "n": 2, "l": 0, "m": 0}])
    chk(rs["ok"] and rs["norm_check"]["normalized"],
        "系数支持 \"1/√2\" / \"1/sqrt(2)\" 写法", rs["norm_check"]["message"])

    rr = superposition([{"coef": 0.7071, "n": 2, "l": 1, "m": 0},
                        {"coef": 0.7071, "n": 3, "l": 1, "m": 1}])
    chk(rr["ok"] and rr["norm_check"]["severity"] == "rounding",
        "0.7071 这种舍入系数 → 判定为 rounding 而不是真错误",
        rr["norm_check"]["message"])
    chk(rn2_norm := (rn2["norm_check"]["severity"] == "unnormalized"),
        "0.5/0.5 这种真未归一化 → severity=unnormalized",
        rn2["norm_check"]["message"])

    chk(superposition([])["ok"] is False, "空列表返回 ok=False")
    chk(superposition([{"coef": 1.0, "n": 2, "l": 2, "m": 0}])["ok"] is False,
        "非法 l≥n 返回 ok=False")
    chk(superposition([{"coef": 1.0, "n": 2}])["ok"] is False, "缺量子数返回 ok=False")

    print()
    print("=" * 78)
    if FAILS:
        print(f"FAILED: {len(FAILS)} 项未通过")
        for name in FAILS:
            print(f"  - {name}")
        sys.exit(1)
    print("全部自测通过（契约 Q1–Q5 断言 + 附加硬门槛）")
    print("=" * 78)
