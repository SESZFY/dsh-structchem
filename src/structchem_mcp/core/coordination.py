# -*- coding: utf-8 -*-
"""coordination.py —— 配位化合物确定性计算（契约 3.5：D1–D2）。

算什么
------
D1 crystal_field          d^n 在八面体/四面体场中的排布、CFSE、自旋态、磁矩
D2 spectrochemical_series 配体在光谱化学序列中的位置与 Δo 相对强弱

口径（关键，答题时必须一致）
--------------------------
- 八面体：t2g 三个轨道能量 -4Dq、eg 两个轨道 +6Dq，Δo = 10Dq（重心守恒 3×(-4)+2×(+6)=0）。
- 四面体：Δt = (4/9)Δo，e 低 t2 高；由重心守恒得 e = -0.6Δt = -(8/3)Dq、
  t2 = +0.4Δt = +(16/9)Dq。四面体场恒为高自旋（Δt 太小，配不出低自旋）。
- CFSE(Dq) = Σ(占据数 × 该轨道能量/Dq)，八面体即 -4·n(t2g) + 6·n(eg)。
  另给 cfse_cm1 = cfse_dq × (Δo/10)，以及扣掉配对能后的 cfse_cm1_net。
- 磁矩用纯自旋公式 μ = √(n(n+2)) BM，不含轨道贡献。

边界（不做什么）
----------------
- 只支持八面体（octahedral）与四面体（tetrahedral）两种场。平面四方、四方锥等不做：
  那些场的 d 轨道分组与填充顺序另有一套，本模块不猜。
- Δo 若无法由 metal/ligand 推定，就不编数值：返回 delta_o_cm1=None，并把高自旋/低自旋
  两套结果并列放在 branches 里，让答题时自己选。
- metal/ligand 表里的 Δo、P 是**典型量级**（教材与文献常见值，不同来源差 ±10~20%），
  不是某个实测样品的数值；返回里都标了 source。
- 纯函数：无 IO、无全局可变状态。

单位：能量 cm^-1（光谱化学单位），磁矩 BM。
"""

from __future__ import annotations

import math

try:  # 包内正常导入
    from ._const import spin_only_moment
except ImportError:  # pragma: no cover - 直接 `python coordination.py` 时兜底
    import os as _os
    import sys as _sys

    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _SRC = _os.path.dirname(_os.path.dirname(_HERE))
    if _SRC not in _sys.path:
        _sys.path.insert(0, _SRC)
    from structchem_mcp.core._const import spin_only_moment

__all__ = ["crystal_field", "spectrochemical_series"]


# ---------------------------------------------------------------- 基础设施

def _ok(**kw):
    d = {"ok": True, "error": None}
    d.update(kw)
    return d


def _err(msg, **kw):
    d = {"ok": False, "error": msg}
    d.update(kw)
    return d


def _g(x, sig=6):
    if x is None:
        return "-"
    return f"{x:.{sig}g}"


def _as_float(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数值，收到 {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，收到 {x!r}")
    return v


# ------------------------------------------------------------ 场的能级结构

# 每个场给出按能量升序的轨道组：slots=轨道数，energy_dq=以八面体 Dq 为单位的能量
_FIELDS = {
    "octahedral": {
        "name_cn": "八面体场",
        "cn": 6,
        "groups": [
            {"name": "t2g", "slots": 3, "energy_dq": -4.0, "orb": "dxy, dxz, dyz"},
            {"name": "eg", "slots": 2, "energy_dq": 6.0, "orb": "dx²-y², dz²"},
        ],
        "delta_ratio": 1.0,          # Δ = 1.0Δo = 10Dq
        "delta_expr": "Δo = 10Dq",
        "allow_low_spin": True,
    },
    "tetrahedral": {
        "name_cn": "四面体场",
        "cn": 4,
        "groups": [
            {"name": "e", "slots": 2, "energy_dq": -8.0 / 3.0, "orb": "dx²-y², dz²"},
            {"name": "t2", "slots": 3, "energy_dq": 16.0 / 9.0, "orb": "dxy, dxz, dyz"},
        ],
        "delta_ratio": 4.0 / 9.0,    # Δt = (4/9)Δo
        "delta_expr": "Δt = (4/9)Δo",
        "allow_low_spin": False,
    },
}

_GEOM_ALIAS = {
    "octahedral": "octahedral", "oct": "octahedral", "oh": "octahedral",
    "octahedron": "octahedral", "八面体": "octahedral", "八面体场": "octahedral", "6": "octahedral",
    "tetrahedral": "tetrahedral", "tet": "tetrahedral", "td": "tetrahedral",
    "tetrahedron": "tetrahedral", "四面体": "tetrahedral", "四面体场": "tetrahedral", "4": "tetrahedral",
}


def _resolve_geometry(geometry):
    if not isinstance(geometry, str):
        raise ValueError(f"geometry 需要字符串（octahedral / tetrahedral），收到 {geometry!r}")
    key = geometry.strip().lower()
    key = _GEOM_ALIAS.get(key, key)
    if key not in _FIELDS:
        raise ValueError(
            f"geometry={geometry!r} 不支持。本模块只做八面体（octahedral）与四面体（tetrahedral）两种场。"
            "平面四方不在本工具范围：它的 d 轨道分组（dxz,dyz −5.14Dq / dxy −2.28Dq / "
            "dz² +2.28Dq / dx²−y² +12.28Dq）是以八面体重心为基准的相对值，"
            "与八面体的 −4Dq / +6Dq 不是同一把尺，混用会算出错误的 CFSE，必须另行推导。"
        )
    return key


def _parse_dn(x):
    """'d6' / '3d6' / 6 / '6' → 6。"""
    if isinstance(x, bool):
        raise ValueError(f"d_n 无法解析：收到布尔值 {x!r}")
    if isinstance(x, int):
        n = x
    else:
        s = str(x).strip().lower().replace(" ", "")
        if not s:
            raise ValueError("d_n 为空")
        import re as _re
        m = _re.search(r"d\s*([0-9]{1,2})", s)
        if m:
            n = int(m.group(1))
        else:
            try:
                n = int(float(s))
            except (TypeError, ValueError):
                raise ValueError(
                    f"d_n={x!r} 无法解析。请写成 'd6'、'3d6'、'd10' 或整数 6。"
                )
    if not (1 <= n <= 10):
        raise ValueError(f"d_n={n} 越界：d 亚层最多容纳 10 个电子，d^n 的 n 应在 1~10 之间")
    return n


# ------------------------------------------------------- 金属 / 配体的经验表
# 说明：Δo 为八面体水合离子的典型值（cm^-1），P 为配对能典型值（cm^-1）。
# 两者都是量级参考（教材与文献常见值，不同来源差 10~20%），不是实测样品数据。

_METAL_TABLE = {
    # key(小写): (显示名, Δo(cm^-1, 水合八面体), P(cm^-1))
    "ti3+": ("Ti³⁺", 20300.0, None),
    "ti2+": ("Ti²⁺", 12000.0, None),
    "v3+": ("V³⁺", 17700.0, 13000.0),
    "v2+": ("V²⁺", 12300.0, 16000.0),
    "cr3+": ("Cr³⁺", 17400.0, 23500.0),
    "cr2+": ("Cr²⁺", 13900.0, 23000.0),
    "mn3+": ("Mn³⁺", 21000.0, 28000.0),
    "mn2+": ("Mn²⁺", 21000.0, 25500.0),
    "fe3+": ("Fe³⁺", 13700.0, 30000.0),
    "fe2+": ("Fe²⁺", 10400.0, 17600.0),
    "co3+": ("Co³⁺", 19000.0, 21000.0),
    "co2+": ("Co²⁺", 9300.0, 22500.0),
    "ni2+": ("Ni²⁺", 8500.0, 20800.0),
    "cu2+": ("Cu²⁺", 12600.0, None),
}

_LIGAND_TABLE = {
    # key(小写): (显示名, Jørgensen 型相对因子 f，以 H2O=1.00 为基准)
    "i-": ("I⁻", 0.70), "br-": ("Br⁻", 0.72), "s2-": ("S²⁻", 0.725),
    "scn-": ("SCN⁻", 0.73), "cl-": ("Cl⁻", 0.78), "no3-": ("NO₃⁻", 0.80),
    "f-": ("F⁻", 0.90), "oh-": ("OH⁻", 0.92), "h2o": ("H₂O", 1.00),
    "ncs-": ("NCS⁻", 1.02), "nh3": ("NH₃", 1.25), "en": ("en", 1.28),
    "bipy": ("bipy", 1.33), "phen": ("phen", 1.34), "no2-": ("NO₂⁻", 1.45),
    "cn-": ("CN⁻", 1.70), "co": ("CO", 1.70),
}

_SUB_TO_ASCII = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                 "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
_SUP_TO_ASCII = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
                 "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                 "⁻": "-", "⁺": "+", "−": "-", "–": "-"}

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def _norm_species(s):
    """把 H₂O / NO₂⁻ / CN⁻ 之类统一成 ASCII 小写键。"""
    out = []
    for ch in str(s).strip():
        if ch in _SUB_TO_ASCII:
            out.append(_SUB_TO_ASCII[ch])
        elif ch in _SUP_TO_ASCII:
            out.append(_SUP_TO_ASCII[ch])
        elif ch in " \t":
            continue
        else:
            out.append(ch)
    return "".join(out).lower()


def _norm_metal(metal):
    """'Fe2+' / 'Fe(II)' / 'fe²⁺' → 表键 'fe2+'；无法识别返回 None。"""
    if metal is None:
        return None
    s = str(metal).strip()
    if not s:
        return None
    import re as _re
    m = _re.match(r"^([A-Za-z]{1,3})\s*[\(\[]\s*([IVX]+)\s*[\)\]]$", s)
    if m:
        rom = m.group(2).upper()
        if rom in _ROMAN:
            cand = (m.group(1) + str(_ROMAN[rom]) + "+").lower()
            if cand in _METAL_TABLE:
                return cand
    key = _norm_species(s)
    if key in _METAL_TABLE:
        return key
    return None


def _norm_ligand(ligand):
    if ligand is None:
        return None
    key = _norm_species(ligand)
    return key if key in _LIGAND_TABLE else None


# -------------------------------------------------------------- 电子填充

def _fill_electrons(n_electrons, groups, low_spin):
    """按能级升序填充；返回 {组名: 电子数}。

    高自旋：先把每组填到 slots（Hund 第一规则，全部平行单占），剩余电子自最低能组起配对。
    低自旋：把低能组填满（2×slots）再进下一组 —— 宁可配对也不上高能级。
    """
    occ = {g["name"]: 0 for g in groups}
    rest = n_electrons
    if low_spin:
        for g in groups:
            if rest <= 0:
                break
            add = min(rest, 2 * g["slots"])
            occ[g["name"]] = add
            rest -= add
    else:
        for g in groups:
            take = min(rest, g["slots"])
            occ[g["name"]] = take
            rest -= take
            if rest == 0:
                break
        for g in groups:                      # 剩余电子自最低能组配起
            if rest == 0:
                break
            room = 2 * g["slots"] - occ[g["name"]]
            add = min(rest, room)
            occ[g["name"]] += add
            rest -= add
    return occ


def _unpaired_in(occ, slots):
    if occ <= slots:
        return occ
    return 2 * slots - occ


def _pairs_in(occ, slots):
    return max(0, occ - slots)


def _config_str(occ, slots):
    """画轨道占据：↑↓ / ↑ / 空。"""
    nd = max(0, occ - slots)
    boxes = []
    for i in range(slots):
        if i < nd:
            boxes.append("↑↓")
        elif i < occ:
            boxes.append("↑")
        else:
            boxes.append("—")
    return " ".join(boxes)


# --------------------------------------------------------------- D1 晶体场

_SPIN_TABLE = {
    # 八面体高自旋（弱场）单电子数
    "oct_hs": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0},
    # 八面体低自旋（强场）单电子数
    "oct_ls": {1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 0, 7: 1, 8: 2, 9: 1, 10: 0},
    # 四面体（恒高自旋）
    "tet_hs": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0},
}


def crystal_field(d_n, geometry="octahedral", pairing_energy_cm1=None,
                  metal=None, ligand=None, delta_o_cm1=None, spin=None):
    """d^n 在八面体/四面体场中的分裂、电子排布、CFSE、自旋态与磁矩。

    参数
    ----
    d_n : "d6" / "3d6" / 6（1~10）
    geometry : "octahedral"（默认）/ "tetrahedral"
    pairing_energy_cm1 : 配对能 P。不给时按 metal 查典型值；仍无则据 Δo 定性
    metal : 中心离子，如 "Fe2+"、"Co3+"、"Mn2+"（决定 Δo 与 P 的典型值）
    ligand : 配体，按光谱化学序列给出相对 Δo 修正（H2O 为基准）
    delta_o_cm1 : 直接给 Δo，优先于 metal/ligand 的推定值
    spin : 显式指定 "high"/"low"（覆盖一切判据，用于题目已给自旋态的情形）

    返回
    ----
    d_configuration / splitting / delta_o_cm1 / cfse_dq / cfse_cm1 /
    spin_state / unpaired_electrons / magnetic_moment_BM / magnetism / steps
    """
    try:
        n = _parse_dn(d_n)
        geom_key = _resolve_geometry(geometry)
        field = _FIELDS[geom_key]
        groups = field["groups"]

        notes = []

        # --- 1. Δ（场分裂能）---
        delta = None
        delta_source = None
        if delta_o_cm1 is not None:
            delta_o = _as_float(delta_o_cm1, "delta_o_cm1")
            if delta_o <= 0:
                raise ValueError("delta_o_cm1 必须为正数（单位 cm^-1）")
            delta = delta_o * field["delta_ratio"]
            delta_source = f"直接给定 Δo = {_g(delta_o,6)} cm^-1"
        else:
            mk = _norm_metal(metal)
            lk = _norm_ligand(ligand)
            if metal is not None and mk is None:
                notes.append(f"中心离子 {metal!r} 不在本模块的典型值表内，Δo 无法推定（不做猜测）。")
            if ligand is not None and lk is None:
                notes.append(f"配体 {ligand!r} 不在本模块的典型值表内，Δo 无法推定（不做猜测）。")
            if mk is not None:
                base_do = _METAL_TABLE[mk][1]
                if lk is None or lk == "h2o":
                    delta_o = base_do
                    delta_source = (
                        f"[{_METAL_TABLE[mk][0]}(H₂O)₆] 的典型 Δo = {_g(base_do,6)} cm^-1"
                    )
                else:
                    f = _LIGAND_TABLE[lk][1]
                    delta_o = base_do * f
                    delta_source = (
                        f"以 {_METAL_TABLE[mk][0]} 水合离子 Δo={_g(base_do,6)} cm^-1 为基准，"
                        f"乘 {_LIGAND_TABLE[lk][0]} 的相对因子 {_g(f,3)}（Jørgensen 型量级）"
                        f" = {_g(delta_o,6)} cm^-1"
                    )
                delta = delta_o * field["delta_ratio"]
            elif lk is not None:
                # 只有配体：给不出数值 Δo，但能给相对强弱
                delta_source = f"只给配体 {_LIGAND_TABLE[lk][0]}：可定强弱次序，给不出 Δo 数值"
            else:
                delta_source = "未给 metal/ligand/Δo，Δo 未知"

        # --- 2. 配对能 P ---
        P = None
        P_source = None
        if pairing_energy_cm1 is not None:
            P = _as_float(pairing_energy_cm1, "pairing_energy_cm1")
            if P <= 0:
                raise ValueError("pairing_energy_cm1 必须为正数（单位 cm^-1）")
            P_source = f"直接给定 P = {_g(P,6)} cm^-1"
        else:
            mk = _norm_metal(metal)
            if mk is not None and _METAL_TABLE[mk][2] is not None:
                P = _METAL_TABLE[mk][2]
                P_source = f"{_METAL_TABLE[mk][0]} 的配对能典型值 P ≈ {_g(P,6)} cm^-1"
        if P is None and delta is not None:
            P = 20000.0
            P_source = "未给 P，按第一过渡系配对能量级取 P ≈ 20000 cm^-1"
            notes.append(
                "配对能 P 用的是第一过渡系量级默认值 20000 cm^-1。若题目给了 P，"
                "请传 pairing_energy_cm1 重算，自旋态的结论可能与这里不同。"
            )

        # --- 3. 自旋态 ---
        spin_state = None
        spin_short = None
        spin_basis = None
        branches = None
        has_two_states = n in (4, 5, 6, 7) and field["allow_low_spin"]

        explicit = None
        if spin is not None:
            s = str(spin).strip().lower()
            if s in ("high", "hs", "高自旋", "高", "weak", "弱场"):
                explicit = "high"
            elif s in ("low", "ls", "低自旋", "低", "strong", "强场"):
                explicit = "low"
            else:
                raise ValueError(
                    f"spin={spin!r} 无法识别，请用 'high'/'low'（或 高自旋/低自旋）"
                )
            if explicit == "low" and not field["allow_low_spin"]:
                raise ValueError(
                    f"{field['name_cn']}不存在低自旋构型：Δt = (4/9)Δo 太小，"
                    "成对所需能量 P 远大于分裂能，四面体配合物一律高自旋。"
                )
            if explicit == "low" and not has_two_states:
                notes.append(f"d{n} 在{field['name_cn']}中只有一种排布，高低自旋的区分对它没有意义。")

        if not has_two_states:
            if geom_key == "tetrahedral" and n in (4, 5, 6, 7):
                # 四面体场虽有 d4–d7，但 Δt 太小，永远配不出低自旋 —— 这是实质结论
                spin_state = "高自旋"
                spin_short = "HS"
                spin_basis = (
                    f"四面体场 Δt=(4/9)Δo 恒小于配对能 P，d{n} 在四面体场中一律高自旋，没有低自旋构型"
                )
            else:
                spin_state = "不区分（该 d^n 只有一种排布）"
                spin_short = "n/a"
                spin_basis = f"d{n} 填{field['name_cn']}时，无论 Δ 与 P 谁大，电子排布都唯一"
        elif explicit is not None:
            spin_state = "高自旋" if explicit == "high" else "低自旋"
            spin_short = "HS" if explicit == "high" else "LS"
            spin_basis = f"按输入 spin={spin!r} 指定"
        elif delta is not None and P is not None:
            if delta > P:
                spin_state, spin_short = "低自旋", "LS"
                spin_basis = f"Δ({_g(delta,6)} cm^-1) > P({_g(P,6)} cm^-1) → 电子宁可成对，低自旋"
            else:
                spin_state, spin_short = "高自旋", "HS"
                spin_basis = f"Δ({_g(delta,6)} cm^-1) < P({_g(P,6)} cm^-1) → 电子占高能级也不配对，高自旋"
        else:
            # 判不了就两套都给，不猜
            spin_basis = "Δ 与 P 无法同时确定，不能判自旋态；下面并列给两种构型"
            notes.append(
                "缺少 Δo 或 P，自旋态无法判定。返回值里 spin_state=None，"
                "高自旋/低自旋两套结果都在 branches 中，请按题目条件选用。"
            )

        def build(state):
            """state: 'high' / 'low' / None（None 时按高自旋占位）。"""
            low = (state == "low")
            occ = _fill_electrons(n, groups, low_spin=low)
            unpaired = sum(_unpaired_in(occ[g["name"]], g["slots"]) for g in groups)
            pairs = sum(_pairs_in(occ[g["name"]], g["slots"]) for g in groups)
            cfse_dq = sum(occ[g["name"]] * g["energy_dq"] for g in groups)
            split = {}
            for g in groups:
                split[g["name"]] = {
                    "electrons": occ[g["name"]],
                    "slots": g["slots"],
                    "orbitals": g["orb"],
                    "spin_config": _config_str(occ[g["name"]], g["slots"]),
                    "energy_dq": g["energy_dq"],
                    "unpaired": _unpaired_in(occ[g["name"]], g["slots"]),
                }
            return {
                "occ": occ, "unpaired": unpaired, "pairs": pairs,
                "cfse_dq": cfse_dq, "splitting": split,
            }

        if spin_state == "低自旋":
            main = build("low")
        elif spin_state == "高自旋":
            main = build("high")
        else:
            # d1-d3/d8-d10：唯一排布；或未知时按高自旋占位（branches 另给两套）
            main = build("high")

        if not has_two_states and spin_state is None:
            spin_state = "不区分（该 d^n 只有一种排布）"
            spin_short = "n/a"

        if has_two_states and spin_state is None:
            hs, ls = build("high"), build("low")
            branches = {
                "high_spin": {
                    "spin_state": "高自旋", "splitting": hs["splitting"],
                    "unpaired_electrons": hs["unpaired"],
                    "magnetic_moment_BM": spin_only_moment(hs["unpaired"]),
                    "cfse_dq": hs["cfse_dq"], "n_pairs": hs["pairs"],
                },
                "low_spin": {
                    "spin_state": "低自旋", "splitting": ls["splitting"],
                    "unpaired_electrons": ls["unpaired"],
                    "magnetic_moment_BM": spin_only_moment(ls["unpaired"]),
                    "cfse_dq": ls["cfse_dq"], "n_pairs": ls["pairs"],
                },
            }

        # --- 4. CFSE / 磁矩 ---
        cfse_dq = main["cfse_dq"]
        unpaired = main["unpaired"]
        mu = spin_only_moment(unpaired)
        cfse_cm1 = None if delta is None else cfse_dq * (delta / 10.0)
        cfse_net_cm1 = None
        if delta is not None and P is not None:
            cfse_net_cm1 = cfse_cm1 - main["pairs"] * P

        # 自校验：与教科书 d^n 单电子数表对照
        if spin_state == "低自旋":
            expect = _SPIN_TABLE["oct_ls"][n] if geom_key == "octahedral" else None
        elif geom_key == "tetrahedral":
            expect = _SPIN_TABLE["tet_hs"][n]
        else:
            expect = _SPIN_TABLE["oct_hs"][n] if spin_state == "高自旋" else None
        if expect is not None:
            assert unpaired == expect, (geom_key, n, spin_state, unpaired, expect)

        # --- 5. steps ---
        delta_symbol = "Δo" if geom_key == "octahedral" else "Δt"
        steps = []
        if geom_key == "octahedral":
            steps.append(
                "① d 轨道在八面体场中分裂成两组：t2g（dxy、dxz、dyz，3 个轨道，能量 -4Dq）"
                "与 eg（dx²-y²、dz²，2 个轨道，能量 +6Dq），Δo = 10Dq，重心守恒 3×(-4)+2×(+6)=0"
            )
        else:
            steps.append(
                "① d 轨道在四面体场中分裂成两组：e（dx²-y²、dz²，2 个，能量 -0.6Δt）"
                "与 t2（dxy、dxz、dyz，3 个，能量 +0.4Δt）；Δt = (4/9)Δo，重心 2×(-0.6)+3×(+0.4)=0"
            )
        steps.append(f"② 分裂能：{delta_symbol} = {_g(delta,6) if delta is not None else '未知'} cm^-1"
                     + (f"（{delta_source}）" if delta_source else ""))
        steps.append(f"③ 配对能 P = {_g(P,6) if P is not None else '未知'} cm^-1"
                     + (f"（{P_source}）" if P_source else ""))
        steps.append(f"④ 自旋态判据：{spin_basis}")

        fill_txt = "，".join(
            f"{g['name']}^{occ_n}（{_config_str(main['occ'][g['name']], g['slots'])}）"
            for g, occ_n in ((g, main["occ"][g["name"]]) for g in groups)
        )
        steps.append(f"⑤ 电子排布（d{n}）：{fill_txt}，未成对电子数 n = {unpaired}")

        cfse_terms = " + ".join(
            f"({g['energy_dq']:.4g})×{main['occ'][g['name']]}" for g in groups
        )
        steps.append(
            f"⑥ CFSE = Σ(占据数×轨道能量) = {cfse_terms} = {_g(cfse_dq,6)} Dq"
            + (f"（= {_g(cfse_dq/10.0,4)}Δo）" if geom_key == "octahedral" else "")
        )
        if cfse_cm1 is not None:
            steps.append(
                f"⑦ 换算成波数：1Dq = Δ/10 = {_g(delta/10.0,6)} cm^-1，"
                f"CFSE = {_g(cfse_dq,6)} × {_g(delta/10.0,6)} = {_g(cfse_cm1,6)} cm^-1"
            )
            if cfse_net_cm1 is not None:
                steps.append(
                    f"⑧ 【选修口径，一般答题不必写】若把配对能也计入（净稳定化能）："
                    f"CFSE_net = CFSE - n(电子对)×P = {_g(cfse_cm1,6)} - "
                    f"{main['pairs']}×{_g(P,6)} = {_g(cfse_net_cm1,6)} cm^-1"
                )
        steps.append(
            f"⑨ 磁矩：μ = sqrt(n(n+2)) = sqrt({unpaired}×{unpaired+2}) = "
            f"{_g(mu,4)} BM → {'顺磁' if unpaired else '抗磁'}"
        )

        # --- 6. 教材要点 ---
        if n == 5 and spin_state == "高自旋":
            notes.append(
                "d5 高自旋（如 Mn²⁺）的 CFSE = 0：电子半充满五个轨道，能量重心不降低。"
                "这也是 Mn²⁺ 配合物颜色极浅、且 d-d 跃迁自旋禁阻的原因。"
            )
        if n == 3 or n == 8:
            notes.append(
                f"d{n} 的 CFSE = {_g(cfse_dq,4)} Dq 是八面体场中绝对值最大的"
                "（d3 与 d8 同为 -1.2Δo），所以 Cr³⁺、Ni²⁺ 的八面体配合物特别稳定。"
            )
        notes.append(
            "磁矩用纯自旋公式 μ=√(n(n+2)) BM，未含轨道贡献。Co²⁺、Fe²⁺ 等因自旋-轨道耦合，"
            "实测值会偏离这个数（如八面体 Co²⁺ 实测常在 4.7~5.2 BM，而 n=3 的自旋-only 值是 3.87 BM）。"
        )
        if geom_key == "tetrahedral":
            notes.append(
                "四面体场没有低自旋：Δt 只有 Δo 的 4/9，成对能 P 始终大于它。"
                "又因 d-d 跃迁是宇称禁阻（无对称中心），四面体配合物颜色通常比八面体浅。"
            )

        return _ok(
            d_n=n,
            d_configuration=f"d{n}",
            geometry=geom_key,
            geometry_name=field["name_cn"],
            coordination_number=field["cn"],
            splitting=main["splitting"],
            delta_cm1=delta,
            delta_symbol=delta_symbol,
            delta_o_cm1=(None if delta is None else delta / field["delta_ratio"]),
            delta_o_given_cm1=(None if delta_o_cm1 is None else _as_float(delta_o_cm1, "delta_o_cm1")),
            delta_t_cm1=(None if delta is None else delta),
            delta_ratio_to_delta_o=field["delta_ratio"],
            delta_expr=field["delta_expr"],
            delta_source=delta_source,
            pairing_energy_cm1=P,
            pairing_source=P_source,
            spin_state=spin_state,
            spin_short=spin_short,
            spin_basis=spin_basis,
            branches=branches,
            n_electron_pairs=main["pairs"],
            cfse_dq=cfse_dq,
            cfse_dq_str=f"{_g(cfse_dq,6)} Dq",
            cfse_cm1=cfse_cm1,
            cfse_cm1_net=cfse_net_cm1,
            unpaired_electrons=unpaired,
            magnetic_moment_BM=mu,
            magnetic_moment_str=f"{_g(mu,4)} BM",
            magnetism=("顺磁" if unpaired else "抗磁"),
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# ------------------------------------------------- D2 光谱化学序列

# 由弱到强。CN⁻ 与 CO 强场端接近相等，按教材写法合并为末项。
_SERIES = [
    {"key": "i-", "display": "I⁻", "name": "碘离子", "cn": "单齿，卤素",
     "aliases": ["i", "碘", "碘离子", "iodide"]},
    {"key": "br-", "display": "Br⁻", "name": "溴离子", "cn": "单齿，卤素",
     "aliases": ["br", "溴", "溴离子", "bromide"]},
    {"key": "s2-", "display": "S²⁻", "name": "硫离子", "cn": "单齿",
     "aliases": ["s2", "s", "硫", "硫离子", "sulfide"]},
    {"key": "scn-", "display": "SCN⁻", "name": "硫氰根（以 S 配位）", "cn": "两可配体，S 端",
     "aliases": ["scn", "硫氰根", "硫氰酸根"]},
    {"key": "cl-", "display": "Cl⁻", "name": "氯离子", "cn": "单齿，卤素",
     "aliases": ["cl", "氯", "氯离子", "chloride"]},
    {"key": "no3-", "display": "NO₃⁻", "name": "硝酸根", "cn": "单齿/两齿",
     "aliases": ["no3", "硝酸根", "硝酸"]},
    {"key": "f-", "display": "F⁻", "name": "氟离子", "cn": "单齿，卤素",
     "aliases": ["f", "氟", "氟离子", "fluoride"]},
    {"key": "oh-", "display": "OH⁻", "name": "氢氧根", "cn": "单齿",
     "aliases": ["oh", "氢氧根", "羟基", "氢氧"]},
    {"key": "h2o", "display": "H₂O", "name": "水", "cn": "单齿（O 配位）",
     "aliases": ["水", "h2o", "aqua", "水分子"]},
    {"key": "ncs-", "display": "NCS⁻", "name": "异硫氰根（以 N 配位）", "cn": "两可配体，N 端",
     "aliases": ["ncs", "异硫氰根"]},
    {"key": "nh3", "display": "NH₃", "name": "氨", "cn": "单齿（N 配位）",
     "aliases": ["氨", "nh3", "ammine", "氨分子"]},
    {"key": "en", "display": "en", "name": "乙二胺（H₂NCH₂CH₂NH₂）", "cn": "双齿（N,N）",
     "aliases": ["乙二胺", "ethylenediamine", "en"]},
    {"key": "bipy", "display": "bipy", "name": "2,2′-联吡啶", "cn": "双齿（N,N）",
     "aliases": ["联吡啶", "bpy", "2,2'-bipyridine", "bipy"]},
    {"key": "phen", "display": "phen", "name": "1,10-邻菲咯啉", "cn": "双齿（N,N）",
     "aliases": ["邻菲咯啉", "邻二氮菲", "o-phenanthroline", "phen"]},
    {"key": "no2-", "display": "NO₂⁻", "name": "硝基（以 N 配位）", "cn": "单齿（N 配位）",
     "aliases": ["no2", "硝基", "亚硝酸根"]},
    {"key": "cn-", "display": "CN⁻ ≈ CO", "name": "氰根 / 一氧化碳", "cn": "单齿（C 配位）",
     "aliases": ["cn", "氰", "氰根", "氰离子", "co", "一氧化碳", "carbonyl", "羰基"]},
]

_SERIES_STR = " < ".join(item["display"] for item in _SERIES)

_LIGAND_F_FACTOR = {item["key"]: None for item in _SERIES}
for _k, (_disp, _f) in _LIGAND_TABLE.items():
    _LIGAND_F_FACTOR[_k] = _f


def _strength_band(idx):
    """按序列位置给场强分档（1 基）。"""
    if idx <= 8:
        return "弱场", "Δ 小，一般给出高自旋配合物"
    if idx <= 12:
        return "中等场", "Δ 居中，自旋态取决于中心离子（如 en/NH₃ 对 Co³⁺ 已是低自旋，对 Ni²⁺ 仍高自旋）"
    return "强场", "Δ 大，一般给出低自旋配合物"


def spectrochemical_series(ligand):
    """配体在光谱化学序列中的位置 + Δo 的相对大小 + 例子。

    序列（弱 → 强）：
        I⁻ < Br⁻ < S²⁻ < SCN⁻ < Cl⁻ < NO₃⁻ < F⁻ < OH⁻ < H₂O < NCS⁻ < NH₃ < en < bipy < phen
        < NO₂⁻ < CN⁻ ≈ CO
    """
    try:
        if ligand is None:
            raise ValueError("ligand 为空：请给一个配体（如 'H2O'、'NH3'、'CN-'）")
        raw = str(ligand).strip()
        if not raw:
            raise ValueError("ligand 为空：请给一个配体（如 'H2O'、'NH3'、'CN-'）")

        key = _norm_species(raw)
        # 先按 key 直接匹配，再按别名匹配
        hits = [i for i, item in enumerate(_SERIES) if item["key"] == key]
        if not hits:
            hits = [
                i for i, item in enumerate(_SERIES)
                if any(_norm_species(a) == key for a in item["aliases"])
            ]
        if not hits:
            raise ValueError(
                f"配体 {ligand!r} 不在光谱化学序列里（或本模块的表未收录）。"
                f"本模块收录：{_SERIES_STR}"
            )

        idx0 = hits[0]                    # 0 基
        item = _SERIES[idx0]
        idx = idx0 + 1
        total = len(_SERIES)
        strength, strength_note = _strength_band(idx)
        weaker = [s["display"] for s in _SERIES[:idx0]]
        stronger = [s["display"] for s in _SERIES[idx0 + 1:]]

        notes = []
        ambiguous = None
        if len(hits) > 1:
            ambiguous = [_SERIES[i]["display"] for i in hits]
            notes.append(
                f"{item['display']} 在序列中出现两次（S 配位端与 N 配位端各一个位置），"
                "本工具取靠前的（弱场端）那个：分子式写法相同时，配位原子不同 Δo 差得很远。"
            )
        if key == "scn-":
            notes.append(
                "SCN⁻ 是两可配体：以 S 配位时是弱场（第 4 位），以 N 配位时（NCS⁻）升到第 10 位。"
                "写答案时要说明配位原子。"
            )
        if key in ("cn-", "co"):
            notes.append(
                "CN⁻ 与 CO 都是强场端，Δo 大到足以把第一过渡系的 d4~d7 全部压成低自旋"
                "（如 [Fe(CN)₆]⁴⁻ 中的 Fe²⁺ 是低自旋、抗磁）。"
            )
        if key == "h2o":
            notes.append("H₂O 是序列的分界基准：比它弱的算弱场，比它强的算强场（相对口径）。")
        notes.append(
            "序列只给 Δo 的**相对次序**，不给绝对数值。要算具体 Δ 需配合中心离子"
            "（不同金属的 Δo 绝对值可差数倍）。"
        )

        # 相对位置指数（0=最弱端，1=最强端），完全由序列位置导出，不掺外部常数
        rel = idx0 / (total - 1)

        steps = [
            f"① 光谱化学序列（由弱到强）：{_SERIES_STR}",
            f"② 定位：{item['display']}（{item['name']}）在序列第 {idx} 位，共 {total} 位",
            f"③ 相对位置：({idx}−1)/({total}−1) = {_g(rel,3)}（0=最弱端，1=最强端）",
            f"④ 场强分档：第 1–8 位为弱场、第 9–12 位为中等场、第 13–16 位为强场 → {strength}",
            f"⑤ Δo 相对大小：比它弱的有 {('、'.join(weaker) if weaker else '无（它就是最弱端）')}；"
            f"比它强的有 {('、'.join(stronger) if stronger else '无（它就是最强端）')}",
        ]

        examples_map = {
            "i-": ["[Ti(I)₆]³⁻ 类极少见：I⁻ 是最弱场配体，Δo 小，几乎总是高自旋"],
            "br-": ["[CrBr₆]³⁻、[NiBr₄]²⁻（四面体）"],
            "s2-": ["[NiS₄]²⁻ 类硫代配合物"],
            "scn-": ["[Fe(SCN)₆]³⁻（S 配位，弱场）"],
            "cl-": ["[CrCl₆]³⁻、[FeCl₄]⁻（四面体、高自旋）、[CoCl₄]²⁻（蓝色）"],
            "no3-": ["[Co(NO₃)₄]²⁻ 类"],
            "f-": ["[CoF₆]³⁻（高自旋）、[FeF₆]³⁻（无色，高自旋）"],
            "oh-": ["[Cu(OH)₄]²⁻、[Cr(OH)₆]³⁻"],
            "h2o": ["[Fe(H₂O)₆]²⁺（高自旋，浅绿）、[Ni(H₂O)₆]²⁺（绿色）、[Ti(H₂O)₆]³⁺（紫色）"],
            "ncs-": ["[Fe(NCS)₆]³⁻（N 配位，比 SCN⁻ 强）、[Cr(NCS)₆]³⁻（血红色）"],
            "nh3": ["[Co(NH₃)₆]³⁺（低自旋，橙黄）、[Ni(NH₃)₆]²⁺（高自旋，蓝紫）"],
            "en": ["[Co(en)₃]³⁺（低自旋，有旋光异构）、[Ni(en)₃]²⁺（高自旋）"],
            "bipy": ["[Fe(bipy)₃]²⁺（低自旋，红色）"],
            "phen": ["[Fe(phen)₃]²⁺（低自旋，深红，邻二氮菲亚铁指示剂）"],
            "no2-": ["[Co(NO₂)₆]³⁻（低自旋，黄色）"],
            "cn-": ["[Fe(CN)₆]⁴⁻（低自旋、抗磁，黄色）、[Fe(CN)₆]³⁻（低自旋，n=1，深红）"],
        }

        return _ok(
            ligand=raw,
            matched=item["key"],
            display=item["display"],
            name=item["name"],
            ligand_type=item["cn"],
            index=idx,
            total=total,
            position=f"第 {idx} 位（共 {total} 位，由弱到强）",
            strength=strength,
            strength_note=strength_note,
            relative_position=rel,
            relative_position_str=f"{_g(rel,3)}（0=最弱端，1=最强端）",
            delta_o_relative=f"{item['display']} 的 Δo 介于 {len(weaker)} 个更弱与 {len(stronger)} 个更强的配体之间",
            weaker_ligands=weaker,
            stronger_ligands=stronger,
            neighbor_weaker=(_SERIES[idx0 - 1]["display"] if idx0 > 0 else None),
            neighbor_stronger=(_SERIES[idx0 + 1]["display"] if idx0 + 1 < total else None),
            jorgensen_factor=_LIGAND_F_FACTOR.get(item["key"]),
            full_series=_SERIES_STR,
            examples=examples_map.get(item["key"], []),
            ambiguous_matches=ambiguous,
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# ------------------------------------------------------------------ 自测

if __name__ == "__main__":
    import sys as _sys

    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    f_pass = f_fail = 0

    def check(name, cond, detail=""):
        global f_pass, f_fail
        if cond:
            f_pass += 1
            print(f"[PASS] {name}")
        else:
            f_fail += 1
            print(f"[FAIL] {name}  {detail}")

    print("=" * 66)
    print("coordination.py 自测：契约第 4 节断言 + 已知答案")
    print("=" * 66)

    # --- 契约断言：d6 八面体弱场 → 4 个单电子、μ=4.90 BM
    r = crystal_field("d6", "octahedral", metal="Fe2+")
    check("crystal_field d6 oct 弱场(Fe2+) → 4 个单电子",
          r["ok"] and r["unpaired_electrons"] == 4,
          repr(r.get("error") or r.get("unpaired_electrons")))
    check("crystal_field d6 oct 弱场 → μ = 4.90 BM",
          r["ok"] and abs(r["magnetic_moment_BM"] - 4.899) < 0.002,
          repr(r.get("error") or r.get("magnetic_moment_BM")))
    check("crystal_field d6 oct 弱场 → 高自旋、顺磁、splitting 是 t2g⁴eg²",
          r["ok"] and r["spin_state"] == "高自旋" and r["magnetism"] == "顺磁"
          and r["splitting"]["t2g"]["electrons"] == 4 and r["splitting"]["eg"]["electrons"] == 2,
          repr(r.get("error") or r.get("splitting")))
    check("crystal_field d6 oct 弱场 → CFSE = -4 Dq = -0.4Δo",
          r["ok"] and abs(r["cfse_dq"] - (-4.0)) < 1e-9,
          repr(r.get("error") or r.get("cfse_dq")))

    # --- d6 强场 → 0 个单电子、抗磁
    r = crystal_field("d6", "octahedral", metal="Co3+", ligand="CN-")
    check("crystal_field d6 oct 强场(Co3+/CN-) → 0 个单电子、抗磁",
          r["ok"] and r["unpaired_electrons"] == 0 and r["magnetism"] == "抗磁"
          and r["spin_state"] == "低自旋",
          repr(r.get("error") or (r.get("unpaired_electrons"), r.get("spin_state"))))
    check("crystal_field d6 oct 强场 → CFSE = -24 Dq",
          r["ok"] and abs(r["cfse_dq"] - (-24.0)) < 1e-9,
          repr(r.get("error") or r.get("cfse_dq")))

    # --- d5 弱场 Mn2+ → 5 个单电子、μ=5.92
    r = crystal_field("d5", "octahedral", metal="Mn2+")
    check("crystal_field d5 oct 弱场(Mn2+) → 5 个单电子、μ=5.92 BM",
          r["ok"] and r["unpaired_electrons"] == 5
          and abs(r["magnetic_moment_BM"] - 5.916) < 0.005,
          repr(r.get("error") or (r.get("unpaired_electrons"), r.get("magnetic_moment_BM"))))
    check("crystal_field d5 高自旋 → CFSE = 0 Dq",
          r["ok"] and abs(r["cfse_dq"]) < 1e-9, repr(r.get("error") or r.get("cfse_dq")))

    # --- d1..d10 八面体高/低自旋单电子数全表
    hs_expect = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    ls_expect = [1, 2, 3, 2, 1, 0, 1, 2, 1, 0]
    ok_hs = ok_ls = True
    for n in range(1, 11):
        rh = crystal_field(f"d{n}", "octahedral", spin="high")
        if not rh["ok"] or rh["unpaired_electrons"] != hs_expect[n - 1]:
            ok_hs = False
            print(f"    d{n} 高自旋得 {rh.get('unpaired_electrons')}，应为 {hs_expect[n-1]}")
        if n in (4, 5, 6, 7):
            rl = crystal_field(f"d{n}", "octahedral", spin="low")
            if not rl["ok"] or rl["unpaired_electrons"] != ls_expect[n - 1]:
                ok_ls = False
                print(f"    d{n} 低自旋得 {rl.get('unpaired_electrons')}，应为 {ls_expect[n-1]}")
    check("crystal_field 八面体 d1–d10 高自旋单电子数 1,2,3,4,5,4,3,2,1,0", ok_hs)
    check("crystal_field 八面体 d4–d7 低自旋单电子数 2,1,0,1", ok_ls)

    # --- 四面体
    r = crystal_field("d6", "tetrahedral", metal="Fe2+")
    check("crystal_field d6 四面体 → 4 个单电子（恒高自旋）",
          r["ok"] and r["unpaired_electrons"] == 4 and r["spin_state"] == "高自旋",
          repr(r.get("error") or r.get("unpaired_electrons")))
    check("crystal_field 四面体 Δt = (4/9)Δo",
          r["ok"] and abs(r["delta_cm1"] - r["delta_o_cm1"] * 4 / 9) < 1e-9,
          repr(r.get("error") or (r.get("delta_cm1"), r.get("delta_o_cm1"))))
    check("crystal_field 四面体 CFSE(d1) = -8/3 Dq（= -0.6Δt）",
          crystal_field("d1", "tetrahedral", metal="Ti3+")["cfse_dq"] == -8.0 / 3.0)
    r = crystal_field("d6", "tetrahedral", metal="Fe2+", spin="low")
    check("crystal_field 四面体禁止低自旋 → ok=False",
          r["ok"] is False and isinstance(r["error"], str))

    # --- CFSE 与教材对照
    cfse_cases = [("d3", "octahedral", "high", -12.0), ("d8", "octahedral", "high", -12.0),
                  ("d5", "octahedral", "high", 0.0), ("d6", "octahedral", "high", -4.0),
                  ("d10", "octahedral", "high", 0.0)]
    for dn, geo, sp, want in cfse_cases:
        r = crystal_field(dn, geo, spin=sp)
        check(f"crystal_field {dn} {geo} {sp} → CFSE={want} Dq",
              r["ok"] and abs(r["cfse_dq"] - want) < 1e-9,
              repr(r.get("error") or r.get("cfse_dq")))

    # --- Δo 推定链与自旋判据
    r = crystal_field("d6", "octahedral", metal="Fe2+")
    check("crystal_field Δo/P 都标了来源",
          r["ok"] and r["delta_source"] and r["pairing_source"]
          and abs(r["delta_o_cm1"] - 10400) < 1e-6)
    r = crystal_field("d6", "octahedral", delta_o_cm1=30000, pairing_energy_cm1=17600)
    check("crystal_field 直接给 Δo=30000 > P=17600 → 低自旋",
          r["ok"] and r["spin_state"] == "低自旋" and r["unpaired_electrons"] == 0,
          repr(r.get("error") or r.get("spin_state")))

    r = crystal_field("d6", "octahedral")
    check("crystal_field 无 metal/ligand/Δo → 不猜：spin_state=None 且给两分支",
          r["ok"] and r["spin_state"] is None and r["branches"] is not None
          and r["branches"]["high_spin"]["unpaired_electrons"] == 4
          and r["branches"]["low_spin"]["unpaired_electrons"] == 0,
          repr(r.get("error") or (r.get("spin_state"), r.get("branches"))))

    # --- steps 完整性
    r = crystal_field("d6", "octahedral", metal="Fe2+")
    check("crystal_field steps 覆盖分裂/判据/排布/CFSE/磁矩（≥8 条）",
          r["ok"] and len(r["steps"]) >= 8
          and any("CFSE" in s for s in r["steps"])
          and any("μ" in s or "BM" in s for s in r["steps"]))

    # --- D2 序列
    r = spectrochemical_series("H2O")
    check("spectrochemical_series H2O → 第 9 位、中等场",
          r["ok"] and r["index"] == 9 and r["strength"] == "中等场",
          repr(r.get("error") or (r.get("index"), r.get("strength"))))

    r = spectrochemical_series("CN-")
    check("spectrochemical_series CN- → 末位（第 16 位）、强场",
          r["ok"] and r["index"] == 16 and r["strength"] == "强场",
          repr(r.get("error") or (r.get("index"), r.get("strength"))))

    r = spectrochemical_series("I-")
    check("spectrochemical_series I- → 第 1 位（最弱）、无更弱者",
          r["ok"] and r["index"] == 1 and r["weaker_ligands"] == [],
          repr(r.get("error") or r.get("index")))

    r = spectrochemical_series("NH3")
    check("spectrochemical_series NH3 → 位于 H2O 之后，weaker 含 H2O",
          r["ok"] and "H₂O" in r["weaker_ligands"]
          and r["neighbor_weaker"] == "NCS⁻",
          repr(r.get("error") or r.get("weaker_ligands")))

    r = spectrochemical_series("SCN-")
    check("spectrochemical_series SCN- → 第 4 位且提示 N 配位端",
          r["ok"] and r["index"] == 4
          and any("N 配位" in s for s in r["notes"]),
          repr(r.get("error") or r.get("index")))

    for alias, want_idx in (("h2o", 9), ("水", 9), ("H₂O", 9), ("氨", 11),
                            ("no2-", 15), ("NO₂⁻", 15), ("co", 16), ("氰根", 16),
                            ("乙二胺", 12), ("bpy", 13)):
        r = spectrochemical_series(alias)
        check(f"spectrochemical_series {alias!r} → 第 {want_idx} 位",
              r["ok"] and r["index"] == want_idx,
              repr(r.get("error") or r.get("index")))

    # 序列单调性自校验
    factors = [_LIGAND_F_FACTOR[s["key"]] for s in _SERIES if _LIGAND_F_FACTOR[s["key"]] is not None]
    check("光谱化学序列的位置次序与相对因子次序一致",
          all(factors[i] <= factors[i + 1] + 1e-9 for i in range(len(factors) - 1)),
          repr(factors))

    # --- 非法输入
    bad = [
        ("d_n 越界", crystal_field("d11")),
        ("d_n 无法解析", crystal_field("xyz")),
        ("geometry 不支持", crystal_field("d6", "square_planar")),
        ("配体不在表内", spectrochemical_series("Xx")),
        ("配体为空", spectrochemical_series("")),
        ("负 Δo", crystal_field("d6", delta_o_cm1=-100)),
    ]
    for name, r in bad:
        check(name + " → ok=False 且有中文原因",
              r["ok"] is False and isinstance(r["error"], str) and len(r["error"]) > 4,
              repr(r))

    print("-" * 66)
    print(f"通过 {f_pass} 项，失败 {f_fail} 项")
    print("=" * 66)
    if f_fail:
        _sys.exit(1)
