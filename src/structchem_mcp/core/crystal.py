# -*- coding: utf-8 -*-
"""crystal.py —— 晶体结构与晶胞确定性计算（契约 3.4：C1–C6）。

算什么
------
C1 unit_cell        7 晶系判定（带容差）+ 晶胞体积
C2 miller_indices   截距 → Miller 指标（分数截距 / ∞ / 负指数）
C3 d_spacing        按晶系选公式算晶面间距，并回述用了哪条公式
C4 packing          5 种堆积方式的原子数 / CN / 空间利用率 / 空隙数
C5 radius_ratio     半径比规则 → 配位数 + 配位多面体
C6 cell_content     式量 + 晶胞参数 → Z / 密度 / 离子空间利用率

边界（不做什么）
----------------
- 晶胞参数只定「晶系 + 简单(P)型定向」。带心（I/F/C/R）判定需要原子坐标，
  本模块在 possible_bravais 里列候选，不猜。
- 半径比规则是几何判据给出的**预测**，不是实测配位数的预言；比值贴近临界值时
  额外给 critical_band 提示，并在 4 配位段附「核间距被当成离子半径」的排查话术。
- cell_content 的离子空间利用率只在 AB 型（每式量 1 阳 + 1 阴）成立，其他型式给 None 并说明。
- 纯函数：无 IO、无网络、无全局可变状态。

单位：长度 pm（同给 Å），体积 pm^3（同给 Å^3），密度 g/cm^3。
常数一律取自 core/_const.py。
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

try:  # 包内正常导入
    from ._const import NA, PM_PER_A
except ImportError:  # pragma: no cover - 直接 `python crystal.py` 时兜底
    import os as _os
    import sys as _sys

    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _SRC = _os.path.dirname(_os.path.dirname(_HERE))
    if _SRC not in _sys.path:
        _sys.path.insert(0, _SRC)
    from structchem_mcp.core._const import NA, PM_PER_A

__all__ = [
    "unit_cell",
    "miller_indices",
    "d_spacing",
    "packing",
    "radius_ratio",
    "cell_content",
]

PM3_TO_A3 = float(PM_PER_A) ** 3      # 1 Å^3 = 1e6 pm^3
PM3_TO_CM3 = 1e-30                    # 1 pm^3 = 1e-30 cm^3
DEFAULT_TOL = 0.005                   # 边长/角度判等相对容差 0.5%


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
    """数值 → 紧凑字符串（不用科学计数法糊弄人）。"""
    if x is None:
        return "-"
    if isinstance(x, Fraction):
        x = float(x)
    return f"{x:.{sig}g}"


def _as_float(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数值，收到 {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，收到 {x!r}")
    return v


def _as_int(x, name):
    if isinstance(x, bool):
        raise ValueError(f"{name} 必须是整数，收到 {x!r}")
    if isinstance(x, int):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数，收到 {x!r}")
    if not math.isfinite(f) or abs(f - round(f)) > 1e-9:
        raise ValueError(f"{name} 必须是整数，收到 {x!r}")
    return int(round(f))


def _close(x, y, tol):
    """相对判等：|x-y| <= tol * max(|x|,|y|,eps)。"""
    return abs(x - y) <= tol * max(abs(x), abs(y), 1e-12)


def _fmt_angle(x):
    return f"{x:.4g}°"


# ---------------------------------------------------------------- 晶系判定

_SYSTEM_CN = {
    "cubic": "立方晶系",
    "tetragonal": "四方晶系",
    "orthorhombic": "正交晶系",
    "hexagonal": "六方晶系",
    "rhombohedral": "三方（菱面体）晶系",
    "monoclinic": "单斜晶系",
    "triclinic": "三斜晶系",
}

# 仅凭晶胞参数只能确定 P 型定向；带心需要原子坐标
_BRAVAIS = {
    "cubic": ("简单立方 (cP)", ["体心立方 (cI)", "面心立方 (cF)"]),
    "tetragonal": ("简单四方 (tP)", ["体心四方 (tI)"]),
    "orthorhombic": ("简单正交 (oP)", ["底心 (oC)", "体心 (oI)", "面心 (oF)"]),
    "hexagonal": ("简单六方 (hP)", []),
    "rhombohedral": ("菱面体格子 (hR)", ["简单三方 (hP)", "六方定向的三方 (hP, a=b≠c, γ=120°)"]),
    "monoclinic": ("简单单斜 (mP)", ["底心单斜 (mC)"]),
    "triclinic": ("简单三斜 (aP)", []),
}


def _classify_cell(a, b, c, al, be, ga, tol=DEFAULT_TOL):
    """7 晶系判据。返回 (system_key, facts)。

    facts 里带边长相等关系与直角关系，供 steps 复用。
    """
    ab = _close(a, b, tol)
    ac = _close(a, c, tol)
    bc = _close(b, c, tol)
    ra = _close(al, 90.0, tol)
    rb = _close(be, 90.0, tol)
    rg = _close(ga, 90.0, tol)
    n_right = int(ra) + int(rb) + int(rg)
    facts = {
        "a_eq_b": ab, "a_eq_c": ac, "b_eq_c": bc,
        "alpha_right": ra, "beta_right": rb, "gamma_right": rg,
        "n_right": n_right,
        "gamma_120": _close(ga, 120.0, tol),
        "angles_equal": _close(al, be, tol) and _close(be, ga, tol),
    }

    if n_right == 3:
        if ab and ac:
            return "cubic", facts
        if ab or ac or bc:          # a=b≠c（含非标准定向 a=c≠b / b=c≠a）
            return "tetragonal", facts
        return "orthorhombic", facts

    if ab and ra and rb and facts["gamma_120"]:
        return "hexagonal", facts

    if n_right == 2:
        return "monoclinic", facts

    if n_right == 1:
        return "triclinic", facts

    # 三个角都不是直角
    if ab and ac and facts["angles_equal"]:
        return "rhombohedral", facts
    return "triclinic", facts


def _cell_volume(a, b, c, al, be, ga):
    """V = abc·sqrt(1 - cos²α - cos²β - cos²γ + 2cosαcosβcosγ)；非法返回 None。"""
    ca, cb, cg = math.cos(math.radians(al)), math.cos(math.radians(be)), math.cos(math.radians(ga))
    inner = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if inner <= 0.0:
        return None
    return a * b * c * math.sqrt(inner)


# ------------------------------------------------------------------ C1 晶胞

def unit_cell(a, b, c, alpha=90.0, beta=90.0, gamma=90.0, tol=DEFAULT_TOL):
    """晶胞参数 → 晶系 / Bravais 定向 / 体积。

    参数
    ----
    a, b, c : 三条棱长，pm（>0）
    alpha, beta, gamma : 三个夹角（度，0<角<180）
    tol : 边长与角度的判等相对容差，默认 0.5%

    返回
    ----
    crystal_system / crystal_system_en / bravais / possible_bravais /
    volume_pm3 / volume_A3 / lattice_params / constraints / steps
    """
    try:
        a = _as_float(a, "a")
        b = _as_float(b, "b")
        c = _as_float(c, "c")
        al = _as_float(alpha, "alpha")
        be = _as_float(beta, "beta")
        ga = _as_float(gamma, "gamma")
        tol = _as_float(tol, "tol")
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("棱长 a,b,c 必须为正数（单位 pm）")
        for nm, v in (("alpha", al), ("beta", be), ("gamma", ga)):
            if not (0.0 < v < 180.0):
                raise ValueError(f"夹角 {nm}={v}° 越界，应在 0<角<180 之间")
        if not (0.0 < tol < 0.5):
            raise ValueError(f"tol={tol} 不合理，应为 0~0.5 的小数（默认 0.005 即 0.5%）")

        V = _cell_volume(a, b, c, al, be, ga)
        if V is None:
            raise ValueError(
                "这组晶胞参数不构成合法晶胞（体积公式根号内 ≤0）：请检查三个夹角，"
                "它们不能任意取值，必须满足 1-cos²α-cos²β-cos²γ+2cosαcosβcosγ > 0"
            )

        system, f = _classify_cell(a, b, c, al, be, ga, tol)
        bravais, alternatives = _BRAVAIS[system]

        # --- steps ---
        def eqword(flag):
            return "相等" if flag else "不等"

        steps = []
        steps.append(
            f"① 读晶胞参数：a={_g(a,7)} pm，b={_g(b,7)} pm，c={_g(c,7)} pm；"
            f"α={_fmt_angle(al)}，β={_fmt_angle(be)}，γ={_fmt_angle(ga)}"
        )
        steps.append(
            f"② 比较棱长（判等容差 {_g(tol*100,4)}%）："
            f"a{eqword(f['a_eq_b'])}b，a{eqword(f['a_eq_c'])}c，b{eqword(f['b_eq_c'])}c"
        )
        steps.append(
            f"③ 比较夹角（直角判据 {_g(tol*100,4)}%，即 ±{_g(90*tol,3)}°）："
            f"α{'' if f['alpha_right'] else '不'}为90°，"
            f"β{'' if f['beta_right'] else '不'}为90°，"
            f"γ{'' if f['gamma_right'] else '不'}为90°；"
            f"共 {f['n_right']} 个直角"
        )
        steps.append(f"④ 套晶系判据：{_rule_text(system, f)} → {_SYSTEM_CN[system]}")
        steps.append(
            f"⑤ 体积：V=abc·sqrt(1-cos²α-cos²β-cos²γ+2cosαcosβcosγ) = "
            f"{_g(a,7)}×{_g(b,7)}×{_g(c,7)}×{_g(math.sqrt(1 - math.cos(math.radians(al))**2 - math.cos(math.radians(be))**2 - math.cos(math.radians(ga))**2 + 2*math.cos(math.radians(al))*math.cos(math.radians(be))*math.cos(math.radians(ga))), 7)}"
            f" = {_g(V, 7)} pm^3 = {_g(V / PM3_TO_A3, 6)} Å^3"
        )

        notes = []
        if alternatives:
            notes.append(
                "仅凭晶胞参数只能确定「晶系 + 简单(P)型定向」；"
                f"同晶系还可能取带心格子（{'、'.join(alternatives)}），"
                "要区分必须给出原子坐标或系统消光信息。"
            )
        if system == "tetragonal" and not f["a_eq_b"]:
            notes.append("四方格子标准定向为 a=b≠c；你给的是另一对棱相等，属非标准定向（等价，但抄答案时应重排为 a=b≠c）。")
        if system == "monoclinic" and not f["beta_right"]:
            notes.append("单斜标准定向为 α=γ=90°、β≠90°（唯一轴 b）；你给的是 β=90° 而另一个角非直角，属非标准定向。")
        if system == "triclinic" and f["n_right"] == 1:
            notes.append("只有 1 个 90° 角，按标准定向应取一般（三斜）格子：这 1 个直角是偶然的，不是对称性要求。")
        if system == "hexagonal":
            notes.append("三方晶系也可取六方定向（a=b≠c，γ=120°）；此处按「六方」判据输出，若题目强调菱面体取法请另给 α=β=γ≠90° 的参数。")

        return _ok(
            crystal_system=_SYSTEM_CN[system],
            crystal_system_en=system,
            bravais=bravais,
            possible_bravais=alternatives,
            lattice_params={
                "a_pm": a, "b_pm": b, "c_pm": c,
                "alpha_deg": al, "beta_deg": be, "gamma_deg": ga,
            },
            constraints={
                "a_eq_b": f["a_eq_b"], "a_eq_c": f["a_eq_c"], "b_eq_c": f["b_eq_c"],
                "alpha_90": f["alpha_right"], "beta_90": f["beta_right"], "gamma_90": f["gamma_right"],
                "n_right_angles": f["n_right"],
            },
            volume_pm3=V,
            volume_A3=V / PM3_TO_A3,
            volume_str=f"{_g(V, 7)} pm^3 = {_g(V / PM3_TO_A3, 6)} Å^3",
            tol=tol,
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


def _rule_text(system, f):
    """把命中的判据写成一句可抄的话。"""
    if system == "cubic":
        return "a=b=c 且 α=β=γ=90°"
    if system == "tetragonal":
        return "a=b≠c 且 α=β=γ=90°"
    if system == "orthorhombic":
        return "a≠b≠c 且 α=β=γ=90°"
    if system == "hexagonal":
        return "a=b≠c、α=β=90°、γ=120°"
    if system == "rhombohedral":
        return "a=b=c 且 α=β=γ≠90°（菱面体取法）"
    if system == "monoclinic":
        if f["n_right"] == 2:
            bad = [n for n, ok in (("α", f["alpha_right"]), ("β", f["beta_right"]), ("γ", f["gamma_right"])) if not ok]
            return f"a≠b≠c、三个角中恰有 2 个为 90°（非直角的是 {'、'.join(bad)}）"
        return "a≠b≠c、三个角中恰有 2 个为 90°"
    return "a≠b≠c 且三个角都不是 90°（或等边但角不等）"


# -------------------------------------------------------------- C2 Miller

_INF_TOKENS = {"inf", "infinity", "∞", "无穷", "无穷大", "-", "—", "na", "n/a"}
_AXIS_PREFIX = re.compile(r"^([abcxyzABCXYZ])\s*(?:/\s*(.*))?$")
_AXIS_TIMES = re.compile(r"^(\d+(?:\.\d+)?)\s*([abcxyzABCXYZ])$")


def _to_fraction(text, axis_name, raw_tok):
    """纯数值 / 分数文本 → Fraction（"1/3" 精确，"0.3333" 走有理化）。"""
    t = text.strip()
    if not t:
        raise ValueError(f"{axis_name} 轴截距 {raw_tok!r} 不完整")
    if "/" in t:
        p, q = t.split("/", 1)
        try:
            n = Fraction(p.strip())
            d = Fraction(q.strip())
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{axis_name} 轴截距 {raw_tok!r} 无法解析为分数")
        if d == 0:
            raise ValueError(f"{axis_name} 轴截距 {raw_tok!r} 的分母为 0")
        return n / d
    try:
        f = float(t)
    except ValueError:
        raise ValueError(
            f"{axis_name} 轴截距 {raw_tok!r} 无法解析（支持 'a/2'、'2a'、'1/2'、'0.5'、'∞'）"
        )
    if not math.isfinite(f):
        return math.inf
    r = _rationalize(f)
    if r is None:
        raise ValueError(f"{axis_name} 轴截距 {raw_tok!r} 无法化简为简单分数")
    return r


def _rationalize(x, max_den=36, tol=2e-3):
    """把十进制浮点收成「最小分母」的有理数；失败返回 None。"""
    if x == 0:
        return Fraction(0, 1)
    for q in range(1, max_den + 1):
        p = round(x * q)
        if p == 0:
            continue
        if abs(x - p / q) <= tol * max(1.0, abs(x)):
            return Fraction(int(p), q)
    return None


def _parse_intercept(tok, axis_name):
    """解析一个截距（以晶轴长度为单位）。返回 Fraction 或 math.inf。"""
    if isinstance(tok, bool):
        raise ValueError(f"{axis_name} 轴截距无法解析：收到布尔值 {tok!r}")
    if isinstance(tok, int):
        v = Fraction(tok)
        if v == 0:
            raise ValueError(
                f"{axis_name} 轴截距为 0：晶面通过原点时该指数为无穷大，不是合法 Miller 指标。"
                "请把原点平移到不落在该面上的位置再取截距。"
            )
        return v
    if isinstance(tok, float):
        if not math.isfinite(tok):
            return math.inf
        if tok == 0.0:
            raise ValueError(
                f"{axis_name} 轴截距为 0：晶面通过原点时该指数为无穷大，不是合法 Miller 指标。"
            )
        r = _rationalize(tok)
        if r is None:
            raise ValueError(
                f"{axis_name} 轴截距 {tok!r} 不是可化简为简单分数的值，无法得到整数 Miller 指标"
            )
        return r

    s = str(tok).strip()
    if not s:
        raise ValueError(f"{axis_name} 轴截距为空")
    for ch in ("−", "–", "—", "－", "﹣", "­"):
        s = s.replace(ch, "-")
    s = s.replace(" ", "")
    sign = 1
    while s and s[0] in "+-":
        if s[0] == "-":
            sign = -sign
        s = s[1:]
    if not s:
        raise ValueError(f"{axis_name} 轴截距 {tok!r} 只有符号没有数值")
    if s.lower() in _INF_TOKENS:
        return math.inf

    mt = _AXIS_TIMES.match(s)                 # "2a" / "0.5b" → 截距 = 2a
    if mt:
        v = _to_fraction(mt.group(1), axis_name, tok)
    else:
        mp = _AXIS_PREFIX.match(s)            # "a" / "a/2"
        if mp:
            rest = (mp.group(2) or "").strip()
            if not rest:
                v = Fraction(1)               # "a" → 截距 = 1a
            elif rest.lower() in _INF_TOKENS:
                return math.inf
            else:
                base = _to_fraction(rest, axis_name, tok)
                if base == 0:
                    raise ValueError(f"{axis_name} 轴截距 {tok!r} 的分母为 0")
                v = Fraction(1) / base        # "a/2" → 截距 = a/2 = 1/2 个 a
        else:
            v = _to_fraction(s, axis_name, tok)

    if v == math.inf:
        return math.inf
    v = v * sign
    if v == 0:
        raise ValueError(
            f"{axis_name} 轴截距为 0：晶面通过原点时该指数为无穷大，不是合法 Miller 指标。"
            "请把原点平移到不落在该面上的位置再取截距。"
        )
    return v


def _fmt_miller(h, k, l, overline=False):
    parts = []
    for v in (h, k, l):
        if v < 0:
            parts.append(("¯" if overline else "-") + str(abs(v)))
        else:
            parts.append(str(v))
    return "(" + "".join(parts) + ")"


def miller_indices(intercepts):
    """截距 → Miller 指标 (hkl)。

    参数
    ----
    intercepts : 长度 3 的序列，元素可为
        "a/2"（带轴标签）、"1/2"、"0.5"、0.5、"∞"/"inf"（该晶面与该轴平行）

    规则：指数 = 晶轴长度 / 截距，再乘以最小公倍数化为互质整数。
    截距为 ∞ → 指数 0（晶面与该轴平行）；截距为 0 → 平面过原点，指数为无穷大，
    不是合法指标，报错。

    返回
    ----
    h, k, l / miller / miller_unicode / miller_ascii / intercepts / reciprocals / steps
    """
    try:
        if isinstance(intercepts, str):
            s = intercepts.strip()
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
            elif re.search(r"\s", s):
                parts = s.split()
            else:
                parts = [s]
        else:
            try:
                parts = list(intercepts)
            except TypeError:
                raise ValueError(
                    f"intercepts 需要 3 个截距（字符串或数值序列），收到 {intercepts!r}"
                )
        if len(parts) != 3:
            raise ValueError(
                f"intercepts 必须给 3 个（a、b、c 轴各一个），收到 {len(parts)} 个：{intercepts!r}"
            )

        axes = ("a", "b", "c")
        vals = []
        for ax, tok in zip(axes, parts):
            vals.append(_parse_intercept(tok, ax))

        # 指数 = 1/截距（∞ → 0）
        rec = []
        for v in vals:
            rec.append(Fraction(0, 1) if v == math.inf else Fraction(1, 1) / v)

        # 乘最小公倍数化整，再除最大公约数
        den = 1
        for r in rec:
            den = math.lcm(den, r.denominator)
        ints = [r.numerator * (den // r.denominator) for r in rec]
        g = 0
        for i in ints:
            g = math.gcd(g, abs(i))
        if g == 0:
            raise ValueError("三个指数化简后全为 0，不是合法 Miller 指标")
        h, k, l = (i // g for i in ints)

        inter_str = []
        for ax, v in zip(axes, vals):
            inter_str.append("∞" if v == math.inf else f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator))

        steps = ["① 写出三个截距（以对应晶轴长度为单位）："]
        for ax, tok, v, s in zip(axes, parts, vals, inter_str):
            if v == math.inf:
                steps.append(f"   {ax} 轴：{tok!r} → 截距 ∞（晶面与该轴平行）")
            else:
                steps.append(f"   {ax} 轴：{tok!r} → 截距 {s}·{ax}")
        steps.append(
            "② 取倒数得指数："
            + "，".join(
                f"{ax} 轴 1/{('∞' if v == math.inf else s)} = {('0' if v == math.inf else f'{r.numerator}/{r.denominator}')}"
                for ax, v, r, s in zip(axes, vals, rec, inter_str)
            )
        )
        steps.append(f"③ 乘以最小公倍数 {den} 化为整数：({', '.join(str(i) for i in ints)})")
        if g != 1:
            steps.append(f"④ 三个数有公约数 {g}，同除之：({h}, {k}, {l})")
        else:
            steps.append(f"④ 三数互质，不再约简：({h}, {k}, {l})")
        steps.append(f"⑤ 写成晶面指标：{_fmt_miller(h, k, l)}")

        notes = []
        if any(v == math.inf for v in vals):
            notes.append("截距为 ∞ 表示该晶面与对应晶轴平行 → 该位指数为 0。")
        if any(i < 0 for i in (h, k, l)):
            notes.append(
                "负指数读作「负 h」等；规范写法是在该数字上方加一横线，"
                "纯文本用 -1 形式，miller_unicode 字段给的是上划线版。"
            )

        return _ok(
            h=h, k=k, l=l,
            miller=_fmt_miller(h, k, l),
            miller_ascii=_fmt_miller(h, k, l),
            miller_unicode=_fmt_miller(h, k, l, overline=True),
            family="{" + f"{h}{k}{l}" + "}",
            intercepts=inter_str,
            reciprocals=[f"{r.numerator}/{r.denominator}" for r in rec],
            common_denominator=den,
            divisor=g,
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# -------------------------------------------------------------- C3 d 间距

def _d_from_triclinic_general(h, k, l, a, b, c, al, be, ga):
    """通用式：1/d² = (1/V²)·[S11 h² + S22 k² + S33 l² + 2S12 hk + 2S23 kl + 2S13 hl]。"""
    ca, cb, cg = math.cos(math.radians(al)), math.cos(math.radians(be)), math.cos(math.radians(ga))
    sa, sb, sg = math.sin(math.radians(al)), math.sin(math.radians(be)), math.sin(math.radians(ga))
    V = _cell_volume(a, b, c, al, be, ga)
    if V is None or V == 0:
        return None
    S11 = b * b * c * c * sa * sa
    S22 = a * a * c * c * sb * sb
    S33 = a * a * b * b * sg * sg
    S12 = a * b * c * c * (ca * cb - cg)
    S23 = a * a * b * c * (cb * cg - ca)
    S13 = a * b * b * c * (cg * ca - cb)
    inv_d2 = (
        S11 * h * h + S22 * k * k + S33 * l * l
        + 2 * S12 * h * k + 2 * S23 * k * l + 2 * S13 * h * l
    ) / (V * V)
    if inv_d2 <= 0:
        return None
    return 1.0 / math.sqrt(inv_d2)


def _d_from_simplified(system, h, k, l, a, b, c, al, be, ga):
    """按晶系给教材简式。定向不标准时返回 (None, None, None)。"""
    ca, cb, cg = math.cos(math.radians(al)), math.cos(math.radians(be)), math.cos(math.radians(ga))
    if system == "cubic":
        txt = "1/d² = (h²+k²+l²)/a²"
        return (1.0 / math.sqrt((h * h + k * k + l * l) / (a * a)), txt,
                f"(h²+k²+l²)/a² = ({h}²+{k}²+{l}²)/({_g(a,7)} pm)²")
    if system == "tetragonal":
        if not _close(a, b, DEFAULT_TOL):
            # 非标准定向（a=c 或 b=c）：交换轴标签后仍是同一公式，这里直接判不适用
            return (None, None, None)
        txt = "1/d² = (h²+k²)/a² + l²/c²"
        val = (h * h + k * k) / (a * a) + (l * l) / (c * c)
        return (1.0 / math.sqrt(val), txt,
                f"(h²+k²)/a² + l²/c² = ({h}²+{k}²)/({_g(a,7)})² + {l}²/({_g(c,7)})²")
    if system == "orthorhombic":
        txt = "1/d² = h²/a² + k²/b² + l²/c²"
        val = (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
        return (1.0 / math.sqrt(val), txt,
                f"h²/a² + k²/b² + l²/c² = {h}²/({_g(a,7)})² + {k}²/({_g(b,7)})² + {l}²/({_g(c,7)})²")
    if system == "hexagonal":
        if not (_close(a, b, DEFAULT_TOL) and _close(al, 90.0, DEFAULT_TOL)
                and _close(be, 90.0, DEFAULT_TOL) and _close(ga, 120.0, DEFAULT_TOL)):
            return (None, None, None)
        txt = "1/d² = 4/3·(h²+hk+k²)/a² + l²/c²"
        val = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
        return (1.0 / math.sqrt(val), txt,
                f"4/3·(h²+hk+k²)/a² + l²/c² = 4/3·({h}²+{h}·{k}+{k}²)/({_g(a,7)})² + {l}²/({_g(c,7)})²")
    if system == "rhombohedral":
        if not (_close(a, b, DEFAULT_TOL) and _close(a, c, DEFAULT_TOL)
                and _close(al, be, DEFAULT_TOL) and _close(be, ga, DEFAULT_TOL)):
            return (None, None, None)
        num = (h * h + k * k + l * l) + 2 * (h * k + k * l + h * l) * ca
        den = a * a * (1 - 3 * ca * ca + 2 * ca ** 3)
        if den <= 0:
            return (None, None, None)
        txt = "1/d² = [(h²+k²+l²) + 2(hk+kl+hl)cosα] / [a²(1-3cos²α+2cos³α)]"
        return (1.0 / math.sqrt(num / den), txt,
                f"[{h}²+{k}²+{l}² + 2({h}·{k}+{k}·{l}+{h}·{l})·{_g(ca,6)}] / [{_g(a,7)}²·(1-3·{_g(ca,6)}²+2·{_g(ca,6)}³)]")
    if system == "monoclinic":
        if not (_close(al, 90.0, DEFAULT_TOL) and _close(ga, 90.0, DEFAULT_TOL) and not _close(be, 90.0, DEFAULT_TOL)):
            return (None, None, None)
        sb = math.sin(math.radians(be))
        val = (1.0 / (sb * sb)) * (
            (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
            - 2 * h * l * cb / (a * c)
        )
        if val <= 0:
            return (None, None, None)
        txt = "1/d² = [1/sin²β]·(h²/a² + k²/b² + l²/c² - 2hl·cosβ/(ac))"
        return (1.0 / math.sqrt(val), txt,
                f"[1/sin²{_g(be,6)}°]·[{h}²/{_g(a,7)}² + {k}²/{_g(b,7)}² + {l}²/{_g(c,7)}² - 2·{h}·{l}·{_g(cb,6)}/({_g(a,7)}·{_g(c,7)})]")
    return (None, None, None)


def d_spacing(h, k, l, a, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0, tol=DEFAULT_TOL):
    """晶面间距 d(hkl)。

    按判定出的晶系选用教材对应公式（立方/四方/正交/六方/三方/单斜），并给出代入式；
    同时用一般（三斜）通用式交叉校验 —— 两者不一致说明定向不标准，则回退到通用式。

    b、c 可省略：两个都省略时按立方晶胞取 b=c=a（教材题求 d(111) 常常只给 a）。
    只省略其中一个无法确定晶胞形状，报错；b、c 省略但给了非 90° 的夹角，与立方不相容，也报错。
    """
    try:
        h = _as_int(h, "h")
        k = _as_int(k, "k")
        l = _as_int(l, "l")
        a = _as_float(a, "a")
        if (b is None) != (c is None):
            raise ValueError(
                "b、c 必须同时给或同时省略：只给其中一个无法确定晶胞形状。"
                "（两个都省略 = 按立方晶胞取 b=c=a）"
            )
        if b is None:
            b = c = a
            assumed_bc = True
        else:
            b = _as_float(b, "b")
            c = _as_float(c, "c")
            assumed_bc = False
        al = _as_float(alpha, "alpha")
        be = _as_float(beta, "beta")
        ga = _as_float(gamma, "gamma")
        if h == 0 and k == 0 and l == 0:
            raise ValueError("(hkl) 不能是 (000)：三个指数全 0 不对应任何晶面")
        if min(a, b, c) <= 0:
            raise ValueError("棱长 a,b,c 必须为正数（单位 pm）")
        for nm, v in (("alpha", al), ("beta", be), ("gamma", ga)):
            if not (0.0 < v < 180.0):
                raise ValueError(f"夹角 {nm}={v}° 越界，应在 0<角<180 之间")
        if assumed_bc:
            for nm, v in (("alpha", al), ("beta", be), ("gamma", ga)):
                if not _close(v, 90.0, tol):
                    raise ValueError(
                        f"只给了 a={_g(a,7)} pm（b、c 省略 → 按立方取 b=c=a），"
                        f"但 {nm}={_fmt_angle(v)} 不是直角，这个组合与立方晶胞不相容。"
                        "请补齐 b、c 与确切的夹角后再算。"
                    )

        system, _f = _classify_cell(a, b, c, al, be, ga, tol)
        d_simple, formula_txt, subst_txt = _d_from_simplified(system, h, k, l, a, b, c, al, be, ga)
        d_general = _d_from_triclinic_general(h, k, l, a, b, c, al, be, ga)
        if d_general is None:
            raise ValueError("这组晶胞参数算不出正的 d 值：请检查棱长与夹角是否构成合法晶胞")

        notes = []
        if assumed_bc:
            notes.append(
                "b、c 未给出，按立方晶胞 b=c=a、α=β=γ=90° 处理。"
                "若实际不是立方晶系，这个 d 值不成立 —— 请补 b、c 与夹角。"
            )
        if d_simple is None:
            d = d_general
            formula_used = "通用式（倒易度量张量）：1/d² = (1/V²)·[S11h²+S22k²+S33l²+2S12hk+2S23kl+2S13hl]"
            subst = f"V={_g(_cell_volume(a,b,c,al,be,ga),7)} pm^3，代入 (h,k,l)=({h},{k},{l}) 得 d={_g(d_general,7)} pm"
            notes.append(
                f"这组参数的定向不满足{_SYSTEM_CN[system]}的教材标准取法，简式（如立方 (h²+k²+l²)/a²）"
                "在非标准定向下不成立，故用通用式；若按标准定向重排晶轴，可用对应简式。"
            )
        else:
            if abs(d_simple - d_general) <= 1e-9 * max(abs(d_simple), 1.0):
                d = d_simple
                formula_used = formula_txt
                subst = subst_txt
            else:
                d = d_general
                formula_used = "通用式（倒易度量张量），简式在本组定向下不成立"
                subst = f"V={_g(_cell_volume(a,b,c,al,be,ga),7)} pm^3，代入 (h,k,l)=({h},{k},{l}) 得 d={_g(d_general,7)} pm"
                notes.append(
                    f"简式 {formula_txt} 算得 {_g(d_simple,7)} pm，与通用式 {_g(d_general,7)} pm 不符 —— "
                    "原因是晶轴定向不是该晶系的标准取法（例如 a=b≠c 的四方格子被给成 a=c≠b）。采用通用式的值。"
                )

        steps = [
            f"① 判晶系：a={_g(a,7)} pm，b={_g(b,7)} pm，c={_g(c,7)} pm，"
            f"α={_fmt_angle(al)}，β={_fmt_angle(be)}，γ={_fmt_angle(ga)} → {_SYSTEM_CN[system]}"
            + ("（b、c 未给，按立方晶胞取 b=c=a）" if assumed_bc else ""),
            f"② 选公式：{formula_used}",
            f"③ 代入 (h,k,l)=({h},{k},{l})：{subst}",
            f"④ 取倒数开方：d = {_g(d,7)} pm = {_g(d / PM_PER_A, 6)} Å",
        ]
        if any(v == 0 for v in (h, k, l)):
            steps.append("说明：指数为 0 表示该晶面与对应晶轴平行，该轴方向对间距无贡献。")

        return _ok(
            h=h, k=k, l=l,
            miller=_fmt_miller(h, k, l),
            d_pm=d,
            d_A=d / PM_PER_A,
            d_nm=d / (PM_PER_A * 10.0),
            d_str=f"{_g(d, 7)} pm = {_g(d / PM_PER_A, 6)} Å",
            formula_used=formula_used,
            crystal_system=_SYSTEM_CN[system],
            crystal_system_en=system,
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# -------------------------------------------------------------- C4 堆积方式

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)

_PACKING = {
    "fcc": {
        "name_cn": "面心立方最密堆积（A1，ccp）",
        "atoms_per_cell": 4,
        "atoms_per_primitive_cell": 1,
        "coordination_number": 12,
        "fraction": math.pi / (3.0 * _SQRT2),
        "fraction_expr": "π/(3√2)",
        "r_coef": _SQRT2 / 4.0,
        "r_vs_a": "r=√2a/4",
        "r_derivation": "沿面对角线三个球相切：4r=√2a → r=√2a/4",
        "count_derivation": "8 个顶点×1/8 + 6 个面心×1/2 = 1+3 = 4 个原子/胞",
        "stacking": "ABCABC…（密置层按 (111) 方向堆）",
        "voids": {
            "octahedral": 4,
            "tetrahedral": 8,
            "octahedral_note": "八面体空隙数 = 原子数：体心 1 + 12 条棱的中点 12×1/4 = 4 个/胞",
            "tetrahedral_note": "四面体空隙数 = 2×原子数：把胞切成 8 个小立方体，每个体心一个 = 8 个/胞",
        },
        "examples": ["Cu", "Ag", "Au", "Al", "Pb", "Ni", "γ-Fe", "Ne（固态）"],
        "extra": "配位数 12 = 同层 6 + 上层 3 + 下层 3",
    },
    "hcp": {
        "name_cn": "六方最密堆积（A3）",
        "atoms_per_cell": 2,
        "atoms_per_primitive_cell": 2,
        "atoms_per_hex_prism": 6,
        "coordination_number": 12,
        "fraction": math.pi / (3.0 * _SQRT2),
        "fraction_expr": "π/(3√2)（理想 c/a=√(8/3)=1.633 时）",
        "cell_volume_coef": _SQRT2,
        "r_coef": 0.5,
        "r_vs_a": "r=a/2",
        "r_derivation": "同一密置层内相邻球相切，球心距等于晶胞棱长 a，即 a=2r → r=a/2",
        "count_derivation": "六方晶胞（菱形棱柱底面，a=b、γ=120°）含 2 个原子：A 层与 B 层各摊到 1 个",
        "stacking": "ABAB…",
        "voids": {
            "octahedral": 2,
            "tetrahedral": 4,
            "octahedral_note": "八面体空隙数 = 原子数 = 2 个/晶胞（若按六方棱柱 3 胞计则为 6 个）",
            "tetrahedral_note": "四面体空隙数 = 2×原子数 = 4 个/晶胞（若按六方棱柱计则为 12 个）",
        },
        "examples": ["Mg", "Zn", "Ti", "Co", "Cd", "Be"],
        "extra": "配位数 12 = 同层 6 + 上层 3 + 下层 3（理想 c/a = √(8/3) = 1.633）",
    },
    "bcc": {
        "name_cn": "体心立方堆积（A2）",
        "atoms_per_cell": 2,
        "atoms_per_primitive_cell": 1,
        "coordination_number": 8,
        "fraction": _SQRT3 * math.pi / 8.0,
        "fraction_expr": "√3π/8",
        "r_coef": _SQRT3 / 4.0,
        "r_vs_a": "r=√3a/4",
        "r_derivation": "沿体对角线三个球相切：4r=√3a → r=√3a/4",
        "count_derivation": "8 个顶点×1/8 + 1 个体心 = 2 个原子/胞",
        "stacking": "非最密堆积（密置层为 8 配位的畸变）",
        "voids": {
            "octahedral": 6,
            "tetrahedral": 12,
            "octahedral_note": "6 个/胞（6 个面心 6×1/2 + 12 条棱中点 12×1/4），但都是畸变八面体，教材一般不作为答题要点",
            "tetrahedral_note": "12 个/胞（每个面上 4 个），同样为畸变四面体",
        },
        "examples": ["α-Fe", "Cr", "W", "Mo", "Na", "K", "Cs（低温）"],
        "extra": "配位数 8 = 最近邻体心-顶点；次近邻 6 个（距离 a）。",
    },
    "sc": {
        "name_cn": "简单立方堆积",
        "atoms_per_cell": 1,
        "atoms_per_primitive_cell": 1,
        "coordination_number": 6,
        "fraction": math.pi / 6.0,
        "fraction_expr": "π/6",
        "r_coef": 0.5,
        "r_vs_a": "r=a/2",
        "r_derivation": "沿棱方向相邻球相切：a=2r → r=a/2",
        "count_derivation": "8 个顶点×1/8 = 1 个原子/胞",
        "stacking": "层内正交排列（AAA…），空隙最大，仅金属 Po 采用",
        "voids": {
            "octahedral": None,
            "tetrahedral": None,
            "octahedral_note": "简单立方不按八面体/四面体空隙讨论；体心处是 1 个立方体空隙（边长 a-2r）",
            "tetrahedral_note": "同上，见 cubic_void",
        },
        "examples": ["α-Po（唯一单质实例）"],
        "extra": "1 个立方体空隙/胞，位于体心。",
    },
    "diamond": {
        "name_cn": "金刚石型堆积（A4，立方金刚石）",
        "atoms_per_cell": 8,
        "atoms_per_primitive_cell": 2,
        "coordination_number": 4,
        "fraction": _SQRT3 * math.pi / 16.0,
        "fraction_expr": "√3π/16",
        "r_coef": _SQRT3 / 8.0,
        "r_vs_a": "r=√3a/8",
        "r_derivation": "沿体对角线 1/4 处的 C-C 键相切：2r=√3a/4 → r=√3a/8",
        "count_derivation": "fcc 的 4 个原子 + 填在 8 个四面体空隙中的 4 个 = 8 个原子/胞",
        "stacking": "每个原子以 sp3 与 4 个邻原子成键（键角 109.47°），非密堆积",
        "voids": {
            "octahedral": 4,
            "tetrahedral": 8,
            "octahedral_note": "以 fcc 骨架计 4 个八面体空隙，金刚石中全部空着",
            "tetrahedral_note": "以 fcc 骨架计 8 个四面体空隙，交替占据其中 4 个（另 4 个空着）→ 4+4=8 原子/胞；立方体空隙 8 个全空（构成孔道）",
        },
        "examples": ["金刚石 C", "Si", "Ge", "α-Sn", "GaAs（闪锌矿型，两套原子各占一半）"],
        "extra": "键长 d=√3a/4，键角 109°28′；空间利用率仅 34.01%，是四种堆积里最低的。",
    },
}

_PACKING_ALIAS = {
    "fcc": "fcc", "ccp": "fcc", "a1": "fcc", "面心立方": "fcc", "立方最密堆积": "fcc",
    "hcp": "hcp", "a3": "hcp", "六方最密堆积": "hcp", "六方": "hcp",
    "bcc": "bcc", "a2": "bcc", "体心立方": "bcc",
    "sc": "sc", "simple cubic": "sc", "简单立方": "sc", "a_h": "sc",
    "diamond": "diamond", "金刚石": "diamond", "a4": "diamond", "立方金刚石": "diamond",
}


def packing(kind):
    """五种堆积方式的确定性数据：原子数/胞、配位数、空间利用率、原子半径与 a 的关系、空隙数。

    空间利用率按「球沿最密方向相切」的几何关系精确计算（不是查表抄来的近似值），
    并与教材给的百分数对照。
    """
    try:
        if not isinstance(kind, str):
            raise ValueError(f"kind 需要字符串（fcc/hcp/bcc/sc/diamond），收到 {kind!r}")
        key = kind.strip().lower()
        key = _PACKING_ALIAS.get(key, key)
        if key not in _PACKING:
            raise ValueError(
                f"kind={kind!r} 不在支持范围。可选：fcc（面心立方）、hcp（六方最密）、"
                "bcc（体心立方）、sc（简单立方）、diamond（金刚石型）"
            )
        p = _PACKING[key]
        frac = p["fraction"]

        # 用 r=coef·a 与原子数独立复算空间利用率，做自校验
        n = p["atoms_per_cell"]
        vol_coef = p.get("cell_volume_coef", 1.0)
        frac_recalc = n * (4.0 / 3.0) * math.pi * p["r_coef"] ** 3 / vol_coef
        assert abs(frac_recalc - frac) < 1e-12, (key, frac_recalc, frac)

        r_expr = p["r_vs_a"].split("=", 1)[1]
        vol_txt = "a³" if vol_coef == 1.0 else f"{_g(vol_coef, 5)}·a³"
        steps = [
            f"① {p['name_cn']}：{p['count_derivation']}",
            f"② 原子半径与晶胞参数的关系：{p['r_derivation']}，即 {p['r_vs_a']}",
            f"③ 空间利用率 = 胞内原子体积 / 晶胞体积 = {n}×(4/3)πr³ / ({vol_txt})",
            f"   = {n}×(4/3)π({r_expr})³ / ({vol_txt}) = {p['fraction_expr']} = {_g(frac*100, 4)}%",
            f"④ 配位数 CN = {p['coordination_number']}（{p['extra']}）",
        ]
        if p["voids"]["octahedral"] is not None:
            steps.append(
                f"⑤ 空隙：八面体 {p['voids']['octahedral']} 个/胞，四面体 {p['voids']['tetrahedral']} 个/胞"
                f"（{p['voids']['octahedral_note']}）"
            )
        else:
            steps.append(f"⑤ 空隙：{p['voids']['octahedral_note']}")

        notes = []
        if key == "hcp":
            notes.append(
                "原子数有两种口径，答题时必须说清用哪个："
                "六方晶胞（菱形棱柱底面，a=b、γ=120°）含 2 个原子（原胞即晶胞，Z=2）；"
                "若取由 3 个晶胞拼成的完整六方棱柱，则为 6 个原子。"
                "只有按「2 个原子/胞 + 晶胞体积 √2·a³（理想 c/a）」计算，空间利用率才等于 74.05%。"
            )
            notes.append(
                "74.05% 有两个前提：Z=2，且 c/a 取理想值 √(8/3)=1.633（这时层内与层间同时相切）。"
                "真实金属若 c/a 偏离这个值，按硬球模型的空间利用率 = (2π/(3√3))·(a/c)，"
                "c/a 越大利用率越低（如 Zn 的 c/a≈1.856 → 约 65%）；"
                "而 c/a < 1.633 意味着层间球已重叠，硬球模型本身失效，不能再套 74.05%。"
            )
        if key == "diamond":
            notes.append(
                "金刚石型的 CN=4、空间利用率 34.01% 都是按「碳原子半径 = 键长的一半」算的；"
                "若题目用范德华半径则会得到别的数，须看清给的是哪个半径。"
            )

        return _ok(
            kind=key,
            name=p["name_cn"],
            atoms_per_cell=n,
            atoms_per_primitive_cell=p["atoms_per_primitive_cell"],
            atoms_per_hex_prism=p.get("atoms_per_hex_prism"),
            coordination_number=p["coordination_number"],
            packing_fraction=f"{_g(frac*100, 4)}%",
            packing_fraction_value=frac,
            r_vs_a=p["r_vs_a"],
            voids=p["voids"],
            stacking=p["stacking"],
            examples=p["examples"],
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# -------------------------------------------------------------- C5 半径比

# (下限含, 上限不含, CN, 多面体, 备注)；>1.0 单列
_RADIUS_BANDS = [
    (0.000, 0.155, 2, "哑铃形（直线形）", "两个大离子夹一个小离子，仅靠静电吸引，极不稳定，实例罕见"),
    (0.155, 0.225, 3, "三角形", "平面三角形配位，如 C(NO2)3 类少见结构"),
    (0.225, 0.414, 4, "四面体", "正四面体配位，ZnS（闪锌矿）型"),
    (0.414, 0.732, 6, "八面体", "正八面体配位，NaCl 型"),
    (0.732, 1.000, 8, "立方体", "立方体配位，CsCl 型"),
]


def radius_ratio(r_plus_pm, r_minus_pm):
    """半径比 r₊/r₋ → 预测配位数与配位多面体（半径比规则）。

    边界：这是几何判据给出的**预测**，不是实测配位数。比值贴近临界值时本函数
    额外给 critical_band 提示；落在 4 配位段时附「核间距被当成 r₋」的排查话术
    （KBr 的 r₀=328 pm 是 r₊+r₋，不是 r₋）。
    """
    try:
        rp = _as_float(r_plus_pm, "r_plus_pm")
        rm = _as_float(r_minus_pm, "r_minus_pm")
        if rp <= 0 or rm <= 0:
            raise ValueError("两个离子半径都必须为正数（单位 pm）")
        if rp >= 3.0 * rm:
            raise ValueError(
                f"r₊/r₋ = {rp}/{rm} = {_g(rp/rm,4)} 过大，已超出半径比规则的适用范围"
                "（该规则假设阳离子可被阴离子围住）。请核对两个半径是否给反了，或单位是否一致。"
            )
        ratio = rp / rm

        band = None
        for lo, hi, cn, poly, note in _RADIUS_BANDS:
            if lo <= ratio < hi:
                band = (lo, hi, cn, poly, note)
                break
        if band is None:            # ratio >= 1.0
            lo, hi, cn, poly, note = (1.000, math.inf, 12,
                                      "立方体（8）或最密堆积（12）",
                                      "r₊>r₋ 时配位数由阳离子自身的堆积方式决定，可为 8 或 12")
            alternative = 8
        else:
            lo, hi, cn, poly, note = band
            alternative = None

        # 最近临界值距离（判断是否落在模糊带）
        bounds = [0.155, 0.225, 0.414, 0.732, 1.000]
        nearest = min(bounds, key=lambda b: abs(ratio - b))
        dist = abs(ratio - nearest)
        critical_band = None
        if dist <= 0.02:
            critical_band = (
                f"比值 {_g(ratio,5)} 距临界值 {nearest} 仅 {_g(dist,3)}，"
                "处于配位数判据的模糊带：实际结构可能取相邻的另一种配位数，应结合离子极化、"
                "配位场稳定化能与实验数据判断，不能只看半径比。"
            )

        steps = [
            f"① 半径比：r₊/r₋ = {_g(rp,7)} pm / {_g(rm,7)} pm = {_g(ratio,6)}",
            f"② 对照临界值表：<0.155 → 2；0.155–0.225 → 3；0.225–0.414 → 4；"
            f"0.414–0.732 → 6；0.732–1.0 → 8；>1.0 → 8 或 12",
            f"③ {_g(ratio,6)} 落在 {_g(lo,4)} ≤ r₊/r₋ < {'∞' if hi == math.inf else _g(hi,4)} 区间"
            f" → 配位数 {cn}，配位多面体：{poly}",
            f"④ 判据依据：{note}",
        ]

        notes = [note]
        if critical_band:
            notes.append(critical_band)
        if cn == 4:
            notes.append(
                "如果题目给的实测配位数是 6（如 KBr 的 NaCl 型结构），最常见的错因是把**核间距**当成了"
                "阴离子半径：KBr 的 r₀ = 328 pm 是 r₊+r₋ 之和，r₋(Br⁻) = 195 pm，"
                "r₊/r₋ = 133/195 = 0.682 → 落在八面体区间 CN=6。请检查传进来的 r_minus 是不是 r₀。"
            )

        return _ok(
            r_plus_pm=rp,
            r_minus_pm=rm,
            ratio=ratio,
            ratio_str=f"{_g(rp,6)}/{_g(rm,6)} = {_g(ratio,5)}",
            coordination_number=cn,
            coordination_number_alternative=alternative,
            polyhedron=poly,
            range=f"{_g(lo,4)} ≤ r₊/r₋ < {'∞' if hi == math.inf else _g(hi,4)}",
            range_low=lo,
            range_high=(None if hi == math.inf else hi),
            nearest_critical=nearest,
            critical_band=critical_band,
            boundary_distances=[
                {"critical": b, "distance": abs(ratio - b), "ratio_above_it": ratio >= b}
                for b in bounds
            ],
            boundaries=[
                {"range": "<0.155", "cn": 2, "polyhedron": "哑铃形（直线形）"},
                {"range": "0.155–0.225", "cn": 3, "polyhedron": "三角形"},
                {"range": "0.225–0.414", "cn": 4, "polyhedron": "四面体"},
                {"range": "0.414–0.732", "cn": 6, "polyhedron": "八面体"},
                {"range": "0.732–1.0", "cn": 8, "polyhedron": "立方体"},
                {"range": ">1.0", "cn": "8 或 12", "polyhedron": "立方体 / 最密堆积"},
            ],
            steps=steps,
            notes=notes,
        )
    except ValueError as e:
        return _err(str(e))


# ------------------------------------------------------- C6 晶胞内容物

# 标准原子量（IUPAC 常用值，g/mol）。表内没有的元素一律报错，不猜。
_ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93,
    "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22,
    "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Th": 232.04, "U": 238.03,
}


def _read_number(s, i):
    j = i
    while j < len(s) and (s[j].isdigit() or s[j] == "."):
        j += 1
    if j == i:
        return 1.0, i
    try:
        return float(s[i:j]), j
    except ValueError:
        raise ValueError(f"化学式数字部分 {s[i:j]!r} 无法解析")


def _parse_group(s):
    """解析无点号的化学式片段，支持 ()[]{} 嵌套与下标。"""
    stack = [{}]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            stack.append({})
            i += 1
        elif ch in ")]}":
            if len(stack) == 1:
                raise ValueError(f"化学式 {s!r} 括号不配对：位置 {i} 出现多余的 {ch!r}")
            grp = stack.pop()
            i += 1
            num, i = _read_number(s, i)
            for el, n in grp.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + n * num
        elif ch.isupper():
            j = i + 1
            while j < len(s) and s[j].islower():
                j += 1
            el = s[i:j]
            if el not in _ATOMIC_WEIGHTS:
                raise ValueError(
                    f"化学式里的元素符号 {el!r} 不在原子量表内（位置 {i}）。"
                    "请检查大小写（如 Na 不要写成 NA），或换用本模块支持的常见元素。"
                )
            num, i2 = _read_number(s, j)
            stack[-1][el] = stack[-1].get(el, 0.0) + num
            i = i2
        elif ch in " \t":
            i += 1
        else:
            raise ValueError(f"化学式 {s!r} 第 {i+1} 个字符 {ch!r} 无法解析")
    if len(stack) != 1:
        raise ValueError(f"化学式 {s!r} 括号不配对")
    return stack[0]


def _parse_formula(formula):
    """化学式 → (元素计数 dict, 总原子数)。支持 'CuSO4·5H2O' 这类水合物点号。"""
    s = str(formula).strip()
    if not s:
        raise ValueError("化学式为空")
    for dot in ("·", "．", "•", "*", "⋅"):
        s = s.replace(dot, ".")
    total = {}
    for seg in s.split("."):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)(.*)$", seg)
        if m:
            mult = float(m.group(1))
            seg = m.group(2).strip()
        else:
            mult = 1.0
        if not seg:
            raise ValueError(f"化学式片段只有系数没有内容：{seg!r}")
        counts = _parse_group(seg)
        for el, n in counts.items():
            total[el] = total.get(el, 0.0) + n * mult
    if not total:
        raise ValueError(f"化学式 {formula!r} 解析不出任何元素")
    n_atoms = sum(total.values())
    return total, n_atoms


def cell_content(formula, a_pm, density_gcm3=None, z=None,
                 r_plus=None, r_minus=None, volume_pm3=None,
                 b_pm=None, c_pm=None):
    """由式量 + 晶胞参数（+ 密度 或 Z）反算/正算晶胞内容物数量、密度、离子空间利用率。

    ρ = Z·M/(N_A·V)。给定 z 正算密度；给定 density 反算 Z（并给四舍五入后的整数与残差）。

    边界：体积默认按立方晶胞 V=a³；给 b_pm / c_pm 则按直角晶胞 a·b·c；
    其它晶系（六方、单斜等）请直接传 volume_pm3。离子空间利用率只在 AB 型
    （每式量 1 个阳离子 + 1 个阴离子，如 NaCl、CsCl）成立，其他型式返回 None 并说明。
    """
    try:
        counts, n_atoms = _parse_formula(formula)
        M = sum(_ATOMIC_WEIGHTS[el] * n for el, n in counts.items())

        if volume_pm3 is not None:
            V_pm3 = _as_float(volume_pm3, "volume_pm3")
            if V_pm3 <= 0:
                raise ValueError("volume_pm3 必须为正数")
            v_source = "由 volume_pm3 直接给出"
        else:
            a = _as_float(a_pm, "a_pm")
            if a <= 0:
                raise ValueError("a_pm 必须为正数（单位 pm）")
            if b_pm is not None or c_pm is not None:
                bb = a if b_pm is None else _as_float(b_pm, "b_pm")
                cc = a if c_pm is None else _as_float(c_pm, "c_pm")
                if bb <= 0 or cc <= 0:
                    raise ValueError("b_pm、c_pm 必须为正数（单位 pm）")
                V_pm3 = a * bb * cc
                v_source = f"直角晶胞 V=a·b·c={_g(a,7)}×{_g(bb,7)}×{_g(cc,7)} pm³"
            else:
                V_pm3 = a ** 3
                v_source = f"立方晶胞 V=a³=({_g(a,7)} pm)³"
        V_cm3 = V_pm3 * PM3_TO_CM3

        z_in = None if z is None else _as_int(z, "z")
        rho_in = None if density_gcm3 is None else _as_float(density_gcm3, "density_gcm3")
        if z_in is not None and z_in <= 0:
            raise ValueError("z（晶胞内式量数）必须为正整数")
        if rho_in is not None and rho_in <= 0:
            raise ValueError("density 必须为正数（g/cm^3）")
        if z_in is None and rho_in is None:
            raise ValueError(
                "至少要给 z 或 density 之一：给 z 可以正算密度 ρ=Z·M/(N_A·V)，"
                "给 density 可以反算每胞式量数 Z"
            )

        steps = [
            f"① 式量：M({formula}) = " + " + ".join(
                f"{counts[el]:g}×{_ATOMIC_WEIGHTS[el]:g}({el})" for el in counts
            ) + f" = {_g(M, 6)} g/mol",
            f"② 晶胞体积：{v_source} = {_g(V_pm3, 7)} pm^3 = {_g(V_pm3 / PM3_TO_A3, 6)} Å^3 "
            f"= {_g(V_cm3, 6)} cm^3",
        ]

        rho = None
        z_val = None
        density_source = None
        notes = []
        if volume_pm3 is None and (b_pm is not None or c_pm is not None):
            notes.append(
                "给了 b_pm / c_pm，按直角晶胞（a⊥b⊥c）算体积 a·b·c；"
                "若存在非 90° 的夹角，请直接传 volume_pm3（本函数没有角度参数）。"
            )

        if z_in is not None:
            rho = z_in * M / (NA * V_cm3)
            z_val = z_in
            density_source = "由 Z 正算"
            steps.append(
                f"③ 密度：ρ = Z·M/(N_A·V) = {z_in}×{_g(M,6)} g/mol ÷ "
                f"({_g(NA,7)} mol⁻¹ × {_g(V_cm3,6)} cm³) = {_g(rho, 6)} g/cm³"
            )

        if rho_in is not None:
            z_raw = rho_in * NA * V_cm3 / M
            z_round = int(round(z_raw))
            if z_val is None:
                z_val = z_round
                density_source = "由密度反算 Z"
                steps.append(
                    f"③ 式量数：Z = ρ·N_A·V/M = {_g(rho_in,6)}×{_g(NA,7)}×{_g(V_cm3,6)} ÷ {_g(M,6)}"
                    f" = {_g(z_raw, 6)} → 取整 Z = {z_round}"
                )
                rho = rho_in
            else:
                rho_check = z_val * M / (NA * V_cm3)
                dev = abs(rho_check - rho_in) / rho_in
                steps.append(
                    f"④ 交叉校验：用 Z={z_val} 正算得 ρ={_g(rho_check,6)} g/cm³，"
                    f"与你给的 {_g(rho_in,6)} g/cm³ 相对偏差 {_g(dev*100,4)}%"
                )
                if dev > 0.05:
                    notes.append(
                        f"Z={z_val} 与密度 {_g(rho_in,6)} g/cm³ 不自洽（偏差 {_g(dev*100,4)}% > 5%）："
                        "两者至少有一个错，或该式量的晶胞里存在部分占据/非整比。请核对。"
                    )
            if abs(z_raw - round(z_raw)) > 0.1 and z_in is None:
                notes.append(
                    f"反算出的 Z={_g(z_raw,6)} 偏离整数 {abs(z_raw - round(z_raw)):.2f}，"
                    "说明密度、式量或晶胞体积中至少有一个不准（或结构含缺陷/非整比）。"
                )

        # 离子空间利用率（仅 AB 型）
        frac = None
        if r_plus is not None and r_minus is not None and z_val:
            rp = _as_float(r_plus, "r_plus")
            rm = _as_float(r_minus, "r_minus")
            if rp <= 0 or rm <= 0:
                raise ValueError("两个离子半径都必须为正数（单位 pm）")
            if len(counts) == 2 and abs(n_atoms - 2.0) < 1e-9:
                v_ions = z_val * (4.0 / 3.0) * math.pi * (rp ** 3 + rm ** 3)
                frac = v_ions / V_pm3
                steps.append(
                    f"⑤ 离子空间利用率：{z_val}×(4/3)π({_g(rp,6)}³+{_g(rm,6)}³) pm³ ÷ {_g(V_pm3,7)} pm^3"
                    f" = {_g(frac*100, 4)}%"
                )
            else:
                notes.append(
                    f"式量 {formula} 不是 AB 型（每式量 1 阳 + 1 阴），"
                    "离子空间利用率需要每种离子的半径与个数，本工具不猜，故不给该字段。"
                )

        return _ok(
            formula=str(formula).strip(),
            composition={el: counts[el] for el in counts},
            atoms_per_formula=n_atoms,
            formula_weight=M,
            molar_mass_gmol=M,
            Z=z_val,
            density=rho,
            density_gcm3=rho,
            density_source=density_source,
            packing_fraction=(None if frac is None else f"{_g(frac*100, 4)}%"),
            packing_fraction_value=frac,
            volume_pm3=V_pm3,
            volume_A3=V_pm3 / PM3_TO_A3,
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
    print("crystal.py 自测：契约第 4 节断言 + 已知答案")
    print("=" * 66)

    # --- C2 Miller：三种输入形式都必须给 (234)
    for inp in (["a/2", "b/3", "c/4"], [0.5, 0.3333, 0.25], ["1/2", "1/3", "1/4"]):
        r = miller_indices(inp)
        check(f"miller {inp!r} → (234)",
              r["ok"] and (r["h"], r["k"], r["l"]) == (2, 3, 4) and r["miller"] == "(234)",
              repr(r.get("error") or (r.get("h"), r.get("k"), r.get("l"))))

    r = miller_indices(["a", "b", "∞"])
    check("miller ['a','b','inf'] → (110)", r["ok"] and (r["h"], r["k"], r["l"]) == (1, 1, 0),
          repr(r.get("error") or (r.get("h"), r.get("k"), r.get("l"))))

    r = miller_indices([2, 1, 2])
    check("miller [2,1,2] → (121)", r["ok"] and (r["h"], r["k"], r["l"]) == (1, 2, 1),
          repr(r.get("error") or (r["h"], r.get("k"), r.get("l"))))

    r = miller_indices(["a", "-b", "c"])
    check("miller 负截距 → (1-11)", r["ok"] and (r["h"], r["k"], r["l"]) == (1, -1, 1),
          repr(r.get("error") or (r.get("h"), r.get("k"), r.get("l"))))

    r = miller_indices(["a", "b", "c"])
    check("miller 无 steps 即不合格", r["ok"] and len(r["steps"]) >= 4)

    # --- C4 堆积
    for kind, n, cn, pct in (("fcc", 4, 12, 74.05), ("hcp", 2, 12, 74.05),
                             ("bcc", 2, 8, 68.02), ("sc", 1, 6, 52.36),
                             ("diamond", 8, 4, 34.01)):
        r = packing(kind)
        got = float(r["packing_fraction"].rstrip("%")) if r["ok"] else -1
        check(f"packing {kind}: {n} 原子/胞, CN={cn}, {pct}%",
              r["ok"] and r["atoms_per_cell"] == n and r["coordination_number"] == cn
              and abs(got - pct) < 0.02,
              repr(r.get("error") or (r.get("atoms_per_cell"), r.get("coordination_number"),
                                      r.get("packing_fraction"))))

    r = packing("hcp")
    check("packing hcp 两口径：晶胞 2 个原子 / 六方棱柱 6 个原子",
          r["ok"] and r["atoms_per_cell"] == 2 and r["atoms_per_hex_prism"] == 6)

    r = packing("fcc")
    check("packing fcc r=√2a/4 且八面体空隙=原子数、四面体=2×原子数",
          r["ok"] and r["r_vs_a"] == "r=√2a/4"
          and r["voids"]["octahedral"] == 4 and r["voids"]["tetrahedral"] == 8)

    # --- C5 半径比
    r = radius_ratio(133, 328)
    check("radius_ratio 133/328 = 0.4055（数学）",
          r["ok"] and abs(r["ratio"] - 133.0 / 328.0) < 1e-12, repr(r.get("error")))

    r = radius_ratio(133, 195)
    check("radius_ratio 133/195 = 0.682 → 八面体 CN=6（KBr 真值）",
          r["ok"] and abs(r["ratio"] - 0.68205) < 1e-4
          and r["coordination_number"] == 6 and r["polyhedron"] == "八面体",
          repr(r.get("error") or (r.get("ratio"), r.get("coordination_number"))))

    for rp, rm, cn in ((50, 400, 2), (60, 300, 3), (60, 200, 4), (150, 250, 6), (170, 200, 8)):
        r = radius_ratio(rp, rm)
        check(f"radius_ratio {rp}/{rm} → CN={cn}", r["ok"] and r["coordination_number"] == cn,
              repr(r.get("error") or r.get("coordination_number")))

    # --- C1 晶系
    cases = [
        ((564, 564, 564, 90, 90, 90), "cubic"),
        ((400, 400, 600, 90, 90, 90), "tetragonal"),
        ((400, 500, 600, 90, 90, 90), "orthorhombic"),
        ((246, 246, 670, 90, 90, 120), "hexagonal"),
        ((400, 400, 400, 75, 75, 75), "rhombohedral"),
        ((400, 400, 500, 75, 75, 75), "triclinic"),   # 等角但边长不全等 → 不是菱面体
        ((400, 500, 600, 90, 100, 90), "monoclinic"),
        ((300, 400, 500, 80, 85, 95), "triclinic"),
        ((500.5, 500, 499.8, 90, 90, 90), "cubic"),      # 容差内
    ]
    for args, want in cases:
        r = unit_cell(*args)
        check(f"unit_cell {args} → {want}", r["ok"] and r["crystal_system_en"] == want,
              repr(r.get("error") or r.get("crystal_system_en")))

    r = unit_cell(564, 564, 564)
    check("unit_cell NaCl a=564 → V≈1.79389e8 pm^3 = 179.389 Å^3",
          r["ok"] and abs(r["volume_pm3"] - 564 ** 3) < 1
          and abs(r["volume_A3"] - 179.4066) < 0.02,
          repr(r.get("error") or r.get("volume_A3")))

    # --- C3 d 间距
    r = d_spacing(1, 1, 1, 564, 564, 564)
    check("d_spacing 立方(111) a=564 → 325.6 pm",
          r["ok"] and abs(r["d_pm"] - 564 / math.sqrt(3)) < 1e-6
          and abs(r["d_pm"] - 325.63) < 0.01,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(2, 2, 2, 564, 564, 564)
    check("d_spacing (222) = (111)/2", r["ok"] and abs(r["d_pm"] - 564 / math.sqrt(12)) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(0, 0, 2, 246, 246, 670, 90, 90, 120)
    check("d_spacing 六方(002) 石墨 a=246,c=670 → d=335 pm",
          r["ok"] and abs(r["d_pm"] - 335.0) < 1e-6, repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 0, 0, 300, 400, 500, 90, 90, 90)
    check("d_spacing 正交(100) → d=a=300 pm", r["ok"] and abs(r["d_pm"] - 300.0) < 1e-9,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 1, 0, 400, 400, 600, 90, 90, 90)
    check("d_spacing 四方(110) → d=a/√2=282.84 pm",
          r["ok"] and abs(r["d_pm"] - 400 / math.sqrt(2)) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 0, 0, 246, 246, 670, 90, 90, 120)
    check("d_spacing 六方(100) → d=a·√3/2=213.04 pm",
          r["ok"] and abs(r["d_pm"] - 246 * math.sqrt(3) / 2) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 1, 0, 500, 500, 500, 90, 90, 90)
    check("d_spacing 必须回述 formula_used", r["ok"] and "1/d²" in r["formula_used"])

    # 通用式与简式一致性（三斜退化成立方）
    r = d_spacing(1, 2, 3, 400, 400, 400, 90, 90, 90)
    check("d_spacing 立方(123) d=a/√14",
          r["ok"] and abs(r["d_pm"] - 400 / math.sqrt(14)) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    # --- b、c 可选（只给 a 的立方用法，教材最常见）
    r = d_spacing(1, 0, 0, 400)
    check("d_spacing(h=1,k=0,l=0, a=400) 省略 b,c → 400 pm",
          r["ok"] and abs(r["d_pm"] - 400.0) < 1e-9, repr(r.get("error") or r.get("d_pm")))
    check("d_spacing 省略 b,c → steps 里注明按立方处理",
          r["ok"] and any("b、c 未给" in s for s in r["steps"]),
          repr(r.get("steps")))

    r = d_spacing(1, 1, 1, 400)
    check("d_spacing(h=1,k=1,l=1, a=400) → 400/√3 ≈ 230.94 pm",
          r["ok"] and abs(r["d_pm"] - 400 / math.sqrt(3)) < 1e-9,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 1, 0, 400)
    check("d_spacing(h=1,k=1,l=0, a=400) → 400/√2 ≈ 282.84 pm",
          r["ok"] and abs(r["d_pm"] - 400 / math.sqrt(2)) < 1e-9,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(2, 0, 0, 246, 246, 670, 90, 90, 120)
    check("d_spacing 六方(200) a=246,c=670 → a·√3/4 ≈ 106.52 pm",
          r["ok"] and abs(r["d_pm"] - 246 * math.sqrt(3) / 4) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 0, 0, 246, 246, 670, 90, 90, 120)
    check("d_spacing 六方(100) a=246,c=670 → a·√3/2 ≈ 213.04 pm",
          r["ok"] and abs(r["d_pm"] - 246 * math.sqrt(3) / 2) < 1e-6,
          repr(r.get("error") or r.get("d_pm")))

    r = d_spacing(1, 0, 0, 400, b=400)
    check("d_spacing 只给 b 不给 c → ok=False（无法确定晶胞形状）",
          r["ok"] is False and "同时" in str(r["error"]), repr(r))

    r = d_spacing(1, 0, 0, 400, alpha=100.0)
    check("d_spacing 只给 a 却给非 90° 角 → ok=False（与立方不相容）",
          r["ok"] is False and "立方" in str(r["error"]), repr(r))

    r = d_spacing(1, 0, 0, 400, 400, 400)
    check("d_spacing 显式给全 b,c 时不出现「按立方取」的补默认说明",
          r["ok"] and not any("b、c 未给" in s for s in r["steps"]))

    # --- C6 晶胞内容物
    r = cell_content("NaCl", 564.0, density_gcm3=2.165)
    check("cell_content NaCl a=564, ρ=2.165 → Z=4",
          r["ok"] and r["Z"] == 4 and abs(r["formula_weight"] - 58.44) < 0.01,
          repr(r.get("error") or (r.get("Z"), r.get("formula_weight"))))

    r = cell_content("NaCl", 564.0, z=4)
    check("cell_content NaCl Z=4 → ρ≈2.164 g/cm³",
          r["ok"] and abs(r["density"] - 2.164) < 0.01,
          repr(r.get("error") or r.get("density")))

    r = cell_content("NaCl", 564.0, z=4, r_plus=102, r_minus=181)
    check("cell_content NaCl 空间利用率≈65.3%",
          r["ok"] and abs(r["packing_fraction_value"] - 0.6528) < 0.005,
          repr(r.get("error") or r.get("packing_fraction")))

    r = cell_content("KBr", 658.0, density_gcm3=2.75)
    check("cell_content KBr a=658, ρ=2.75 → Z=4",
          r["ok"] and r["Z"] == 4, repr(r.get("error") or r.get("Z")))

    r = cell_content("CsCl", 411.0, z=1)
    check("cell_content CsCl a=411, Z=1 → ρ≈4.0 g/cm³",
          r["ok"] and abs(r["density"] - 4.0) < 0.15, repr(r.get("error") or r.get("density")))

    r = cell_content("CuSO4·5H2O", 500.0, z=2)
    check("cell_content 水合物式量 CuSO4·5H2O = 249.68",
          r["ok"] and abs(r["formula_weight"] - 249.68) < 0.05,
          repr(r.get("error") or r.get("formula_weight")))

    r = cell_content("(NH4)2SO4", 500.0, z=4)
    check("cell_content 括号式量 (NH4)2SO4 = 132.14",
          r["ok"] and abs(r["formula_weight"] - 132.14) < 0.05,
          repr(r.get("error") or r.get("formula_weight")))

    r = cell_content("NaCl", 564.0, z=4, b_pm=564.0, c_pm=564.0)
    check("cell_content 传 b_pm/c_pm（正交口径）与立方结果一致",
          r["ok"] and abs(r["density"] - 2.164) < 0.01,
          repr(r.get("error") or r.get("density")))

    # --- 非法输入必须 ok=False 且给中文原因
    bad = [
        ("unit_cell 负棱长", unit_cell(-1, 1, 1)),
        ("unit_cell 角越界", unit_cell(1, 1, 1, 200, 90, 90)),
        ("unit_cell 非法格子", unit_cell(1, 1, 1, 30, 30, 170)),
        ("miller 截距为 0", miller_indices(["0", "b", "c"])),
        ("miller 个数不足", miller_indices(["a/2", "b/3"])),
        ("miller 无法解析", miller_indices(["a/2", "b/3", "c/x"])),
        ("d_spacing (000)", d_spacing(0, 0, 0, 500, 500, 500)),
        ("d_spacing 负棱长", d_spacing(1, 0, 0, -500, 500, 500)),
        ("packing 未知类型", packing("qwe")),
        ("radius_ratio 负半径", radius_ratio(-1, 200)),
        ("cell_content 缺 Z 与密度", cell_content("NaCl", 564)),
        ("cell_content 未知元素", cell_content("XxCl", 564, z=4)),
        ("cell_content 括号不配对", cell_content("Ca(OH2", 500, z=4)),
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
