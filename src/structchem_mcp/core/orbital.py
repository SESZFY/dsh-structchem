# -*- coding: utf-8 -*-
"""分子轨道与化学键（契约 3.3）：Hückel MO / 芳香性 / 双原子分子 MO / VSEPR。

对外四个函数：

    huckel(topology, n_electrons=None, n_atoms=None, alpha=0.0, beta=-1.0, adj=None)
    aromaticity(n_pi, n_atoms, ring=True, planar=True, conjugative=True)
    diatomic_mo(n_electrons=None, period=2, homo="O2")
    vsepr(central, ligands, lone_pairs=0, ligand_charges=None, charge=0)

每个函数返回 dict，必含 ``ok`` / ``error`` / ``steps``；``steps`` 是可直接抄进答题
纸的逐步算式（中文、含代入数字）。输入非法一律返回
``{"ok": False, "error": "<说人话：缺什么/哪里错/怎么改>"}``，不抛裸异常。

================================================================
一、Hückel 的符号约定（全模块统一；答题照这一段写）
================================================================

令      x ≡ (E - α)/β        ⟺        E = α + xβ

久期方程（H_ii = α、相邻 H_rs = β、S_rs = δ_rs）：

        (α - E)c_r + β·Σ_{s~r} c_s = 0        （s~r 表示 s 与 r 相邻）

同除 β（β ≠ 0）得

        x·c_r - Σ_{s~r} c_s = 0      即   (xI - A)c = 0，A 为分子图邻接矩阵
        久期行列式：|x·δ_ij - A_ij| = 0

所以 **x 就是邻接矩阵 A 的本征值**，E = α + xβ。

教材《结构化学基础》用 y ≡ (α - E)/β（即 y = -x），对应行列式 |y·δ_ij + 1| = 0、
E = α - yβ。两种写法的 y 与 x 互为相反数，**能级集合与 MO 系数完全相同**。
本模块内部求 A 的本征值（数值最干净），steps 里同时给出两个形式的多项式作对照，
输出字段则采用教材能级图的写法：

    苯  x = 2, 1, 1, -1, -1, -2  →  E = α+2β, α+β, α+β, α-β, α-β, α-2β
    （β < 0，故 x 越大能量越低；离域能 2β 也是负值，表示比局部化更稳定）

================================================================
二、自由价口径
================================================================

    F_r = √3 - Σ_s P_rs        （√3 = 1.7320508…，即教材的 1.732 - ΣP）

不用 Wilson 的 4.732 - ΣP 口径。苯碳原子得 F = 0.398，与教材数值一致。

================================================================
三、边界（不做什么）
================================================================

- HMO 只处理共轭 π 体系；不做 σ 骨架、不做几何优化、不做 SCF、不给反应性预测。
- 简并轨道取本征分解给出的任一组正交归一基，MO 系数的排列随基选择而变。
  P_rs / q_r / F_r 只在简并子空间被**完全占据**时才与基选择无关；部分占据时
  （环丁二烯型、自由基型）本来就只有对称性分析能定，此时会给 warning。
- 双原子分子 MO 只覆盖 period=1（σ1s/σ*1s）与 period=2（第二周期两套能级顺序）；
  第三周期及以下的重元素（d 轨道参与）不在本工具范围。
- VSEPR 判断的是理想构型与孤对占位规则，不预测真实键角的实验值（只给教材典型值）。
"""

from __future__ import annotations

import math

import numpy as np

try:  # 包内相对导入优先；直接运行文件时退化
    from ._const import spin_only_moment
except ImportError:  # pragma: no cover
    from _const import spin_only_moment  # type: ignore

__all__ = ["huckel", "aromaticity", "diatomic_mo", "vsepr"]

_SQRT3 = math.sqrt(3.0)
_TOL = 1e-8


# =====================================================================
# 通用小工具
# =====================================================================


def _err(msg: str) -> dict:
    """统一失败返回。"""
    return {"ok": False, "error": msg}


def _fmt(x: float, nd: int = 3) -> str:
    """数值 → 可抄进答题纸的短字符串（去尾零；负号用 ASCII '-'）。"""
    if x is None:
        return ""
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if abs(x) < 5e-10:
        return "0"
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    if s in ("-0", "", "-"):
        return "0"
    return s


def _snap(v: float, tol: float = 1e-8) -> float:
    """把浮点噪声吸附到最近的整数（-6.000000000002 → -6.0）。"""
    r = round(v)
    if abs(v - r) < tol:
        return float(r)
    return float(v)


def _e_label(x: float, nd: int = 3) -> str:
    """以 β 为单位的能级系数 → 教材写法：'α+1.414β' / 'α+β' / 'α' / 'α-2β'。"""
    x = float(x)
    if abs(x) < 5e-9:
        return "α"
    sign = "+" if x > 0 else "-"
    a = abs(x)
    mag = "β" if abs(a - 1.0) < 5e-9 else f"{_fmt(a, nd)}β"
    return f"α{sign}{mag}"


def _energy_sum_str(k_alpha: int, coef_beta: float) -> str:
    """总能量表达式字符串，如 '6α + 8β' / '4α + 4.472β'。"""
    s = f"{_fmt(k_alpha, 3)}α"
    if abs(coef_beta) > 5e-10:
        s += f" {'+' if coef_beta > 0 else '-'} {_fmt(abs(coef_beta), 3)}β"
    return s


def _poly_str(coeffs: list[float], var: str = "x") -> str:
    """[c1..cn] → 'x^6 - 6x^4 + 9x^2 - 4'（对应 var^n + c1 var^(n-1) + ... + cn）。"""
    n = len(coeffs)
    parts = [f"{var}^{n}" if n > 1 else var]
    for i, c in enumerate(coeffs, start=1):
        p = n - i
        if abs(c) < 5e-10:
            continue
        body: str
        if p == 0:
            body = _fmt(abs(c), 4)
        else:
            v = var if p == 1 else f"{var}^{p}"
            a = abs(c)
            body = v if abs(a - 1.0) < 5e-10 else f"{_fmt(a, 4)}{v}"
        parts.append(("+ " if c > 0 else "- ") + body)
    return " ".join(parts)


def _charpoly_coeffs(A: np.ndarray) -> list[float]:
    """Faddeev–LeVerrier 求 A 的特征多项式系数（手写，便于把多项式写进 steps）。

    返回 [c1, c2, ..., cn]，对应 p(λ) = λ^n + c1·λ^(n-1) + ... + cn。
    这是真展开，不是查表：苯会得到 x^6 - 6x^4 + 9x^2 - 4。
    """
    n = A.shape[0]
    ident = np.eye(n)
    coeffs: list[float] = []
    m_prev = np.zeros((n, n))
    for k in range(1, n + 1):
        c_prev = 1.0 if not coeffs else coeffs[-1]
        m_k = A @ m_prev + c_prev * ident
        coeffs.append(_snap(-float(np.trace(A @ m_k)) / k))
        m_prev = m_k
    return coeffs


def _degenerate_groups(values: np.ndarray, tol: float = _TOL) -> list[tuple[int, int, float]]:
    """把（已降序的）本征值按数值相等分组 → [(start, stop, value)]，stop 为开区间。"""
    groups: list[tuple[int, int, float]] = []
    n = len(values)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and abs(float(values[stop]) - float(values[start])) <= tol:
            stop += 1
        groups.append((start, stop, float(values[start])))
        start = stop
    return groups


def _join_note(a: str, b: str) -> str:
    """两条提示用「；」串起来（空串自动跳过）。"""
    return f"{a}；{b}" if a else b


def _flag(value) -> bool:
    """把 'false' / '0' / False 之类统一成 bool（MCP 层可能传字符串）。"""
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "", "非", "否")
    return bool(value)


# =====================================================================
# O1  huckel —— Hückel 分子轨道
# =====================================================================


def _as_matrix(obj):
    """邻接矩阵输入的规整与校验 → (A, err)。"""
    try:
        arr = np.array(obj, dtype=float)
    except (TypeError, ValueError):
        return None, (
            "邻接矩阵必须是数字组成的二维表，例如 "
            "[[0,1,1],[1,0,1],[1,1,0]]（相邻记 1、不相邻记 0）。"
        )
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None, f"邻接矩阵必须是方阵，当前形状为 {arr.shape}。"
    n = arr.shape[0]
    if n < 2:
        return None, "共轭体系至少要有 2 个 p 轨道原子（1 个原子无法成键）。"
    if not np.allclose(arr, arr.T, atol=1e-9):
        return None, "邻接矩阵必须关于主对角线对称（题目里 A_rs = A_sr 恒成立）。"
    if not np.allclose(np.diag(arr), 0.0, atol=1e-9):
        return None, (
            "邻接矩阵主对角线必须为 0：对角元对应 H_rr = α，已由 alpha 参数统一给出，"
            "不能在矩阵里重复填。"
        )
    if np.any(arr < -1e-12):
        return None, "邻接矩阵的元素不能为负（相邻用 1；加权时用 β_rs/β 的正倍数）。"
    return arr, None


def _build_adjacency(topology, n_atoms, adj) -> dict:
    """解析 topology / adj → 邻接矩阵 + 体系描述。"""
    warnings: list[str] = []

    # ① 显式 adj 优先；② topology 本身是嵌套列表时当作矩阵
    raw = None
    if adj is not None:
        raw, err = _as_matrix(adj)
        if raw is None:
            return _err(f"adj 参数有问题：{err}")
        kind, kind_key = "自定义邻接矩阵（分子图由 adj 给定）", "custom"
        if n_atoms is not None:
            try:
                if int(n_atoms) != raw.shape[0]:
                    return _err(
                        f"n_atoms={n_atoms} 与邻接矩阵阶数 {raw.shape[0]} 不一致；"
                        "给了 adj 就不要再给 n_atoms，或让两者相等。"
                    )
            except (TypeError, ValueError):
                return _err("n_atoms 必须是正整数。")
        return {
            "ok": True,
            "error": None,
            "adj": raw,
            "kind": kind,
            "kind_key": kind_key,
            "warnings": warnings,
        }

    if isinstance(topology, (list, tuple, np.ndarray)):
        raw, err = _as_matrix(topology)
        if raw is None:
            return _err(f"topology 传成了矩阵，但{err}")
        return {
            "ok": True,
            "error": None,
            "adj": raw,
            "kind": "自定义邻接矩阵（分子图由 topology 给定）",
            "kind_key": "custom",
            "warnings": warnings,
        }

    if not isinstance(topology, str):
        return _err(
            "topology 只能是字符串 \"linear\"/\"cyclic\"/\"ring\"，或直接给邻接矩阵 "
            "[[0,1,...],[1,0,...],...]，或用 adj= 关键字传入。"
        )

    key = topology.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    linear_keys = {"linear", "chain", "polyene", "open", "openchain", "line", "链", "直链", "线性", "共轭多烯"}
    cyclic_keys = {
        "cyclic", "ring", "circle", "cycle", "monocycle", "annulene",
        "环", "单环", "环状", "闭环",
    }

    if key not in linear_keys and key not in cyclic_keys:
        return _err(
            f"不认识的 topology={topology!r}。支持：\"linear\"（共轭多烯直链）、"
            "\"cyclic\"（单环，端点相连）、\"ring\"（同 cyclic，若同时给 adj 则以 adj 为准），"
            "或直接给邻接矩阵 adj=[[...]]。"
        )

    if n_atoms is None:
        return _err(
            f"topology={topology!r} 需要 n_atoms 指明 p 轨道原子数（例如 n_atoms=6 表示苯环）。"
        )
    try:
        n_atoms_i = int(n_atoms)
    except (TypeError, ValueError):
        return _err("n_atoms 必须是正整数。")
    if n_atoms_i != n_atoms and not isinstance(n_atoms, str):
        return _err("n_atoms 必须是整数。")
    if n_atoms_i < 2:
        return _err("共轭体系至少要有 2 个 p 轨道原子（n_atoms ≥ 2）。")

    n = n_atoms_i
    A = np.zeros((n, n))
    if key in linear_keys:
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        kind, kind_key = "共轭多烯直链（端点不相连）", "linear"
        if n == 2:
            warnings.append("n_atoms=2 的直链就是一根双键，离域能必然为 0。")
    else:
        if n < 3:
            return _err("环状共轭体系至少 3 个原子（n_atoms ≥ 3）。")
        for i in range(n):
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
        kind, kind_key = f"{n} 元单环共轭体系（端点相连）", "cyclic"

    return {"ok": True, "error": None, "adj": A, "kind": kind, "kind_key": kind_key, "warnings": warnings}


def _fill_levels(lam: np.ndarray, n_el: int, tol: float = _TOL):
    """按能量从低到高填充电子（lam 已按降序 = 能量升序）；简并组内按 Hund 先单占后配对。

    返回 (occupation, partial_degenerate)。
    """
    occ = np.zeros(len(lam), dtype=float)
    remaining = int(n_el)
    partial = False
    for start, stop, _val in _degenerate_groups(lam, tol):
        if remaining <= 0:
            break
        g = stop - start
        cap = 2 * g
        if remaining >= cap:
            occ[start:stop] = 2.0
            remaining -= cap
        elif remaining <= g:
            # 组内先单占（Hund）；该简并子空间未填满 → 占据依赖基选择
            occ[start:stop] = 0.0
            occ[start:start + remaining] = 1.0
            partial = True
            remaining = 0
        else:
            # 每个轨道先 1 个，剩下的把前 (remaining-g) 个补成对；子空间仍未填满
            occ[start:stop] = 1.0
            extra = remaining - g
            occ[start:start + extra] = 2.0
            partial = True
            remaining = 0
    return occ, partial


def huckel(
    topology="linear",
    n_electrons=None,
    n_atoms=None,
    alpha=0.0,
    beta=-1.0,
    adj=None,
) -> dict:
    """Hückel 分子轨道（HMO）：真解久期行列式，给出能级、MO 系数、键级、自由价。

    算什么
    ------
    共轭 π 体系的 Hückel 处理。久期方程 (α-E)c_r + βΣ_{s~r}c_s = 0 化为
    (xI - A)c = 0，x ≡ (E-α)/β，A 为分子图邻接矩阵；|xδ_ij - A_ij| = 0 用
    numpy 对称本征分解求解（eigh），并用 Faddeev–LeVerrier 真展开特征多项式写进 steps。
    随后按能量由低到高填充 π 电子（简并组内 Hund 先单占），算出：
    总 π 电子能、离域能、键级 P_rs、电荷密度 q_r、自由价 F_r。

    参数
    ----
    topology : "linear"（共轭多烯直链）/ "cyclic" 或 "ring"（单环，端点相连），
               也可直接传邻接矩阵 [[0,1,1],[1,0,1],[1,1,0]]。
    n_electrons : π 电子数。省略时取 n_atoms（中性共轭烃每个 p 轨道贡献 1 个电子）。
                  环丙烯正离子 = 3 原子 2 电子 → n_atoms=3, n_electrons=2。
    n_atoms : p 轨道原子数（topology 为字符串时必填）。
    alpha, beta : 库仑积分与共振积分；默认 α=0、β=-1（即以 β 为单位）。
                  **β 必须为负**（成键共振积分取负值），否则报错。
    adj : 自定义邻接矩阵（优先于 topology 字符串）。

    约定（写进答案时照抄）
    ----------------------
    x ≡ (E-α)/β，E = α + xβ，x 为邻接矩阵本征值；|xδ_ij - A_ij| = 0。
    教材用 y ≡ (α-E)/β（= -x）时行列式写作 |yδ_ij + 1| = 0、E = α - yβ，结果一致。
    自由价 F_r = √3 - Σ_s P_rs（√3 = 1.7320508，教材 1.732 口径）。
    离域能 = 总 π 能 - (n_π/2)×2β = (Σn_i x_i - 2⌊n_π/2⌋)β。

    边界
    ----
    只处理共轭 π 骨架，不含 σ 体系。简并轨道取本征分解给出的任一组正交归一基，
    MO 系数的排列随之而变；若某简并子空间被**完全占据**（如苯的 α±β 简并对），
    键级/电荷密度/自由价与基选择无关；若只是**部分占据**（如环丁二烯的简并对、
    自由基型体系），这三个量在 HMO 层面本就依赖基选择，本工具会给 warning，
    需按对称性（Hund 规则/简并微扰）另行分析，不要把数值当唯一答案。
    """
    # ---------- 1. 分子图 ----------
    built = _build_adjacency(topology, n_atoms, adj)
    if built.get("ok") is False:
        return built
    A: np.ndarray = built["adj"]
    kind: str = built["kind"]
    kind_key: str = built["kind_key"]
    warnings: list[str] = list(built["warnings"])
    n = A.shape[0]

    if np.any((A > 0) & (np.abs(A - 1.0) > 1e-9)):
        warnings.append("邻接矩阵含非 1 的权重，已按 β_rs = 权重 × β 处理（加权 HMO）。")

    # 连通性（断开就不是一个共轭体系）
    seen = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for r in frontier:
            for s in range(n):
                if A[r, s] != 0 and s not in seen:
                    seen.add(s)
                    nxt.append(s)
        frontier = nxt
    if len(seen) != n:
        warnings.append(
            f"分子图不连通（{len(seen)}/{n} 个原子在第 1 个连通块内）；"
            "结果只是各片段本征值的并集，物理上应分别处理每个共轭片段。"
        )

    # ---------- 2. 参数校验 ----------
    try:
        alpha = float(alpha)
        beta = float(beta)
    except (TypeError, ValueError):
        return _err("alpha / beta 必须是数值（默认 α=0、β=-1，HMO 只用它们的相对值）。")
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        return _err("alpha / beta 必须是有限实数。")
    if beta >= 0:
        return _err(
            f"β 必须为负值（当前 β={_fmt(beta, 6)}）。Hückel 中 β 是相邻 p 轨道的共振积分，"
            "成键时取负值；若只想给相对大小请传 β=-1（默认）。"
        )

    if n_electrons is None:
        n_el = n
        warnings.append(
            f"未给 n_electrons，按「每个 p 轨道贡献 1 个 π 电子」取 n_electrons = n_atoms = {n}。"
        )
    else:
        try:
            n_el = int(n_electrons)
        except (TypeError, ValueError):
            return _err("n_electrons 必须是整数（π 电子数）。")
        if isinstance(n_electrons, float) and abs(float(n_electrons) - n_el) > 1e-12:
            return _err(f"n_electrons 必须是整数，收到 {n_electrons}。")
        if n_el != n_electrons and not isinstance(n_electrons, str):
            return _err(f"n_electrons 必须是整数，收到 {n_electrons}。")
    if n_el < 0 or n_el > 2 * n:
        return _err(
            f"n_electrons={n_el} 超出范围：{n} 个 p 轨道最多容纳 {2 * n} 个 π 电子。"
        )

    # ---------- 3. 解久期行列式 ----------
    lam_all, vec_all = np.linalg.eigh(A)  # 升序；对称矩阵 → 实本征值、正交归一向量
    order = np.argsort(-lam_all, kind="stable")  # 降序 = 能量由低到高（β<0）
    lam = lam_all[order]
    vec = vec_all[:, order].copy()

    # 相位整理：每个 MO 绝对值最大的系数取正（教材习惯）
    for k in range(n):
        col = vec[:, k]
        j = int(np.argmax(np.abs(col)))
        if col[j] < 0:
            vec[:, k] = -col

    residual = float(np.max(np.abs(A @ vec - vec * lam))) if n else 0.0

    # ---------- 4. 电子填充 ----------
    occ, partial_degenerate = _fill_levels(lam, n_el)
    if partial_degenerate:
        warnings.append(
            "电子数使某个简并能级未被完全占据（该子空间只填了一部分，例如环丁二烯的简并对）："
            "此时 MO 基的选择会影响键级、电荷密度与自由价，HMO 本身给不出唯一答案，"
            "需按对称性（Hund 规则 / 简并微扰）另行分析；本实现取正交归一基顺序填充，"
            "这三个量仅供定性参考。"
        )

    # ---------- 5. 键级 / 电荷密度 / 自由价 ----------
    P = np.zeros((n, n))
    q = np.zeros(n)
    for k in range(n):
        if occ[k] <= 0:
            continue
        c = vec[:, k]
        P += occ[k] * np.outer(c, c)
        q += occ[k] * c * c

    free_val = []
    for r in range(n):
        s_sum = float(sum(P[r, s] for s in range(n) if s != r and A[r, s] != 0))
        free_val.append(_SQRT3 - s_sum)

    bond_pairs = [(r, s) for r in range(n) for s in range(r + 1, n) if A[r, s] != 0]

    # ---------- 6. 能量 ----------
    sum_lam = float(sum(occ[k] * lam[k] for k in range(n)))
    pi_total = float(n_el * alpha + sum_lam * beta)
    n_double = n_el // 2
    ref_beta = 2.0 * n_double
    de_beta = sum_lam - ref_beta
    de = de_beta * beta

    occupied = [k for k in range(n) if occ[k] > 0]
    homo = occupied[-1] if occupied else None
    lumo = None
    for k in range(n):
        if occ[k] < 2 and (homo is None or k > homo):
            lumo = k
            break
    gap_beta = (float(lam[homo]) - float(lam[lumo])) if (homo is not None and lumo is not None) else None

    # ---------- 7. steps ----------
    steps: list[str] = []
    steps.append(
        f"① 分子图：{n} 个 p 轨道原子，{kind}；取 H_rr = α、相邻 H_rs = β、S_rs = δ_rs，"
        f"久期方程 (α-E)c_r + β·Σ_{{s~r}}c_s = 0（r = 1…{n}）。"
    )
    steps.append(
        "② 令 x = (E-α)/β，即 E = α + xβ，同除 β 得 x·c_r - Σ_{s~r}c_s = 0，"
        "久期行列式写作 |x·δ_ij - A_ij| = 0（A 为邻接矩阵），x 即 A 的本征值。"
    )
    if n <= 10:
        coeffs = _charpoly_coeffs(A)
        poly_x = _poly_str(coeffs, "x")
        poly_y = _poly_str(coeffs, "y")
        steps.append(f"③ 展开行列式得 {poly_x} = 0。")
        steps.append(
            f"   （教材若按 y = (α-E)/β 定义，同一行列式写作 |y·δ_ij + 1| = 0，"
            f"多项式为 {poly_y} = 0，y = -x。）"
        )
    else:
        steps.append(f"③ 行列式为 {n} 阶，展开式过长，此处直接给数值本征值。")

    lam_list = "、".join(_fmt(v, 3) for v in lam)
    steps.append(
        f"④ 解得 x = {{{lam_list}}}"
        f"（对称矩阵本征分解，残差 max|(xI-A)c| = {residual:.1e}）。"
    )
    e_list = "、".join(f"E{i + 1} = {_e_label(lam[i], 3)}" for i in range(n))
    steps.append(f"⑤ 换算能级 E = α + xβ：{e_list}（β < 0，x 越大能量越低）。")

    if kind_key == "linear" and n >= 2:
        ana = sorted((2.0 * math.cos(k * math.pi / (n + 1)) for k in range(1, n + 1)), reverse=True)
        ana_s = "、".join(_fmt(v, 3) for v in ana)
        steps.append(
            f"⑥ 解析解校核：n 原子直链 x_k = 2cos(kπ/(n+1))（k = 1…n）→ {{{ana_s}}}，与④一致。"
        )
    elif kind_key == "cyclic" and n >= 3:
        ana = sorted((2.0 * math.cos(2.0 * math.pi * k / n) for k in range(n)), reverse=True)
        ana_s = "、".join(_fmt(v, 3) for v in ana)
        steps.append(
            f"⑥ 解析解校核：n 元环 x_k = 2cos(2πk/n)（k = 0…n-1）→ {{{ana_s}}}，与④一致。"
        )

    occ_desc = []
    for start, stop, val in _degenerate_groups(lam):
        g = stop - start
        ne = float(sum(occ[start:stop]))
        if ne <= 0:
            continue
        tag = f"（{g} 重简并）" if g > 1 else ""
        occ_desc.append(f"{_e_label(val, 3)}{tag}上 {_fmt(ne, 3)} 个")
    if occ_desc:
        homo_txt = f"HOMO = {_e_label(lam[homo], 3)}" if homo is not None else "HOMO 为空"
        lumo_txt = f"LUMO = {_e_label(lam[lumo], 3)}" if lumo is not None else "LUMO 不存在"
        steps.append(f"⑦ {n_el} 个 π 电子按能量由低到高填充：{'；'.join(occ_desc)}；{homo_txt}，{lumo_txt}。")
    else:
        steps.append(f"⑦ n_electrons = {n_el}，无电子填充。")

    steps.append(
        f"⑧ 总 π 电子能 E_π = Σ n_i E_i = {_energy_sum_str(n_el, sum_lam)}"
        f"（Σn_i x_i = {_fmt(sum_lam, 3)}）。"
    )
    if n_double > 0:
        steps.append(
            f"⑨ 局部化参考：{n_double} 个孤立 C=C 各占 2 个 π 电子、能量 2α+2β，"
            f"合计 {_energy_sum_str(2 * n_double, ref_beta)}；"
            f"离域能 DE = E_π - 局部化能 = {_fmt(n_el - 2 * n_double, 3)}α + "
            f"{_fmt(de_beta, 3)}β = {_fmt(de_beta, 3)}β"
            f"（β < 0，故 {_fmt(de_beta, 3)}β 为负值，表示离域更稳定）。"
        )
    else:
        steps.append("⑨ π 电子不足 2 个，不存在孤立双键参考，离域能记为 0。")
    steps.append(
        "⑩ 键级 P_rs = Σ_i n_i c_ir c_is（i 取占据轨道）；电荷密度 q_r = Σ_i n_i c_ir²；"
        "自由价 F_r = √3 - Σ_s P_rs。本工具用精确 √3 = 1.7320508 计算；"
        "教材按 1.732 与三位小数键级代入，苯会写 F = 1.732 - 2×0.667 = 0.398，"
        "与精确值 0.3987 的差异只在第三位小数，属舍入差异。"
    )
    if bond_pairs:
        sample = bond_pairs[0]
        steps.append(
            f"   例：P_{sample[0] + 1}{sample[1] + 1} = {_fmt(P[sample[0], sample[1]], 3)}，"
            f"q_1 = {_fmt(q[0], 3)}，F_1 = {_fmt(free_val[0], 3)}。"
        )

    # ---------- 8. 组装 ----------
    orbitals = []
    for k in range(n):
        orbitals.append(
            {
                "index": k + 1,
                "E_beta": round(float(lam[k]), 10),
                "E_alpha_beta": _e_label(lam[k], 3),
                "E_value": round(float(alpha + lam[k] * beta), 10),
                "coeffs": [round(float(v), 10) for v in vec[:, k]],
                "coeffs_str": [_fmt(v, 4) for v in vec[:, k]],
                "occupation": round(float(occ[k]), 10),
                "degenerate_count": next(
                    stop - start for start, stop, _v in _degenerate_groups(lam) if start <= k < stop
                ),
            }
        )

    degeneracies = [
        {
            "E_beta": round(val, 10),
            "E_alpha_beta": _e_label(val, 3),
            "count": stop - start,
            "indices": list(range(start + 1, stop + 1)),
            "occupation": round(float(sum(occ[start:stop])), 10),
        }
        for start, stop, val in _degenerate_groups(lam)
    ]

    return {
        "ok": True,
        "error": None,
        "topology": kind,
        "topology_kind": kind_key,
        "n_atoms": n,
        "n_electrons": n_el,
        "pi_electrons": n_el,
        "alpha": round(alpha, 10),
        "beta": round(beta, 10),
        "convention": {
            "x_definition": "x = (E - α)/β",
            "energy": "E = α + xβ",
            "secular_determinant": "|x·δ_ij - A_ij| = 0（x 为邻接矩阵本征值）",
            "textbook_equivalent": "y = (α - E)/β = -x，|y·δ_ij + 1| = 0，E = α - yβ",
            "free_valence": "F_r = √3 - Σ_s P_rs（√3 = 1.7320508，教材 1.732 口径）",
            "beta_sign": "β < 0，故 x 越大能量越低",
        },
        "orbitals": orbitals,
        "degeneracies": degeneracies,
        "homo": None if homo is None else homo + 1,
        "lumo": None if lumo is None else lumo + 1,
        "homo_lumo_gap_beta": None if gap_beta is None else round(gap_beta, 10),
        "pi_energy_total": round(pi_total, 10),
        "pi_energy_total_beta": round(sum_lam, 10),
        "pi_energy_total_str": _energy_sum_str(n_el, sum_lam),
        "delocalization_energy": round(de, 10),
        "delocalization_energy_beta": round(de_beta, 10),
        "delocalization_energy_str": f"{_fmt(de_beta, 3)}β",
        "bond_orders": [
            {
                "i": r + 1,
                "j": s + 1,
                "order": round(float(P[r, s]), 10),
                "order_str": _fmt(P[r, s], 3),
            }
            for r, s in bond_pairs
        ],
        "bond_order_matrix": [[round(float(v), 10) for v in row] for row in P],
        "charge_densities": [round(float(v), 10) for v in q],
        "charge_densities_str": [_fmt(v, 3) for v in q],
        "free_valence": [round(float(v), 10) for v in free_val],
        "free_valence_str": [_fmt(v, 3) for v in free_val],
        "eigen_residual": residual,
        "warnings": warnings,
        "steps": steps,
    }


# =====================================================================
# O2  aromaticity —— 4n+2 芳香性判定
# =====================================================================

_AROMATICITY_EXAMPLES = [
    {
        "name": "苯", "n_pi": 6, "n_atoms": 6, "verdict": "芳香",
        "rule": "4n+2（n=1）",
        "note": "平面正六边形闭环，6 个 p 轨道形成闭合离域，HMO 离域能 2β。",
    },
    {
        "name": "吡啶", "n_pi": 6, "n_atoms": 6, "verdict": "芳香",
        "rule": "4n+2（n=1）",
        "note": "N 以 sp² 参与骨架，孤对占 sp² 杂化轨道（在环平面内、不参与 π 体系），只向 π 体系提供 1 个 p 电子，故仍是 6π。",
    },
    {
        "name": "环丙烯正离子 C3H3⁺", "n_pi": 2, "n_atoms": 3, "verdict": "芳香",
        "rule": "4n+2（n=0）",
        "note": "最小芳香体系；空 p 轨道使 2 个 π 电子填满 α+2β 成键轨道。",
    },
    {
        "name": "环戊二烯负离子 C5H5⁻", "n_pi": 6, "n_atoms": 5, "verdict": "芳香",
        "rule": "4n+2（n=1）",
        "note": "lost proton 后多出的电子进入 p 轨道，凑成 6π。",
    },
    {
        "name": "环庚三烯正离子 C7H7⁺（䓬正离子）", "n_pi": 6, "n_atoms": 7, "verdict": "芳香",
        "rule": "4n+2（n=1）",
    },
    {
        "name": "萘", "n_pi": 10, "n_atoms": 10, "verdict": "芳香",
        "rule": "4n+2（n=2）",
        "note": "10 个 p 轨道共轭，离域能 3.68β。",
    },
    {
        "name": "环戊二烯正离子 C5H5⁺", "n_pi": 4, "n_atoms": 5, "verdict": "反芳香",
        "rule": "4n（n=1）",
        "note": "平面闭环但 4π：两个电子必进简并的非键轨道，体系不稳定。",
    },
    {
        "name": "环丁二烯 C4H4", "n_pi": 4, "n_atoms": 4, "verdict": "反芳香",
        "rule": "4n（n=1）",
        "note": "理论上是反芳香；实际分子强烈畸变成非平面矩形以规避反芳香性。",
    },
    {
        "name": "环辛四烯 C8H8（COT）", "n_pi": 8, "n_atoms": 8, "verdict": "反芳香（若平面）",
        "rule": "4n（n=2）",
        "note": "8π 满足 4n，若保持平面共轭就是反芳香；实际分子取盆状（D2d）非平面构型，π 体系断开 → 实测为非芳香。这是「平面性前提」的经典反例。",
    },
    {
        "name": "环辛四烯二负离子 C8H8²⁻", "n_pi": 10, "n_atoms": 8, "verdict": "芳香",
        "rule": "4n+2（n=2）",
        "note": "加两个电子凑成 10π，分子转为平面，成为芳香体系（D8h）。",
    },
    {
        "name": "1,3-丁二烯（开链）", "n_pi": 4, "n_atoms": 4, "verdict": "非芳香",
        "rule": "—",
        "note": "4π 但非闭环，π 电子不能首尾相接形成环电流 → 非芳香（也不叫反芳香）。",
    },
]


def aromaticity(n_pi=None, n_atoms=None, ring=True, planar=True, conjugative=True) -> dict:
    """Hückel 4n+2 芳香性判定（芳香 / 反芳香 / 非芳香三分）。

    算什么
    ------
    判据链：共轭 → 环状闭合 → 平面 → 4n+2。
      · 平面 + 闭环 + 共轭 + π 电子数 = 4n+2 → **芳香**
      · 平面 + 闭环 + 共轭 + π 电子数 = 4n   → **反芳香**
      · 非闭环 / 非平面 / 非共轭 / 奇数 π 电子 → **非芳香**
    反芳香只对平面闭环体系定义；开链的 4π（如丁二烯）不是反芳香而是非芳香——这是
    阅卷高频失分点，函数会在 steps 里点明。

    参数
    ----
    n_pi : π 电子总数（含杂原子贡献、含电荷带来的增减）。
    n_atoms : 参与共轭的 p 轨道原子数（用于校验 n_pi ≤ 2·n_atoms 并写进 steps）。
    ring : 是否形成闭环共轭。False → 非芳香。
    planar : 是否平面（p 轨道平行、能形成闭合离域）。False → 非芳香。
    conjugative : 是否存在连续共轭。False → 非芳香。

    边界
    ----
    只做计数判定，不做 HMO 能量计算（那用 huckel）；不给反应性、不给稳定性排序。
    平面性需由使用者从结构判断（本函数不猜几何）。
    """
    if n_pi is None:
        return _err("缺少 n_pi（π 电子总数）：请给出参与共轭的 π 电子数，例如苯为 6。")
    if n_atoms is None:
        return _err("缺少 n_atoms（参与共轭的 p 轨道原子数）：例如苯为 6。")
    try:
        n_pi_i = int(n_pi)
        n_atoms_i = int(n_atoms)
    except (TypeError, ValueError):
        return _err("n_pi 与 n_atoms 必须是整数。")
    if isinstance(n_pi, float) and abs(n_pi - n_pi_i) > 1e-12:
        return _err(f"n_pi 必须是整数，收到 {n_pi}。")
    if isinstance(n_atoms, float) and abs(n_atoms - n_atoms_i) > 1e-12:
        return _err(f"n_atoms 必须是整数，收到 {n_atoms}。")
    if n_pi_i < 1:
        return _err("n_pi 必须 ≥ 1：没有 π 电子就谈不上共轭芳香性。")
    if n_atoms_i < 1:
        return _err("n_atoms 必须 ≥ 1。")
    if n_pi_i > 2 * n_atoms_i:
        return _err(
            f"n_pi={n_pi_i} 超过 2×n_atoms={2 * n_atoms_i}：每个 p 轨道最多容纳 2 个 π 电子，"
            "请检查原子数或电子数。"
        )

    ring_f = _flag(ring)
    planar_f = _flag(planar)
    conj_f = _flag(conjugative)

    mod = n_pi_i % 4
    n_huckel = (n_pi_i - 2) // 4 if mod == 2 else (n_pi_i // 4 if mod == 0 else None)

    steps: list[str] = []
    steps.append(f"① 数 π 电子：n_pi = {n_pi_i}（{n_atoms_i} 个 p 轨道参与共轭）。")

    verdict = "非芳香"
    rule = "—"
    reason = ""
    aromatic = False

    if not conj_f:
        reason = "不存在连续共轭（有 sp³ 原子打断），p 轨道不能首尾相接 → 非芳香。"
        steps.append(f"② 共轭判据：不成立 —— {reason}")
    elif not ring_f:
        reason = (
            "非闭环：π 体系是开链，电子无法绕环离域（No cyclic conjugation）→ 非芳香。"
            "注意开链的 4n 体系（如 1,3-丁二烯 4π）属非芳香，不叫反芳香。"
        )
        steps.append(f"② 闭环判据：不成立 —— {reason}")
    elif not planar_f:
        reason = (
            "非平面：p 轨道不平行、重叠不足，不能形成闭合离域 → 非芳香。"
            "环辛四烯（8π，若平面则为反芳香）正是靠取盆状非平面构型把 π 体系断开，"
            "从而实测为非芳香。"
        )
        steps.append(f"② 平面判据：不成立 —— {reason}")
    else:
        steps.append("② 三项前提成立：共轭、闭环、平面（p 轨道彼此平行、首尾相接）。")
        if mod == 2:
            steps.append(
                f"③ 套 Hückel 规则：n_pi = {n_pi_i} = 4n + 2 → 4n = {n_pi_i} - 2 = "
                f"{n_pi_i - 2} → n = {n_huckel}。"
            )
        elif mod == 0:
            steps.append(
                f"③ 套 Hückel 规则：n_pi = {n_pi_i} = 4n → n = {n_pi_i // 4}（对 4 取余为 0，4n 型）。"
            )
        else:
            steps.append(
                f"③ 套 Hückel 规则：n_pi = {n_pi_i} 对 4 取余 = {mod}，既非 4n 也非 4n+2。"
            )
        if mod == 2:
            verdict, rule, aromatic = "芳香", "4n+2", True
            reason = (
                f"平面闭合共轭，π 电子数 {n_pi_i} = 4n+2（n = {n_huckel}）→ 满足 Hückel 规则，芳香性。"
                f"闭壳层：{n_pi_i} 个电子恰好填满 {n_pi_i // 2} 个成键（含简并的成键）轨道，"
                "形成稳定的闭合离域。"
            )
            steps.append(f"④ n = {n_huckel} 为非负整数 → 满足 4n+2 → **芳香**。")
        elif mod == 0:
            verdict, rule, aromatic = "反芳香", "4n", False
            reason = (
                f"平面闭合共轭，但 π 电子数 {n_pi_i} = 4n（n = {n_pi_i // 4}）→ "
                "属反芳香：电子必然进入简并的非键轨道（半充满），体系能量反常升高、不稳定。"
            )
            steps.append(
                f"④ n = {n_pi_i // 4} 为整数且 π 电子数为 4n → **反芳香**"
                "（反芳香只对平面闭环体系定义）。"
            )
        else:
            verdict, rule = "非芳香", "—"
            reason = (
                f"π 电子数 {n_pi_i} 既不是 4n 也不是 4n+2（模 4 余 {mod}），"
                "无法形成闭壳层离域 → 非芳香。"
            )
            steps.append(f"④ {n_pi_i} 对 4 取余为 {mod} → 既非 4n 也非 4n+2 → **非芳香**。")

    steps.append(
        "⑤ 判据链小结：共轭（连续 p 轨道）→ 闭环（首尾相接）→ 平面（p 轨道平行）→ 4n+2。"
        "任何一环缺失都落在「非芳香」，只有「平面闭环 + 4n」才是「反芳香」。"
    )

    return {
        "ok": True,
        "error": None,
        "rule": rule,
        "aromatic": aromatic,
        "aromatic_type": verdict,
        "n": n_huckel,
        "n_pi": n_pi_i,
        "n_atoms": n_atoms_i,
        "pi_per_atom": round(n_pi_i / n_atoms_i, 4),
        "prerequisites": {"conjugative": conj_f, "ring": ring_f, "planar": planar_f},
        "reason": reason,
        "examples": [dict(e) for e in _AROMATICITY_EXAMPLES],
        "steps": steps,
    }


# =====================================================================
# O3  diatomic_mo —— 第二周期双原子分子 MO
# =====================================================================

# 能级顺序两套（教材口径）：每项 = (教材标签, 简并轨道数, 成键/反键)
_SCHEME_PERIOD1 = [
    ("σ1s", 1, "成键"),
    ("σ*1s", 1, "反键"),
]

_SCHEME_N2TYPE = [
    ("σ2s", 1, "成键"),
    ("σ*2s", 1, "反键"),
    ("π2px = π2py", 2, "成键"),
    ("σ2pz", 1, "成键"),
    ("π*2px = π*2py", 2, "反键"),
    ("σ*2pz", 1, "反键"),
]

_SCHEME_O2TYPE = [
    ("σ2s", 1, "成键"),
    ("σ*2s", 1, "反键"),
    ("σ2pz", 1, "成键"),
    ("π2px = π2py", 2, "成键"),
    ("π*2px = π*2py", 2, "反键"),
    ("σ*2pz", 1, "反键"),
]

_SHORT = {
    "σ1s": "σ1s", "σ*1s": "σ*1s",
    "σ2s": "σ2s", "σ*2s": "σ*2s",
    "π2px = π2py": "π2p", "σ2pz": "σ2pz",
    "π*2px = π*2py": "π*2p", "σ*2pz": "σ*2pz",
}

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# 分子标签 → 价电子数（只收第二周期及 period=1 的常见体系）
_SPECIES: dict[str, int] = {
    "H2": 2, "H2+": 1, "H2-": 3, "He2": 4, "He2+": 3,
    "Li2": 2, "Li2+": 1, "Be2": 4, "Be2+": 3,
    "B2": 6, "B2+": 5, "B2-": 7,
    "C2": 8, "C2+": 7, "C2-": 9,
    "N2": 10, "N2+": 9, "N2-": 11,
    "O2": 12, "O2+": 11, "O2-": 13, "O2-2": 14,
    "F2": 14, "F2+": 13, "F2-": 15,
    "Ne2": 16, "Ne2+": 15,
    "NO": 11, "NO+": 10, "NO-": 12,
    "CO": 10, "CO+": 9,
    "CN": 9, "CN-": 10, "CN+": 8,
}

_SPECIES_LOWER = {k.lower(): k for k in _SPECIES}

# 价层只有 1s 的分子：homo 指到这些标签时自动改用第一周期能级方案
_PERIOD1_SPECIES = {"H2", "H2+", "H2-", "He2", "He2+", "He2-"}

# 价电子数 → 能唯一确定的代表物种（判定物种名用）。
# 同价电子数的离子（如价电子 9 的 C2⁻ / CN / CO⁺ / N2⁺）不进这张表 → 物种名记 null，
# 绝不用某个默认名字顶替，否则「N2 的分子轨道」会被答成「O2 型」。
_CANON_P1: dict[int, str] = {2: "H2", 4: "He2"}
_CANON_P2: dict[int, str] = {
    2: "Li2",
    4: "Be2",
    6: "B2",
    8: "C2",
    10: "N2",
    11: "NO",
    12: "O2",
    13: "O2-",
    14: "F2",
    15: "F2-",
    16: "Ne2",
}


def _canonical_species(n_val: int, period: int) -> str | None:
    """价电子数 → 能唯一确定的代表物种名；多解或未收录返回 None（不猜）。"""
    table = _CANON_P1 if period == 1 else _CANON_P2
    return table.get(n_val)


def _species_candidates(n_val: int, period: int) -> list[str]:
    """价电子数相同的全部内置物种（用于告知用户「有多种可能」）。"""
    p1 = period == 1
    return [k for k, v in _SPECIES.items() if v == n_val and (k in _PERIOD1_SPECIES) == p1]


def _norm_species(label) -> str | None:
    """'o2+' / 'O2⁺' / 'O2 +' / 'O2^+' → 'O2+'；查不到返回 None。"""
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    for bad, good in (("−", "-"), ("–", "-"), ("—", "-"), ("⁻", "-"), ("⁺", "+"), ("＋", "+"), ("－", "-")):
        s = s.replace(bad, good)
    s = s.replace("^", "").replace(" ", "").replace("(", "").replace(")", "")
    # 处理 "O2(-1)" / "O2-1" 之类
    key = s.lower()
    if key in _SPECIES_LOWER:
        return _SPECIES_LOWER[key]
    # 末尾形如 "2-" 的电荷（如 "O2-2" 已在表内，这里兜底 "N2^2-" 之类）
    if len(s) >= 3 and s[-1] in "+-" and s[-2].isdigit():
        alt = s[:-1]
        if alt.lower() in _SPECIES_LOWER:
            return _SPECIES_LOWER[alt.lower()]
    return None


def _fill_scheme(scheme, n_el: int):
    """按能级顺序填充；每个能级内简并轨道考 Hund。→ (per_orbital_occ, electrons, unpaired)"""
    per_orbital: list[list[float]] = []
    remaining = int(n_el)
    for _label, g, _kind in scheme:
        cap = 2 * g
        occ = [0.0] * g
        if remaining <= 0:
            per_orbital.append(occ)
            continue
        if remaining >= cap:
            occ = [2.0] * g
            remaining -= cap
        elif remaining <= g:
            for i in range(remaining):
                occ[i] = 1.0
            remaining = 0
        else:
            occ = [1.0] * g
            extra = remaining - g
            for i in range(extra):
                occ[i] = 2.0
            remaining = 0
        per_orbital.append(occ)
    return per_orbital


def _config_str(scheme, per_orbital) -> tuple[str, str]:
    """→ (Unicode 上标组态串, ASCII 组态串)。"""
    uni: list[str] = []
    asc: list[str] = []
    for (_label, _g, _kind), occ in zip(scheme, per_orbital):
        total = sum(occ)
        if total <= 0:
            continue
        short = _SHORT.get(_label, _label)
        uni.append(f"({_label}){str(int(total)).translate(_SUP)}")
        asc.append(f"({short})^{int(total)}")
    return "".join(uni), "".join(asc)


def diatomic_mo(n_electrons=None, period=2, homo=None, n_valence_electrons=None) -> dict:
    """第二周期同核/异核双原子分子的 MO 电子组态（含两套能级顺序）。

    算什么
    ------
    按教材能级顺序填充**价电子**，给出电子组态串、键级、未成对电子数、磁性、
    MO 能级示意数据（可直接拿去画能级图）。

    **两套能级顺序必须区分**（这是本题型的核心考点）：
      · N2 及更早（Li2, Be2, B2, C2, N2，价电子 ≤ 10）：
        σ2s < σ*2s < π2p < σ2pz < π*2p < σ*2pz —— σ2pz 高于 π2p（2s-2p 混杂显著）
      · O2 / F2 / Ne2 及 NO（价电子 ≥ 11）：
        σ2s < σ*2s < σ2pz < π2p < π*2p < σ*2pz —— σ2pz 低于 π2p（有效核电荷大）
    本实现按价电子数自动选：≤ 10 → N2 型；≥ 11 → O2 型。

    电子数口径（两个参数只能给一个，都给会报错避免歧义）
    ----------------------------------------------------
    · n_electrons —— **总电子数**（默认口径，符合学生直觉）：
      第二周期每个原子内层 1s² 共 2 个，双原子合计扣掉 4 个内层电子：
        O2 = 16 → 价电子 12；N2 = 14 → 10；NO = 15 → 11；F2 = 18 → 14；O2⁻ = 17 → 13
      第一周期（period=1）没有内层，不扣。
    · n_valence_electrons —— **价电子数**（显式口径，不再扣内层）：
        O2 传 12，N2 传 10。
    · 两个都不给 → 由 homo 标签查内置价电子表。
    返回里的 ``n_electrons`` 恒为总电子数，``n_valence_electrons`` 恒为价电子数，
    ``input_mode`` 标明这次用的是哪种口径。

    物种名（species 字段）怎么定
    ---------------------------
    **只由电子数判定，绝不回落到某个默认分子名**：
      给了电子数 → 由价电子数查表（16 → O2、14 → N2、15 → NO、18 → F2、17 → O2⁻…）；
      价电子数对应多个物种时（如 9 可为 C2⁻ / CN / CO⁺ / N2⁺）→ ``species: null``，
      并给出 ``species_candidates``，提示用 homo= 显式指定；
      没给电子数而给了 homo → 用 homo 的名字（此时它才是命名依据）。
    物种名会被 skill 直接写进答案，写错等于答错题，所以这里宁可给 null 也不猜。

    已知标准答案（自测全跑）
    ------------------------
    N2 键级 3 抗磁；O2 键级 2、2 个单电子、顺磁；F2 键级 1 抗磁；
    NO 键级 2.5 顺磁；O2⁺ 键级 2.5；O2⁻ 键级 1.5；N2⁺ 键级 2.5；B2 键级 1、2 个单电子。

    参数
    ----
    n_electrons : 总电子数（如 O2 = 16）。省略时看 homo 或 n_valence_electrons。
    period : 1（σ1s/σ*1s）或 2（第二周期两套顺序）。3 及以上不在范围内。
    homo : 分子标签（"O2"/"N2"/"NO"/"O2+"/"CN-"/…）。**默认 None**：只在没给电子数时
           用来查价电子数与命名；给了电子数时它只是对照标签，物种名仍由电子数决定。
    n_valence_electrons : 价电子数，显式口径（不再扣内层）。

    边界
    ----
    只做到第二周期；不含 d 轨道参与、不做键长/键能的定量计算（只给定性趋势）。
    """
    # ---------- 1. 分子标签（homo 只管「没给电子数时查价电子」） ----------
    species_raw = _norm_species(homo) if homo is not None else None
    species_note = ""

    # ---------- 2. period ----------
    try:
        period_i = int(period)
    except (TypeError, ValueError):
        return _err("period 必须是整数 1 或 2。")
    if period_i not in (1, 2):
        return _err(
            f"period={period} 不在范围：本工具只做第一周期（σ1s/σ*1s）与第二周期双原子分子；"
            "第三周期及以下涉及 3s/3p 与 d 轨道混杂，不在契约范围。"
        )
    if species_raw in _PERIOD1_SPECIES and period_i == 2:
        period_i = 1
        species_note = f"{species_raw} 价层只有 1s，改用第一周期方案 σ1s/σ*1s"
    # 只有在「没给电子数、homo 真被当作取值依据」时，period 与 homo 的冲突才致命；
    # 给了电子数时 homo 只是标签，不该拦下 period=1 的调用。
    explicit_electrons = n_electrons is not None or n_valence_electrons is not None
    if period_i == 1 and species_raw is not None and not explicit_electrons and species_raw not in _PERIOD1_SPECIES:
        return _err(
            f"period=1 只覆盖 H2 / He2 家族；{species_raw} 属第二周期，请用 period=2（默认）。"
        )

    # 每个原子内层电子数 × 2 个原子：第二周期 1s² → 共 4；第一周期无内层 → 0
    core = 4 if period_i == 2 else 0

    # ---------- 3. 电子数（两种口径二选一） ----------
    if n_electrons is not None and n_valence_electrons is not None:
        return _err(
            "n_electrons（总电子数）与 n_valence_electrons（价电子数）只能给一个。"
            "两个都给会产生歧义（例如 16 到底是 O2 的总电子还是别的），请只保留其中一个。"
        )

    if n_valence_electrons is not None:
        try:
            n_val = int(n_valence_electrons)
        except (TypeError, ValueError):
            return _err("n_valence_electrons 必须是整数（价电子总数）。")
        if isinstance(n_valence_electrons, float) and abs(float(n_valence_electrons) - n_val) > 1e-12:
            return _err(f"n_valence_electrons 必须是整数，收到 {n_valence_electrons}。")
        input_mode = "valence"
        n_total = n_val + core
    elif n_electrons is not None:
        try:
            n_total = int(n_electrons)
        except (TypeError, ValueError):
            return _err("n_electrons 必须是整数（双原子分子的总电子数，如 O2 = 16）。")
        if isinstance(n_electrons, float) and abs(float(n_electrons) - n_total) > 1e-12:
            return _err(f"n_electrons 必须是整数，收到 {n_electrons}。")
        input_mode = "total"
        n_val = n_total - core
        if n_val < 0:
            return _err(
                f"总电子数 {n_total} 太少：period={period_i} 的每个原子至少有 {core // 2 if core else 0} "
                f"个内层电子，双原子合计扣 {core} 个后价电子为 {n_val}（负值）。"
                + ("H2 / He2 请用 period=1。" if period_i == 2 else "")
            )
    else:
        if species_raw is None:
            return _err(
                "缺少电子数：请给 n_electrons（总电子数，如 O2 = 16、N2 = 14），"
                "或给 n_valence_electrons（价电子数，如 O2 = 12），"
                "或给 homo 指定分子标签，如 homo=\"O2\"、\"N2\"、\"NO\"、\"O2+\"。"
            )
        n_val = _SPECIES[species_raw]
        n_total = n_val + core
        input_mode = "species"

    # ---------- 3.5 物种名：由电子数判定，绝不回落到某个默认分子名 ----------
    # species 字段会被 skill 直接写进答案，写错名字等于答错题（N2 被写成「O2 型」），
    # 所以这里只认电子数这个硬事实；电子数对应多个物种时宁可给 null 也不猜。
    species_candidates: list[str] = []
    if input_mode == "species":
        species_label = species_raw
        species_note = _join_note(
            species_note, f"按 {species_raw} 取价电子数 {n_val}（总电子 {n_total}）"
        )
    else:
        species_label = _canonical_species(n_val, period_i)
        if species_label is None:
            species_candidates = _species_candidates(n_val, period_i)
            cand_txt = "、".join(species_candidates) if species_candidates else "多种离子或激发态"
            species_note = _join_note(
                species_note,
                f"价电子 {n_val} 可对应 {cand_txt} 等，无法由电子数唯一确定物种 → species 记为 null；"
                "组态与键级仍按价电子数正常给出，如需命名请用 homo= 显式指定",
            )
        else:
            species_note = _join_note(
                species_note, f"价电子 {n_val} → 物种 {species_label}（由电子数判定，非默认值）"
            )
            if species_raw is not None and species_raw != species_label:
                species_note = _join_note(
                    species_note, f"homo={species_raw} 与该电子数不符，已以电子数为准"
                )

    if period_i == 1:
        scheme = _SCHEME_PERIOD1
        max_e = sum(2 * g for _l, g, _k in scheme)
    else:
        scheme = _SCHEME_N2TYPE if n_val <= 10 else _SCHEME_O2TYPE
        max_e = sum(2 * g for _l, g, _k in scheme)  # = 16

    if n_val < 0:
        return _err("价电子数不能为负，请检查输入。")
    if n_val > max_e:
        return _err(
            f"价电子数 {n_val} 超过 {period_i} 周期价层 MO 的容量 {max_e} 个电子。"
            + (f"（总电子 {n_total} 扣掉 {core} 个内层电子后为 {n_val}）" if input_mode == "total" else "")
        )

    scheme_name = (
        "第一周期：σ1s < σ*1s"
        if period_i == 1
        else (
            "N2 型（σ2s < σ*2s < π2p < σ2pz < π*2p < σ*2pz，σ2pz 高于 π2p）"
            if n_val <= 10
            else "O2 型（σ2s < σ*2s < σ2pz < π2p < π*2p < σ*2pz，σ2pz 低于 π2p）"
        )
    )

    # ---------- 4. 填充 ----------
    per_orbital = _fill_scheme(scheme, n_val)
    config_uni, config_ascii = _config_str(scheme, per_orbital)

    n_bonding = 0.0
    n_antibonding = 0.0
    unpaired = 0
    schematic = []
    for level, ((label, g, kind), occ) in enumerate(zip(scheme, per_orbital), start=1):
        e_count = float(sum(occ))
        if kind == "成键":
            n_bonding += e_count
        else:
            n_antibonding += e_count
        up = int(sum(1 for o in occ if abs(o - 1.0) < 1e-12))
        unpaired += up
        schematic.append(
            {
                "level": level,
                "label": label,
                "short_label": _SHORT.get(label, label),
                "degeneracy": g,
                "type": kind,
                "occupancy_per_orbital": occ,
                "electrons": int(e_count),
                "unpaired": up,
                "schematic_y": level,  # 绘图用：1 = 最低；同一 level 内 g 个简并轨道同高
            }
        )

    bond_order = (n_bonding - n_antibonding) / 2.0
    magnetism = "顺磁" if unpaired > 0 else "抗磁"
    moment = spin_only_moment(unpaired) if unpaired > 0 else 0.0

    # ---------- 4. 键长趋势（只给定性结论，不给未经校验的实验数值） ----------
    trend = (
        "同周期双原子分子中，键级越大 → 核间电子云越密 → 键长越短、键能越大。"
        "因此 B2(1.0) > C2(2.0) > N2(3.0) 键长依次缩短；"
        "O2(2.0) < F2(1.0) 同理。"
        "同一分子得失电子时，键级大者键长短：O2⁺(2.5) < O2(2.0) < O2⁻(1.5)；"
        "N2⁺(2.5) < N2(3.0) 但 N2⁺ 键级已降，键长相对变长。"
    )
    if period_i == 1:
        trend = "第一周期：H2 键级 1、He2 键级 0（不存在稳定分子），键级越大键长越短。"

    # ---------- 5. steps ----------
    steps: list[str] = []
    if period_i == 2:
        steps.append(
            "① 定能级顺序（关键）：第二周期有两套。"
            "N2 及以前（价电子 ≤ 10：Li2、Be2、B2、C2、N2）σ2s < σ*2s < π2p < σ2pz < π*2p < σ*2pz，"
            "σ2pz 因 2s-2p 混杂被推高到 π2p 之上；"
            "O2、F2、Ne2（价电子 ≥ 11）σ2s < σ*2s < σ2pz < π2p < π*2p < σ*2pz，σ2pz 落到 π2p 之下。"
        )
        if input_mode == "total":
            conv = (
                f"本题按总电子数算：输入 {n_total}，两个原子各有一个内层 1s²（共 {core} 个内层电子），"
                f"价电子 n = {n_total} - {core} = {n_val}"
            )
        elif input_mode == "valence":
            conv = f"本题输入为价电子数 n = {n_val}（对应总电子 {n_total}，不再扣内层）"
        else:
            conv = f"本题价电子数 n = {n_val}（总电子 {n_total}）"
        steps.append(
            f"② {conv} → 采用 {scheme_name.split('（')[0].strip()}顺序。"
        )
    else:
        steps.append("① 第一周期双原子：只填 σ1s、σ*1s 两个 MO（H、He 没有内层电子，不扣）。")
        steps.append(f"② 价电子数 n = {n_val}（总电子 {n_total}）。")
    if species_note:
        steps.append(f"   物种判定：{species_note}。")
    steps.append(f"③ 按能量由低到高填充（每轨道 2 个，简并轨道先各占 1 个）：{config_uni or '（空）'}。")
    if unpaired > 0:
        steps.append(
            f"   简并轨道按 Hund 规则先单占，故有 {unpaired} 个未成对电子（自旋平行）。"
        )
    steps.append(
        f"④ 键级 = (成键电子数 - 反键电子数)/2 = ({_fmt(n_bonding, 1)} - "
        f"{_fmt(n_antibonding, 1)})/2 = {_fmt(bond_order, 2)}。"
    )
    steps.append(
        f"⑤ 未成对电子数 = {unpaired} → {magnetism}"
        + (f"，自旋-only 磁矩 μ = √(n(n+2)) = {_fmt(moment, 3)} BM。" if unpaired > 0 else "（电子全部成对，自旋磁矩抵消）。")
    )
    if n_val % 2 == 1 and unpaired == 0:
        steps.append("   注意：奇数价电子体系不可能全部成对，请核对电子数。")

    return {
        "ok": True,
        "error": None,
        "species": species_label,
        "species_candidates": species_candidates,
        "species_source": "homo" if input_mode == "species" else "electron_count",
        "period": period_i,
        "n_electrons": n_total,
        "n_valence_electrons": n_val,
        "core_electrons": core,
        "input_mode": input_mode,
        "level_scheme": scheme_name,
        "level_scheme_order": [s[0] for s in scheme],
        "config": config_uni,
        "config_ascii": config_ascii,
        "bond_order": round(bond_order, 10),
        "bond_order_str": _fmt(bond_order, 2),
        "bonding_electrons": int(round(n_bonding)),
        "antibonding_electrons": int(round(n_antibonding)),
        "unpaired_electrons": unpaired,
        "magnetism": magnetism,
        "magnetic_moment_BM": round(moment, 6),
        "homo": next((s["label"] for s in reversed(schematic) if s["electrons"] >= 1), None),
        "lumo": next(
            (
                s["label"]
                for s in schematic
                if any(abs(o) < 1e-12 for o in s["occupancy_per_orbital"])
            ),
            None,
        ),
        "mo_energies_schematic": schematic,
        "bond_length_trend": trend,
        "steps": steps,
    }


# =====================================================================
# O4  vsepr —— 价层电子对互斥
# =====================================================================

# 价电子数（用于 VSEPR 电子对数的交叉核对）
_VALENCE_E = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2, "Tl": 3, "Pb": 4, "Bi": 5, "Po": 6, "At": 7, "Rn": 8,
    "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6, "W": 6,
    "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8, "Co": 9, "Rh": 9, "Ir": 9,
    "Ni": 10, "Pd": 10, "Pt": 10, "Cu": 11, "Ag": 11, "Au": 11, "Zn": 12, "Cd": 12, "Hg": 12,
}

# 配体向中心提供的电子数（教材口径：卤素/H 各 1；O、S 作配体贡献 0，因为它们的孤对
# 已计入中心成对电子，实际以多重键存在）
_LIGAND_E = {
    "H": 1.0, "F": 1.0, "Cl": 1.0, "Br": 1.0, "I": 1.0, "At": 1.0,
    "O": 0.0, "S": 0.0, "Se": 0.0, "Te": 0.0,
}

# (配体数, 孤对数) → 构型数据
_VSEPR_TABLE: dict[tuple[int, int], dict] = {
    (1, 0): {
        "geometry": "直线形（双原子）", "hybrid": "s", "hybrid_display": "s",
        "bond_angles": "—", "point_group": "C∞v",
        "lone_pair_sites": "无孤对电子。",
    },
    (2, 0): {
        "geometry": "直线形", "hybrid": "sp", "hybrid_display": "sp",
        "bond_angles": "180°", "point_group": "D∞h",
        "lone_pair_sites": "无孤对电子。",
    },
    (3, 0): {
        "geometry": "平面三角形", "hybrid": "sp2", "hybrid_display": "sp²",
        "bond_angles": "120°", "point_group": "D3h",
        "lone_pair_sites": "无孤对电子。",
    },
    (2, 1): {
        "geometry": "角形（V 形）", "hybrid": "sp2", "hybrid_display": "sp²",
        "bond_angles": "≈120°（孤对排斥大于成键对，略小于 120°，SO₂ 约 119.5°）",
        "point_group": "C2v",
        "lone_pair_sites": "孤对与 2 个配体在同一平面，三个电子域互成 ≈120°。",
    },
    (4, 0): {
        "geometry": "正四面体", "hybrid": "sp3", "hybrid_display": "sp³",
        "bond_angles": "109.5°", "point_group": "Td",
        "lone_pair_sites": "无孤对电子。",
    },
    (3, 1): {
        "geometry": "三角锥形", "hybrid": "sp3", "hybrid_display": "sp³",
        "bond_angles": "≈107°（小于 109.5°，因孤对-成键对排斥更大，NH₃ 为 107.3°）",
        "point_group": "C3v",
        "lone_pair_sites": "孤对占四面体的一个顶点，三个配体构成锥底。",
    },
    (2, 2): {
        "geometry": "角形（V 形）", "hybrid": "sp3", "hybrid_display": "sp³",
        "bond_angles": "≈104.5°（两对孤对排斥最大，H₂O 为 104.5°）",
        "point_group": "C2v",
        "lone_pair_sites": "两对孤对占四面体的两个顶点，排斥最强，把键角压到 104.5°。",
    },
    (5, 0): {
        "geometry": "三角双锥", "hybrid": "sp3d", "hybrid_display": "sp³d",
        "bond_angles": "轴向–赤道 90°，赤道–赤道 120°，轴向–轴向 180°",
        "point_group": "D3h",
        "lone_pair_sites": "无孤对电子。5 个位置不等价（2 轴向 + 3 赤道）。",
    },
    (4, 1): {
        "geometry": "跷跷板形（变形四面体）", "hybrid": "sp3d", "hybrid_display": "sp³d",
        "bond_angles": "赤道 F–S–F ≈102°、轴向 ≈173°（被孤对压弯）、轴–赤道 ≈88°",
        "point_group": "C2v",
        "lone_pair_sites": (
            "孤对优先占**赤道（平伏）位**：赤道位只有 2 个 90° 近邻，轴向位有 3 个 90° 近邻，"
            "孤对放在赤道位受排斥最小。剩下 2 轴向 + 2 赤道成键 → 跷跷板。"
        ),
    },
    (3, 2): {
        "geometry": "T 形", "hybrid": "sp3d", "hybrid_display": "sp³d",
        "bond_angles": "≈87.5°（孤对把轴向键压向赤道）", "point_group": "C2v",
        "lone_pair_sites": (
            "两对孤对都占**赤道（平伏）位**（彼此 120°、各只有 2 个 90° 近邻），"
            "剩 1 赤道 + 2 轴向成键 → T 形。"
        ),
    },
    (2, 3): {
        "geometry": "直线形", "hybrid": "sp3d", "hybrid_display": "sp³d",
        "bond_angles": "180°", "point_group": "D∞h",
        "lone_pair_sites": "三对孤对占满三个赤道位（互成 120°），两个轴向配体成直线。",
    },
    (6, 0): {
        "geometry": "正八面体", "hybrid": "sp3d2", "hybrid_display": "sp³d²",
        "bond_angles": "90°（相邻）、180°（相对）", "point_group": "Oh",
        "lone_pair_sites": "无孤对电子；6 个位置全等价。",
    },
    (5, 1): {
        "geometry": "四方锥形", "hybrid": "sp3d2", "hybrid_display": "sp³d²",
        "bond_angles": "锥顶–基面 ≈85°（小于 90°），基面内 90°", "point_group": "C4v",
        "lone_pair_sites": "孤对占八面体的一个顶点，把 4 个基面配体压向远离孤对的方向。",
    },
    (4, 2): {
        "geometry": "平面四方形", "hybrid": "sp3d2", "hybrid_display": "sp³d²",
        "bond_angles": "90°", "point_group": "D4h",
        "lone_pair_sites": (
            "两对孤对取**反位（trans，180°）**分别占据两个轴向位置，彼此排斥最小；"
            "4 个配体落在赤道平面内 → 平面四方（不是四面体）。"
        ),
    },
    (7, 0): {
        "geometry": "五角双锥", "hybrid": "sp3d3", "hybrid_display": "sp³d³",
        "bond_angles": "赤道 72°，轴向–赤道 90°", "point_group": "D5h",
        "lone_pair_sites": "无孤对电子（教材较少涉及七配位）。",
    },
    (6, 1): {
        "geometry": "变形八面体（单帽八面体）", "hybrid": "sp3d3", "hybrid_display": "sp³d³",
        "bond_angles": "≈70°–90°（显著畸变）", "point_group": "C3v",
        "lone_pair_sites": "孤对占据七配位的一个位置，八面体发生畸变（XeF₆ 属此类，教材较少考）。",
    },
    (5, 2): {
        "geometry": "五角锥形", "hybrid": "sp3d3", "hybrid_display": "sp³d³",
        "bond_angles": "≈72°", "point_group": "C2v",
        "lone_pair_sites": "两对孤对占据赤道位（教材较少涉及）。",
    },
}

_VSEPR_EXAMPLES: dict[tuple[int, int], list[str]] = {
    (1, 0): ["HCl", "CO"],
    (2, 0): ["CO₂", "BeCl₂", "HgCl₂", "HCN", "N₂O", "CS₂"],
    (3, 0): ["BF₃", "SO₃", "CO₃²⁻", "NO₃⁻", "AlCl₃"],
    (2, 1): ["SO₂", "O₃", "SnCl₂", "NO₂⁻", "ClO₂⁻"],
    (4, 0): ["CH₄", "NH₄⁺", "SO₄²⁻", "SiF₄", "CCl₄", "PO₄³⁻"],
    (3, 1): ["NH₃", "PCl₃", "SO₃²⁻", "H₃O⁺", "AsCl₃"],
    (2, 2): ["H₂O", "H₂S", "SCl₂", "OF₂", "NH₂⁻"],
    (5, 0): ["PCl₅", "PF₅", "SbCl₅", "AsF₅"],
    (4, 1): ["SF₄", "SeF₄", "TeCl₄", "IF₄⁺", "SCl₄"],
    (3, 2): ["ClF₃", "BrF₃", "ICl₃"],
    (2, 3): ["XeF₂", "I₃⁻", "IF₂⁻", "BrF₂⁻"],
    (6, 0): ["SF₆", "PF₆⁻", "SiF₆²⁻", "AlF₆³⁻", "NaCl₆⁵⁻"],
    (5, 1): ["BrF₅", "IF₅", "XeOF₄", "SbF₅²⁻"],
    (4, 2): ["XeF₄", "ICl₄⁻", "BrF₄⁻"],
    (7, 0): ["IF₇", "UF₇⁻"],
    (6, 1): ["XeF₆"],
    (5, 2): ["XeF₅⁻"],
}


def _vsepr_symbol(n_bond: int, n_lone: int) -> str:
    """AXmEn 记号：AX / AX2 / AX3E / AX4E2。"""
    s = "AX" + (str(n_bond) if n_bond > 1 else "")
    if n_lone > 0:
        s += "E" + (str(n_lone) if n_lone > 1 else "")
    return s


def vsepr(central, ligands, lone_pairs=0, ligand_charges=None, charge=0) -> dict:
    """VSEPR：由中心原子 + 配体数 + 孤对数推出价层电子对构型、VSEPR 记号、杂化与点群。

    算什么
    ------
    空间构型数 = 配体数 + 孤对数 → 电子对构型 → 去掉孤对后的分子构型 →
    AXmEn 记号 → 杂化轨道 → 键角 → 点群，并给出孤对占位规则与实例。

    孤对占位规则（必考）
    --------------------
    · 2–4 对：位置等价，孤对占哪都一样，只影响键角（孤对-成键对排斥 > 成键对-成键对）。
    · 5 对（三角双锥）：**孤对优先占赤道（平伏）位**。因为赤道位只有 2 个 90° 近邻，
      轴向位有 3 个 90° 近邻，孤对放在赤道位受排斥最小。
      → AX4E 跷跷板、AX3E2 T 形、AX2E3 直线形。
    · 6 对（八面体）：6 个位置等价，两对孤对取 **反位（trans）** 使排斥最小
      → AX4E2 平面四方；一对孤对 → AX5E 四方锥。

    参数
    ----
    central : 中心原子符号，如 "S"、"Xe"、"Cl"。
    ligands : 配体符号列表，如 ["F","F","F","F"]。元素符号重复即表示多个配体。
    lone_pairs : 中心原子的孤对电子数（不含配体上的孤对）。
    ligand_charges : 与 ligands 等长的配体电荷列表（可选），如 ["-1","-1"]。
    charge : 整个离子的电荷（可选，默认 0），如 NH₄⁺ 传 1。

    边界
    ----
    给的是理想构型与教材典型键角，不预测真实分子的畸变数值；
    元素的价电子核对只在中心/配体都落在内置表时给出，否则留空（不猜）。
    """
    if central is None or not isinstance(central, str) or not central.strip():
        return _err("缺少中心原子：请给 central，如 \"S\"、\"Xe\"、\"Cl\"（元素符号）。")
    central_s = central.strip()
    central_s = central_s[0].upper() + central_s[1:].lower() if len(central_s) > 1 else central_s.upper()

    if ligands is None or isinstance(ligands, str):
        if isinstance(ligands, str):
            return _err("ligands 必须是列表，如 [\"F\",\"F\",\"F\",\"F\"]；只给一个符号时用 [\"F\"]。")
        return _err("缺少配体：ligands 必须是元素符号列表，如 [\"F\",\"F\",\"F\",\"F\"]。")
    try:
        ligands_l = [str(x).strip() for x in ligands]
    except TypeError:
        return _err("ligands 必须是元素符号列表，如 [\"F\",\"F\",\"F\",\"F\"]。")
    if not ligands_l:
        return _err("ligands 不能为空：至少需要 1 个配体。")
    if any(not s for s in ligands_l):
        return _err("ligands 里有空字符串，请检查。")
    ligands_l = [s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper() for s in ligands_l]

    try:
        n_lone = int(lone_pairs)
    except (TypeError, ValueError):
        return _err("lone_pairs 必须是整数（中心原子的孤对电子数）。")
    if isinstance(lone_pairs, float) and abs(lone_pairs - n_lone) > 1e-12:
        return _err(f"lone_pairs 必须是整数，收到 {lone_pairs}。")
    if n_lone < 0:
        return _err("lone_pairs 不能为负。")

    n_bond = len(ligands_l)
    sn = n_bond + n_lone
    if sn < 1:
        return _err("价层电子对数至少为 1。")

    key = (n_bond, n_lone)
    table = _VSEPR_TABLE.get(key)
    if table is None:
        if sn > 7:
            return _err(
                f"价层电子对数 = {sn} 超过 7，超出本工具覆盖范围（2–7 对）。"
                "请核对配体数与孤对数。"
            )
        return _err(
            f"缺少 {_vsepr_symbol(n_bond, n_lone)}（价层电子对数 {sn}）的构型数据。"
            "本工具覆盖：(2,0) (3,0) (2,1) (4,0) (3,1) (2,2) (5,0) (4,1) (3,2) (2,3) "
            "(6,0) (5,1) (4,2) (7,0) (6,1) (5,2)。"
        )

    vsepr_type = _vsepr_symbol(n_bond, n_lone)
    examples = list(_VSEPR_EXAMPLES.get(key, []))

    # ---------- 配体电荷 ----------
    charges: list[float] = []
    charge_note = ""
    if ligand_charges is not None:
        if isinstance(ligand_charges, (int, float)):
            ligand_charges = [ligand_charges] * n_bond
        try:
            charges = [float(c) for c in ligand_charges]
        except (TypeError, ValueError):
            return _err("ligand_charges 必须是数值列表（与 ligands 等长），如 [\"-1\",\"-1\"]。")
        if len(charges) != n_bond:
            return _err(
                f"ligand_charges 长度 {len(charges)} 与配体数 {n_bond} 不一致，请一一对应。"
            )
        charge_note = f"配体电荷合计 {_fmt(sum(charges), 3)}。"
    try:
        total_charge = float(charge)
    except (TypeError, ValueError):
        return _err("charge 必须是数值（整个离子的电荷，如 NH₄⁺ 传 1）。")
    if ligand_charges is None:
        charges = [0.0] * n_bond

    # ---------- 电子对数交叉核对（教材口径） ----------
    valence_check = None
    ve = _VALENCE_E.get(central_s)
    if ve is not None and all(l in _LIGAND_E for l in ligands_l):
        e_total = (
            ve
            + sum(_LIGAND_E[l] for l in ligands_l)
            + sum(-c for c in charges)
            - total_charge
        )
        if abs(e_total - round(e_total)) < 1e-9 and int(round(e_total)) % 2 == 0:
            pairs = int(round(e_total)) // 2
            valence_check = {
                "center_valence_electrons": ve,
                "ligand_electrons": _fmt(sum(_LIGAND_E[l] for l in ligands_l), 3),
                "sum_electrons": _fmt(e_total, 3),
                "predicted_pairs": pairs,
                "steric_number": sn,
                "consistent": pairs == sn,
                "formula": "(中心价电子数 + 配体提供电子数 - 配体电荷合计 - 离子电荷)/2",
            }

    # ---------- steps ----------
    steps: list[str] = []
    steps.append(
        f"① 数价层电子对：配体 {n_bond} 个 + 中心孤对 {n_lone} 对 = 空间构型数 {sn}。"
    )
    if valence_check is not None:
        steps.append(
            f"   电子对核算：({valence_check['center_valence_electrons']} + "
            f"{valence_check['ligand_electrons']}"
            + (f" + 配体电荷 {-sum(charges):+g}" if any(charges) else "")
            + (f" - 离子电荷 {total_charge:+g}" if total_charge else "")
            + f")/2 = {valence_check['predicted_pairs']} 对，"
            + ("与上式一致。" if valence_check["consistent"] else f"与 {sn} 不一致，请核对孤对数。")
        )
    steps.append(
        f"② {sn} 对电子在中心原子周围的排斥最小排布 → 电子对构型为"
        f"{ {2:'直线',3:'平面三角',4:'四面体',5:'三角双锥',6:'八面体',7:'五角双锥'}.get(sn, '—') }。"
    )
    steps.append(f"③ 扣掉 {n_lone} 对孤对（孤对不占键的方向位置）→ 分子构型为{table['geometry']}，记作 {vsepr_type}。")
    steps.append(f"④ 孤对占位：{table['lone_pair_sites']}")
    steps.append(
        f"⑤ 杂化：{sn} 对电子需要 {sn} 个等价杂化轨道 → {table['hybrid_display']}；"
        f"键角 {table['bond_angles']}；点群 {table['point_group']}。"
    )
    if examples:
        steps.append(f"⑥ 同型实例：{'、'.join(examples[:5])}。")

    return {
        "ok": True,
        "error": None,
        "central": central_s,
        "ligands": ligands_l,
        "n_ligands": n_bond,
        "lone_pairs": n_lone,
        "steric_number": sn,
        "electron_pair_geometry": {
            2: "直线形", 3: "平面三角形", 4: "正四面体",
            5: "三角双锥", 6: "正八面体", 7: "五角双锥",
        }.get(sn, "—"),
        "vsepr_type": vsepr_type,
        "geometry": table["geometry"],
        "hybrid": table["hybrid"],
        "hybrid_display": table["hybrid_display"],
        "bond_angles": table["bond_angles"],
        "point_group": table["point_group"],
        "lone_pair_sites": table["lone_pair_sites"],
        "ligand_charges": charges,
        "charge": total_charge,
        "charge_note": charge_note,
        "valence_check": valence_check,
        "examples": examples,
        "steps": steps,
    }


# =====================================================================
# 自测（契约第 4 节 + 教材标准答案）
# =====================================================================


def _selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    def close(a, b, tol=1e-6) -> bool:
        return abs(float(a) - float(b)) <= tol

    def seq_close(seq, ref, tol=1e-6) -> bool:
        return len(seq) == len(ref) and all(close(a, b, tol) for a, b in zip(seq, ref))

    # ---------------- huckel ----------------
    r = huckel("cyclic", n_electrons=6, n_atoms=6)
    ck("huckel/苯 ok", r.get("ok") is True, str(r.get("error")))
    if r.get("ok"):
        lam = [o["E_beta"] for o in r["orbitals"]]
        labels = [o["E_alpha_beta"] for o in r["orbitals"]]
        ck("huckel/苯 能级", seq_close(lam, [2, 1, 1, -1, -1, -2], 1e-6), str(lam))
        ck(
            "huckel/苯 能级字符串",
            labels == ["α+2β", "α+β", "α+β", "α-β", "α-β", "α-2β"],
            str(labels),
        )
        ck("huckel/苯 离域能 = 2β", close(r["delocalization_energy_beta"], 2.0), str(r["delocalization_energy_beta"]))
        ck("huckel/苯 总 π 能 = 6α+8β", close(r["pi_energy_total_beta"], 8.0) and r["pi_energy_total_str"] == "6α + 8β", r["pi_energy_total_str"])
        ck("huckel/苯 总 π 能数值(β=-1)", close(r["pi_energy_total"], -8.0), str(r["pi_energy_total"]))
        bos = [b["order"] for b in r["bond_orders"]]
        ck("huckel/苯 C–C 键级 0.667", all(close(v, 2.0 / 3.0, 1e-6) for v in bos), str(bos))
        ck("huckel/苯 电荷密度 1.000", all(close(v, 1.0, 1e-6) for v in r["charge_densities"]), str(r["charge_densities"]))
        ck("huckel/苯 自由价 0.398", all(close(v, 1.7320508 - 4.0 / 3.0, 1e-6) for v in r["free_valence"]), str(r["free_valence"]))
        ck("huckel/苯 steps 含行列式", any("x^6 - 6x^4 + 9x^2 - 4" in s for s in r["steps"]), "")
        ck("huckel/苯 steps 数 ≥ 8", len(r["steps"]) >= 8, str(len(r["steps"])))
        ck("huckel/苯 HOMO-LUMO gap = 2β", close(r["homo_lumo_gap_beta"], 2.0), str(r["homo_lumo_gap_beta"]))

    r = huckel("cyclic", n_electrons=2, n_atoms=3)
    ck("huckel/环丙烯正离子 ok", r.get("ok") is True, str(r.get("error")))
    if r.get("ok"):
        lam = [o["E_beta"] for o in r["orbitals"]]
        ck("huckel/环丙烯正离子 能级", seq_close(lam, [2, -1, -1], 1e-6), str(lam))
        ck(
            "huckel/环丙烯正离子 能级字符串",
            [o["E_alpha_beta"] for o in r["orbitals"]] == ["α+2β", "α-β", "α-β"],
            str([o["E_alpha_beta"] for o in r["orbitals"]]),
        )
        ck("huckel/环丙烯正离子 离域能 = 2β", close(r["delocalization_energy_beta"], 2.0), str(r["delocalization_energy_beta"]))
        ck("huckel/环丙烯正离子 steps 含 x^3 - 3x - 2", any("x^3 - 3x - 2" in s for s in r["steps"]), "")

    r = huckel("linear", n_electrons=4, n_atoms=4)
    ck("huckel/丁二烯 ok", r.get("ok") is True, str(r.get("error")))
    if r.get("ok"):
        lam = [o["E_beta"] for o in r["orbitals"]]
        ck(
            "huckel/丁二烯 能级",
            seq_close(lam, [1.618034, 0.618034, -0.618034, -1.618034], 1e-5),
            str(lam),
        )
        ck(
            "huckel/丁二烯 能级字符串",
            [o["E_alpha_beta"] for o in r["orbitals"]] == ["α+1.618β", "α+0.618β", "α-0.618β", "α-1.618β"],
            str([o["E_alpha_beta"] for o in r["orbitals"]]),
        )
        ck("huckel/丁二烯 离域能 0.472β", close(r["delocalization_energy_beta"], 4.47213595 - 4.0, 1e-5), str(r["delocalization_energy_beta"]))
        ck("huckel/丁二烯 steps 含 x^4 - 3x^2 + 1", any("x^4 - 3x^2 + 1" in s for s in r["steps"]), "")
        ck("huckel/丁二烯 端基键级最大", r["bond_orders"][0]["order"] > r["bond_orders"][1]["order"], "")

    # 自定义邻接矩阵 == cyclic
    benz = [[0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]]
    r1 = huckel("ring", n_electrons=6, adj=benz)
    r2 = huckel("cyclic", n_electrons=6, n_atoms=6)
    ck("huckel/adj 与 cyclic 一致", r1.get("ok") and r2.get("ok")
       and seq_close([o["E_beta"] for o in r1["orbitals"]], [o["E_beta"] for o in r2["orbitals"]]), "")

    ck("huckel/非法 n_electrons 超容量", huckel("cyclic", n_electrons=100, n_atoms=3).get("ok") is False, "")
    ck("huckel/β>0 报错", huckel("cyclic", n_electrons=6, n_atoms=6, beta=1.0).get("ok") is False, "")
    ck("huckel/缺 n_atoms 报错", huckel("cyclic").get("ok") is False, "")
    ck("huckel/未知 topology 报错", huckel("banana", n_atoms=4).get("ok") is False, "")
    ck("huckel/非对称矩阵报错", huckel(None, adj=[[0, 1], [0, 0]]).get("ok") is False, "")

    # 环丁二烯（4 元环 4π）：正方形反芳香，离域能 0，简并半充满 → 必须告警
    r = huckel("cyclic", n_electrons=4, n_atoms=4)
    if r.get("ok"):
        lam = [o["E_beta"] for o in r["orbitals"]]
        ck("huckel/环丁二烯 能级 2,0,0,-2", seq_close(lam, [2, 0, 0, -2], 1e-6), str(lam))
        ck("huckel/环丁二烯 离域能 0", close(r["delocalization_energy_beta"], 0.0, 1e-9), str(r["delocalization_energy_beta"]))
        ck("huckel/环丁二烯 部分占据告警", any("未被完全占据" in w for w in r["warnings"]), str(r["warnings"])[:80])

    # 交替烃 Coulson–Rushbrooke 成对定理：无奇元环时能级关于 α 对称（±x 成对）
    for topo, nn, ne in (("linear", 4, 4), ("cyclic", 6, 6), ("cyclic", 8, 8), ("cyclic", 4, 4), ("linear", 6, 6)):
        rr = huckel(topo, n_electrons=ne, n_atoms=nn)
        ls = sorted(o["E_beta"] for o in rr["orbitals"])
        paired = all(close(ls[i], -ls[len(ls) - 1 - i], 1e-8) for i in range(len(ls) // 2))
        ck(f"huckel/{topo}{nn} 交替烃能级±x成对", paired, str(ls))

    # 电荷密度守恒 Σq_r = n_π（占据轨道系数的完备性）
    for topo, nn, ne in (("linear", 4, 4), ("cyclic", 6, 6), ("linear", 5, 5)):
        rr = huckel(topo, n_electrons=ne, n_atoms=nn)
        ck(f"huckel/{topo}{nn} Σq_r = n_π", close(sum(rr["charge_densities"]), float(ne), 1e-7),
           f"{sum(rr['charge_densities'])} vs {ne}")

    # ---------------- aromaticity ----------------
    a = aromaticity(6, 6)
    ck("aromaticity/苯 → 芳香", a.get("ok") and a["aromatic"] is True and a["n"] == 1, str(a.get("reason"))[:60])
    a = aromaticity(8, 8)
    ck("aromaticity/环辛四烯(平面) → 反芳香", a.get("ok") and a["aromatic"] is False and a["aromatic_type"] == "反芳香", str(a.get("aromatic_type")))
    a = aromaticity(8, 8, planar=False)
    ck("aromaticity/环辛四烯(非平面) → 非芳香", a.get("ok") and a["aromatic_type"] == "非芳香", str(a.get("aromatic_type")))
    a = aromaticity(2, 3)
    ck("aromaticity/环丙烯正离子 → 芳香(n=0)", a.get("ok") and a["aromatic_type"] == "芳香" and a["n"] == 0, str(a.get("aromatic_type")))
    a = aromaticity(6, 5)
    ck("aromaticity/环戊二烯负离子 → 芳香", a.get("ok") and a["aromatic_type"] == "芳香", str(a.get("aromatic_type")))
    a = aromaticity(4, 5)
    ck("aromaticity/环戊二烯正离子 → 反芳香", a.get("ok") and a["aromatic_type"] == "反芳香", str(a.get("aromatic_type")))
    a = aromaticity(6, 6, ring=False)
    ck("aromaticity/开链 → 非芳香", a.get("ok") and a["aromatic_type"] == "非芳香", str(a.get("aromatic_type")))
    a = aromaticity(6, 6)
    names = [e["name"] for e in a["examples"]] if a.get("ok") else []
    need = ["苯", "环辛四烯", "环戊二烯", "吡啶", "环丙烯正离子"]
    ck("aromaticity/examples 覆盖指定体系", all(any(k in n for n in names) for k in need), str(names)[:80])
    ck("aromaticity/非法 n_pi=0", aromaticity(0, 0).get("ok") is False, "")

    # ---------------- diatomic_mo ----------------
    cases = [
        ("O2", 2.0, 2, "顺磁"),
        ("N2", 3.0, 0, "抗磁"),
        ("F2", 1.0, 0, "抗磁"),
        ("NO", 2.5, 1, "顺磁"),
        ("O2+", 2.5, 1, "顺磁"),
        ("O2-", 1.5, 1, "顺磁"),
        ("N2+", 2.5, 1, "顺磁"),
        ("B2", 1.0, 2, "顺磁"),
        ("C2", 2.0, 0, "抗磁"),
        ("Ne2", 0.0, 0, "抗磁"),
    ]
    for label, bo, up, mag in cases:
        r = diatomic_mo(homo=label)
        ok = (
            r.get("ok") is True
            and close(r["bond_order"], bo, 1e-9)
            and r["unpaired_electrons"] == up
            and r["magnetism"] == mag
        )
        ck(f"diatomic/{label} 键级 {bo} / {up} 单电子 / {mag}", ok, f"{r.get('bond_order')} {r.get('unpaired_electrons')} {r.get('magnetism')}")

    # 电子数口径回归（Lead 验收项）：n_electrons 默认按【总电子数】理解
    total_cases = [
        (16, "O2", 2.0, 2, "顺磁"),
        (14, "N2", 3.0, 0, "抗磁"),
        (15, "NO", 2.5, 1, "顺磁"),
        (18, "F2", 1.0, 0, "抗磁"),
        (17, "O2-", 1.5, 1, "顺磁"),
        (19, "F2-", 0.5, 1, "顺磁"),
    ]
    for n_tot, expect_species, bo, up, mag in total_cases:
        r = diatomic_mo(n_tot)
        conv = n_tot - 4
        ok = (
            r.get("ok") is True
            and r["input_mode"] == "total"
            and r["n_electrons"] == n_tot
            and r["n_valence_electrons"] == conv
            and r["core_electrons"] == 4
            and r["species"] == expect_species
            and r["species_source"] == "electron_count"
            and any(f"价电子 n = {n_tot} - 4 = {conv}" in s for s in r["steps"])
            and close(r["bond_order"], bo, 1e-9)
            and r["unpaired_electrons"] == up
            and r["magnetism"] == mag
        )
        ck(f"diatomic/总电子数 {n_tot} → {conv} 价电子 / {expect_species}", ok,
           f"species={r.get('species')} BO={r.get('bond_order')} up={r.get('unpaired_electrons')} "
           f"mode={r.get('input_mode')} nval={r.get('n_valence_electrons')}")

    # 物种名必须跟着电子数走，不能被某个默认分子名顶替
    for n_tot, name in ((14, "N2"), (16, "O2"), (15, "NO"), (18, "F2"), (17, "O2-")):
        r = diatomic_mo(n_tot)
        ck(f"diatomic/n_electrons={n_tot} → species 含 {name}",
           r.get("ok") and r["species"] is not None and name in r["species"], str(r.get("species")))

    # 13 总电子（价电子 9）对应 C2⁻ / CN / CO⁺ / N2⁺ 等多个物种 → 必须给 null，不许猜
    r = diatomic_mo(13)
    ck("diatomic/总电子 13（价 9 多解）→ species 为 null 且给候选",
       r.get("ok") is True and r["species"] is None and len(r["species_candidates"]) > 0
       and any("无法由电子数唯一确定" in s for s in r["steps"]),
       f"{r.get('species')} {r.get('species_candidates')}")
    ck("diatomic/多解时键级仍正确（价 9 → 2.5）", r.get("ok") and close(r["bond_order"], 2.5, 1e-9), str(r.get("bond_order")))

    r = diatomic_mo(16)
    ck("diatomic/n_electrons=16 → O2 键级 2 / 2 单电子 / 顺磁 / μ≈2.83",
       r.get("ok") and close(r["bond_order"], 2.0, 1e-9) and r["unpaired_electrons"] == 2
       and r["magnetism"] == "顺磁" and close(r["magnetic_moment_BM"], 2.8284, 1e-3),
       f"{r.get('bond_order')} {r.get('unpaired_electrons')} {r.get('magnetic_moment_BM')}")
    r = diatomic_mo(14)
    ck("diatomic/n_electrons=14 → N2 键级 3 抗磁",
       r.get("ok") and close(r["bond_order"], 3.0, 1e-9) and r["magnetism"] == "抗磁", str(r.get("bond_order")))
    r = diatomic_mo(15)
    ck("diatomic/n_electrons=15 → NO 键级 2.5 / 1 单电子 / 顺磁",
       r.get("ok") and close(r["bond_order"], 2.5, 1e-9) and r["unpaired_electrons"] == 1
       and r["magnetism"] == "顺磁", str(r.get("bond_order")))
    r = diatomic_mo(n_valence_electrons=12)
    ck("diatomic/n_valence_electrons=12 → 同 O2（键级 2，2 单电子）",
       r.get("ok") and r["input_mode"] == "valence" and r["n_valence_electrons"] == 12
       and close(r["bond_order"], 2.0, 1e-9) and r["unpaired_electrons"] == 2, str(r.get("bond_order")))
    r = diatomic_mo(18)
    ck("diatomic/n_electrons=18 → F2 键级 1 抗磁",
       r.get("ok") and close(r["bond_order"], 1.0, 1e-9) and r["magnetism"] == "抗磁", str(r.get("bond_order")))
    r = diatomic_mo(17)
    ck("diatomic/n_electrons=17 → O2⁻ 键级 1.5",
       r.get("ok") and close(r["bond_order"], 1.5, 1e-9), str(r.get("bond_order")))
    ck("diatomic/两个电子数参数同给要报错",
       diatomic_mo(16, n_valence_electrons=12).get("ok") is False, "")
    ck("diatomic/总电子数过少要报错", diatomic_mo(2).get("ok") is False, "")
    r = diatomic_mo(16)
    ck("diatomic/O2 用 O2 型顺序(σ2pz 在 π 前)", r.get("ok") and "O2 型" in r["level_scheme"], str(r.get("level_scheme")))
    r = diatomic_mo(14)
    ck("diatomic/N2 用 N2 型顺序(σ2pz 在 π 后)", r.get("ok") and "N2 型" in r["level_scheme"], str(r.get("level_scheme")))
    r = diatomic_mo(homo="O2")
    ck("diatomic/O2 组态串", r.get("ok") and r["config"] == "(σ2s)²(σ*2s)²(σ2pz)²(π2px = π2py)⁴(π*2px = π*2py)²", str(r.get("config")))
    ck("diatomic/显式 homo 时物种名来自 homo",
       r.get("ok") and r["species"] == "O2" and r["species_source"] == "homo", str(r.get("species_source")))
    ck("diatomic/无参调用现在报错（不再默认 O2）", diatomic_mo().get("ok") is False, "")
    r = diatomic_mo(homo="N2")
    ck("diatomic/N2 组态串", r.get("ok") and r["config"] == "(σ2s)²(σ*2s)²(π2px = π2py)⁴(σ2pz)²", str(r.get("config")))
    ck("diatomic/period=3 报错", diatomic_mo(16, period=3).get("ok") is False, "")
    ck("diatomic/缺电子数报错", diatomic_mo(None, homo="XYZ").get("ok") is False, "")
    r = diatomic_mo(homo="H2")
    ck("diatomic/H2 自动 period=1 键级 1",
       r.get("ok") and close(r["bond_order"], 1.0) and r["period"] == 1
       and r["input_mode"] == "species" and r["core_electrons"] == 0, str(r.get("bond_order")))
    r = diatomic_mo(2, period=1)
    ck("diatomic/总电子 2 + period=1 → H2 键级 1",
       r.get("ok") and close(r["bond_order"], 1.0) and r["input_mode"] == "total", str(r.get("bond_order")))
    r = diatomic_mo(4, period=1)
    ck("diatomic/总电子 4 + period=1 → He2 键级 0",
       r.get("ok") and close(r["bond_order"], 0.0, 1e-9) and r["magnetism"] == "抗磁", str(r.get("bond_order")))

    # ---------------- vsepr ----------------
    vcases = [
        (("S", ["F"] * 4, 1), "AX4E", "跷跷板", "C2v", "sp3d"),
        (("Xe", ["F"] * 4, 2), "AX4E2", "平面四方", "D4h", "sp3d2"),
        (("Cl", ["F"] * 3, 2), "AX3E2", "T 形", "C2v", "sp3d"),
        (("C", ["H"] * 4, 0), "AX4", "正四面体", "Td", "sp3"),
        (("N", ["H"] * 3, 1), "AX3E", "三角锥", "C3v", "sp3"),
        (("O", ["H"] * 2, 2), "AX2E2", "角形", "C2v", "sp3"),
        (("C", ["O", "O"], 0), "AX2", "直线形", "D∞h", "sp"),
        (("B", ["F"] * 3, 0), "AX3", "平面三角形", "D3h", "sp2"),
        (("S", ["O", "O"], 1), "AX2E", "角形", "C2v", "sp2"),
        (("P", ["Cl"] * 5, 0), "AX5", "三角双锥", "D3h", "sp3d"),
        (("Xe", ["F"] * 2, 3), "AX2E3", "直线形", "D∞h", "sp3d"),
        (("S", ["F"] * 6, 0), "AX6", "正八面体", "Oh", "sp3d2"),
        (("Br", ["F"] * 5, 1), "AX5E", "四方锥", "C4v", "sp3d2"),
    ]
    for (central, ligs, lp), vt, geo, pg, hyb in vcases:
        r = vsepr(central, ligs, lone_pairs=lp)
        ok = (
            r.get("ok") is True
            and r["vsepr_type"] == vt
            and geo in r["geometry"]
            and r["point_group"] == pg
            and r["hybrid"] == hyb
            and r["steric_number"] == len(ligs) + lp
        )
        ck(f"vsepr/{central}{ligs}{lp} → {vt} {geo} {pg}", ok,
           f"{r.get('vsepr_type')} {r.get('geometry')} {r.get('point_group')} {r.get('hybrid')}")

    r = vsepr("S", ["F"] * 4, lone_pairs=1)
    ck("vsepr/SF4 孤对占赤道位", r.get("ok") and "赤道" in r["lone_pair_sites"], "")
    r = vsepr("Xe", ["F"] * 4, lone_pairs=2)
    ck("vsepr/XeF4 孤对取反位", r.get("ok") and "反位" in r["lone_pair_sites"], "")
    r = vsepr("N", ["H"] * 4, lone_pairs=0, charge=1)
    ck("vsepr/NH4+ 电子对核算一致", r.get("ok") and r["valence_check"] and r["valence_check"]["consistent"] is True, str(r.get("valence_check")))
    r = vsepr("S", ["F"] * 4, lone_pairs=1)
    ck("vsepr/SF4 电子对核算一致", r.get("ok") and r["valence_check"] and r["valence_check"]["consistent"] is True, str(r.get("valence_check")))
    ck("vsepr/缺 central 报错", vsepr(None, ["H"]).get("ok") is False, "")
    ck("vsepr/缺 ligands 报错", vsepr("C", []).get("ok") is False, "")
    ck("vsepr/超范围 8 对报错", vsepr("Xe", ["F"] * 8, 0).get("ok") is False, "")

    # ---------------- 汇总 ----------------
    fails = [c for c in checks if not c[1]]
    print("=" * 68)
    print(f"orbital.py 自测：共 {len(checks)} 项，通过 {len(checks) - len(fails)}，失败 {len(fails)}")
    print("=" * 68)
    for name, ok, detail in checks:
        if not ok:
            print(f"[FAIL] {name}    {detail}")
    if not fails:
        print("全部通过。")
        # 关键结果摘要
        b = huckel("cyclic", n_electrons=6, n_atoms=6)
        print("苯 6π：", "、".join(o["E_alpha_beta"] for o in b["orbitals"]),
              f"| E_π = {b['pi_energy_total_str']} | 离域能 {b['delocalization_energy_str']}")
        d = huckel("linear", n_electrons=4, n_atoms=4)
        print("丁二烯 4π：", "、".join(o["E_alpha_beta"] for o in d["orbitals"]),
              f"| 离域能 {d['delocalization_energy_str']}")
        c = huckel("cyclic", n_electrons=2, n_atoms=3)
        print("环丙烯正离子 2π：", "、".join(o["E_alpha_beta"] for o in c["orbitals"]),
              f"| 离域能 {c['delocalization_energy_str']}")
        o2 = diatomic_mo(homo="O2")
        n2 = diatomic_mo(homo="N2")
        print(f"O2: {o2['config']} 键级 {o2['bond_order_str']} 未成对 {o2['unpaired_electrons']} {o2['magnetism']}")
        print(f"N2: {n2['config']} 键级 {n2['bond_order_str']} 未成对 {n2['unpaired_electrons']} {n2['magnetism']}")
        sf4 = vsepr("S", ["F"] * 4, lone_pairs=1)
        print(f"SF4: {sf4['vsepr_type']} {sf4['geometry']} {sf4['point_group']} {sf4['hybrid_display']}")
        print("=" * 68)
    return len(fails)


if __name__ == "__main__":
    import sys

    sys.exit(1 if _selftest() else 0)
