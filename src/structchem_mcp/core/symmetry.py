# -*- coding: utf-8 -*-
"""分子对称性：点群判定（真算）、对称元素、特征标表、约化、3N 模式、选律、操作乘法。

算什么
------
S1 point_group          从原子坐标出发，枚举 3×3 矩阵作用于坐标后做「同元素一一对应」集合
                        匹配，检验真转动/反演/反映/映轴，群元乘法闭包，再按群论判据树定
                        Schoenflies 记号。不用查表法猜。
S2 symmetry_elements    点群 → 阶、类、对称元素清单、生成元。
S3 character_table      读 data/character_tables.json（教材附表口径）。
S4 reduce_representation  n_i = (1/h)Σ_c g_c·χ_i(c)·χ_r(c)，带还原自校验。
S5 normal_modes         Γ_3N = Γ_trans + Γ_rot + Γ_vib，IR/Raman 选律。
S6 selection_rules      偶极矩 / 旋光性 / 允许的基函数。
S7 compose_operations   两个对称操作的乘积（3×3 矩阵 + 行列式 + 等价操作名）。

边界（不做什么）
----------------
* 只处理有限点群 + 线形群（C∞v / D∞h）；单原子返回 K_h 并在 notes 里说明教材不讨论。
* 不含自旋、不含双群、不做振动频率数值（只做对称性分类）。
* 输入非法返回 {"ok": False, "error": "中文可执行原因"}，不抛裸异常。

教材口径（重要）
----------------
1. 坐标一律先按**几何质心**平移（任一对称操作都固定质心，故必为操作的不动点）。
2. 容差 tol 默认 0.15 Å（15 pm），判定用「同元素完全匹配」：二部图最大匹配，全部落在 tol 内。
3. 循环群 C_n(n≥3)、C_nh(n 奇)、S_2n、T、Th 的 E 表示按**教材合并约定**
   （一对共轭一维复表示并写成一个二维 E，特征标 2cos），约化时用
   n=(1/(2h))Σ g_c·χ_i·χ_r；这些表示在数据表里带 "merged": true。
4. 轴向群取 z 为主轴；D_nh 中 C2' 轴取与原子共线者，σv 含 C2' 轴、σd 平分之。
5. 线形分子按 C∞v / D∞h 的 Σ/Π 计数法处理（不用有限群的约化公式）。

S1 里两个必须写明的实现决策（坑都在这里）
----------------------------------------
A. **每个检出的操作都要 Kabsch 精修（`_refine_op`），再做群元闭包。**
   候选方向里必然混有「略微偏斜的轴」（例如椅式环己烷的 C2' 轴候选就有
   (0.866,±0.5,0) 与 (0.871,−0.491,0) 两个近乎重合的方向），二者在 tol=0.15 Å 下
   **都能通过匹配检验**，但只有前者是真正的群元。不精修就直接做闭包，误差会在乘法中
   累积出「转角 1.15° 的伪恒等元」，群阶从 12 膨胀到 132（苯、萘、环丙烯正离子
   都会中招），随后判据树必然判不出来。精修后不同候选轴收敛到**同一个**矩阵，
   按矩阵键去重即生效，闭包精确。

B. **大群（O / T / I / Ih）不做「构造分子 → point_group」的端到端反验。**
   一个只有 O（而非 Oh）对称性的点集需要 ≥24 个一般位置原子，I/Ih 要 60/120 个，
   代价不成比例。改为把判据树抽成纯矩阵函数 `_classify_mats(mats)`，直接拿
   `_build_group(点名)` 的全部群元做**39 个点群的全量反验**；坐标侧的端到端覆盖由
   CH2Cl2/苯/SF6/丙二烯/椅式环己烷/萘/BF3/CH4/H2O2/HCN/CO2/反式二氯乙烯/
   环丙烯正离子/XeF4/正八面体/正四面体/正二十面体等 21 个用例承担。
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np

try:  # 允许 `python -m structchem_mcp.core.symmetry` 与直接脚本两种跑法
    from ._const import PM_PER_A
except ImportError:  # pragma: no cover
    PM_PER_A = 100.0

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "character_tables.json"
)

_META = {"class_irrep_theorem": "不可约表示数 = 类的数目"}
_TABLES: dict = {}


def _load_data() -> dict:
    """惰性读一次特征标表；读不到就返回空表（各函数据此报错，不抛异常）。"""
    global _TABLES, _META
    if _TABLES:
        return _TABLES
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError:
        _TABLES = {}
        return _TABLES
    except json.JSONDecodeError:
        _TABLES = {}
        return _TABLES
    _META = raw.get("_meta", _META)
    for key, val in raw.items():
        if not key.startswith("_"):
            _TABLES[key] = val
    return _TABLES


def _err(msg: str) -> dict:
    return {"ok": False, "error": msg}


# ---------------------------------------------------------------------------
# 3×3 矩阵工具
# ---------------------------------------------------------------------------

_EPS_MAT = 1e-6


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return a / n


def _rot(axis, deg: float) -> np.ndarray:
    """绕 axis 转 deg 度的右手旋转矩阵（Rodrigues，手写以便 steps 可读）。"""
    a = _unit(axis)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _refl(normal) -> np.ndarray:
    """以 normal 为法向的反映面矩阵 σ = I - 2nnᵀ。"""
    n = _unit(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _srot(axis, deg: float) -> np.ndarray:
    """映轴操作 S = σ_h·C（同轴反射与旋转可交换，只含一次反射）。"""
    return _refl(axis) @ _rot(axis, deg)


def _sn(axis, n: int, k: int) -> np.ndarray:
    """真正的 S_n^k：转 k·360/n，**只有 k 为奇数时才含那一次反射**。

    易错点：S_n 的偶次幂是纯转动（S4² = C2、S8² = C4、S6² = C3），
    一律写成 σ_h·C 会把 S4 群的 C2 变成 i、把 D2d 的元素数搞错。
    """
    M = _rot(axis, 360.0 * k / n)
    return _refl(axis) @ M if k % 2 else M


def _same(A, B) -> bool:
    return bool(np.allclose(A, B, atol=_EPS_MAT))


def _rot_angle(M) -> float:
    """返回 (0,180] 内的转角（度）；对映轴操作用同一公式的 |θ|。"""
    tr = float(np.trace(M))
    det = float(np.linalg.det(M))
    if det > 0:
        c = (tr - 1.0) / 2.0
    else:  # 映轴/反映：tr = 2cosθ - 1
        c = (tr + 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _signed_angle(M) -> float:
    """带符号转角（用于区分 C3 与 C3²、S6 与 S6⁵）。"""
    ang = _rot_angle(M)
    if ang < 1e-9:
        return 0.0
    axis = _axis_of(M)
    if axis is None:
        return ang
    b = M - M.T
    v = np.array([b[2, 1] - b[1, 2], b[0, 2] - b[2, 0], b[1, 0] - b[0, 1]])
    return ang if float(np.dot(v, axis)) >= 0 else -ang


def _axis_of(M) -> np.ndarray | None:
    """真转动的转轴（特征值 +1 的特征向量）；非真转动给 None。"""
    if float(np.linalg.det(M)) < 0:
        return None
    w, vecs = np.linalg.eig(M)
    idx = int(np.argmin(np.abs(w - 1.0)))
    if abs(w[idx] - 1.0) > 1e-6:
        return None
    v = np.real(vecs[:, idx])
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _plane_normal(M) -> np.ndarray | None:
    """反映面法向（特征值 -1 的特征向量）。"""
    if abs(float(np.trace(M)) - 1.0) > 1e-6:
        return None
    w, vecs = np.linalg.eig(M)
    idx = int(np.argmin(np.abs(w + 1.0)))
    if abs(w[idx] + 1.0) > 1e-6:
        return None
    v = np.real(vecs[:, idx])
    return _unit(v)


def _dirkey(v, nd: int = 3) -> tuple:
    """方向去重键（把 v 与 -v 视为同一方向）。"""
    a = _unit(v)
    for x in a:
        if abs(x) > 1e-9:
            if x < 0:
                a = -a
            break
    return tuple(round(float(x), nd) + 0.0 for x in a)


def _elem_order(M) -> int:
    """该操作作为群元素的阶：最小的 m 使 M^m = E。

    注意不是 360/|θ| 直接取整——转角 144° 的元素阶是 5（144×5=720≡0 mod 360），
    而 360/144 = 2.5 取整会得到 2（这曾让 I 群的 12C5² 认不出阶、类标签配不上）。
    正确式：m = 360 / gcd(360, |θ|)。
    """
    ang = _rot_angle(M)
    if ang < 1e-6:
        return 1
    a = int(round(ang))
    if a <= 0:
        return 1
    return max(1, 360 // math.gcd(360, a))


_SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _sup(k: int) -> str:
    if 0 <= k <= 9:
        return _SUPS[k]
    return "^%d" % k


def _phi(m: int) -> int:
    """Euler φ。"""
    if m <= 0:
        return 0
    cnt, x, p = m, m, 2
    while p * p <= x:
        if x % p == 0:
            cnt -= cnt // p
            while x % p == 0:
                x //= p
        p += 1
    if x > 1:
        cnt -= cnt // x
    return cnt


# ---------------------------------------------------------------------------
# 原子输入与集合匹配
# ---------------------------------------------------------------------------

def _parse_atoms(atoms) -> tuple:
    """→ (符号列表, 坐标数组 N×3, 错误串)。只认 sym/element/symbol 与 xyz/(x,y,z)。"""
    if not isinstance(atoms, (list, tuple)):
        return [], None, "atoms 必须是列表，形如 [{\"sym\":\"H\",\"xyz\":[x,y,z]}, ...]"
    if len(atoms) == 0:
        return [], None, "atoms 为空：至少给 1 个原子（1 个原子返回 K_h）"
    syms, coords = [], []
    for i, a in enumerate(atoms):
        if not isinstance(a, dict):
            return [], None, "第 %d 个原子不是字典，应形如 {\"sym\":\"H\",\"xyz\":[0,0,0]}" % (i + 1)
        sym = a.get("sym") or a.get("element") or a.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            return [], None, "第 %d 个原子缺 sym（元素符号，如 \"H\"）" % (i + 1)
        xyz = a.get("xyz")
        if xyz is None and all(k in a for k in ("x", "y", "z")):
            xyz = [a["x"], a["y"], a["z"]]
        if not isinstance(xyz, (list, tuple, np.ndarray)) or len(xyz) != 3:
            return [], None, "第 %d 个原子的 xyz 必须是长度 3 的数组（Å）" % (i + 1)
        try:
            xyz = [float(v) for v in xyz]
        except (TypeError, ValueError):
            return [], None, "第 %d 个原子的 xyz 有非数字分量" % (i + 1)
        if not all(math.isfinite(v) for v in xyz):
            return [], None, "第 %d 个原子的 xyz 含 NaN/Inf" % (i + 1)
        syms.append(sym.strip())
        coords.append(xyz)
    arr = np.asarray(coords, dtype=float)
    keys = set(syms)
    if len(keys) == 1 and len(syms) == 1:
        return syms, arr, None
    return syms, arr, None


def _matching_pairs(adj: np.ndarray):
    """二部图完美匹配的**具体配对**。返回 [(行, 列)] 或 None。"""
    n = adj.shape[0]
    if n == 0:
        return []
    if adj.shape[0] != adj.shape[1]:
        return None
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching

        perm = maximum_bipartite_matching(csr_matrix(adj.astype(np.int8)), perm_type="column")
        if not bool(np.all(perm >= 0)):
            return None
        return [(int(perm[j]), int(j)) for j in range(n)]
    except Exception:  # pragma: no cover - scipy 不在时的兜底
        match_l = [-1] * n

        def dfs(u, seen):
            for v in range(n):
                if adj[u, v] and not seen[v]:
                    seen[v] = True
                    if match_l[v] < 0 or dfs(match_l[v], seen):
                        match_l[v] = u
                        return True
            return False

        for u in range(n):
            if not dfs(u, [False] * n):
                return None
        return [(int(match_l[j]), int(j)) for j in range(n)]


def _perfect_matching(adj: np.ndarray) -> bool:
    """二部图完美匹配存在性。"""
    return _matching_pairs(adj) is not None


def _match_ok(new: np.ndarray, old: np.ndarray, blocks: list, tol: float) -> bool:
    """同元素一一对应匹配：每块二部图必须存在完美匹配，且所有边权 ≤ tol。"""
    for idx in blocks:
        a = new[idx]
        b = old[idx]
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        if not _perfect_matching(d <= tol):
            return False
    return True


def _match_pairs(new: np.ndarray, old: np.ndarray, blocks: list, tol: float):
    """→ 全局原子配对 [(i, j)]（new_i ≈ old_j）或 None。"""
    pairs = []
    for idx in blocks:
        a = new[idx]
        b = old[idx]
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        loc = _matching_pairs(d <= tol)
        if loc is None:
            return None
        for r, c in loc:
            pairs.append((int(idx[r]), int(idx[c])))
    return pairs


def _dist_sig(coords: np.ndarray) -> np.ndarray:
    """距离多重集指纹（必须被对称操作保持的必要条件，用来便宜的否掉大部分候选）。"""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    iu = np.triu_indices(coords.shape[0], 1)
    return np.sort(d[iu])


def _op_ok(coords: np.ndarray, M: np.ndarray, blocks: list, tol: float) -> bool:
    new = coords @ M.T
    # 预筛：任一原子位移 ≤ tol 时，所有两两距离变化 ≤ 2·tol；不满足必然不是对称操作。
    # 这一步把绝大部分候选挡在二部图匹配之外（正二十面体那类场景提速一个量级）。
    if coords.shape[0] > 2:
        if np.max(np.abs(_dist_sig(new) - _dist_sig(coords))) > 2.0 * tol + 1e-9:
            return False
    return _match_ok(new, coords, blocks, tol)


def _refine_op(coords: np.ndarray, M: np.ndarray, blocks: list, tol: float):
    """把一个「差不多对」的操作精修成最近的正交变换（Kabsch，保持 det 符号）。

    为什么必须做：候选方向里存在略微偏斜的轴，在 tol=0.15 Å 下也能通过匹配检验，
    但这样的矩阵不是真正的群元；直接拿去做群元乘法闭包，误差会累积出
    「转角 1.15° 的伪恒等元」，群阶就会从 12 膨胀到 132。
    精修后不同候选轴会收敛到**同一个**矩阵，去重即生效，闭包变精确。
    """
    new = coords @ M.T
    pairs = _match_pairs(new, coords, blocks, tol)
    if pairs is None:
        return None
    P = np.array([coords[i] for i, _j in pairs], dtype=float)
    Q = np.array([coords[j] for _i, j in pairs], dtype=float)
    H = P.T @ Q
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    V = Vt.T
    want = 1.0 if float(np.linalg.det(M)) > 0 else -1.0
    best = None
    for d in (1.0, -1.0):
        R = V @ np.diag([1.0, 1.0, d]) @ U.T
        if abs(float(np.linalg.det(R)) - want) < 1e-8:
            if best is None or np.allclose(R, M, atol=1e-3):
                best = R
                if np.allclose(R, M, atol=1e-3):
                    break
    return best


def _mat_key(M: np.ndarray, nd: int = 5) -> tuple:
    return tuple(round(float(x), nd) + 0.0 for x in np.asarray(M).ravel())


def _op_dev(coords: np.ndarray, M: np.ndarray, blocks: list, tol: float) -> float:
    """该操作下的最大偏差（Å）——作为 detected 里的「重合度」。"""
    new = coords @ M.T
    pairs = _match_pairs(new, coords, blocks, max(tol, 1e-6))
    if pairs is None:
        return float("nan")
    return max((float(np.linalg.norm(new[i] - coords[j])) for i, j in pairs), default=0.0)


# ---------------------------------------------------------------------------
# 候选方向：任意对称轴的必过点集（原子径向 / 原子和 / 原子差 / 叉积 / 惯量主轴）
# ---------------------------------------------------------------------------

def _candidate_dirs(coords: np.ndarray) -> list:
    dirs: list = []
    seen: set = set()

    def add(v):
        a = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(a))
        if n < 1e-8:
            return
        a = a / n
        k = _dirkey(a)
        if k in seen:
            return
        seen.add(k)
        dirs.append(a)

    for v in coords:
        add(v)
    n_at = coords.shape[0]
    for i in range(n_at):
        for j in range(i + 1, n_at):
            ci, cj = coords[i], coords[j]
            add(ci + cj)
            add(ci - cj)
            add(np.cross(ci, cj))
    # 惯量主轴（能覆盖正多面体里没有任何原子对的轴向）
    try:
        w, vecs = np.linalg.eigh(coords.T @ coords)
        for k in range(3):
            add(np.real(vecs[:, k]))
    except np.linalg.LinAlgError:
        pass
    return dirs


def _detect_ops(coords: np.ndarray, blocks: list, tol: float, maxn: int = 8) -> dict:
    """枚举全部候选操作，逐个验证 + Kabsch 精修 + 去重，直接给可闭包的生成元。"""
    res = {
        "gens": [np.eye(3)],
        "tested": 0,
        "dirs": 0,
        "n_rot_axes": 0,
        "n_planes": 0,
        "n_improper": 0,
        "inversion": False,
        "_keys": {_mat_key(np.eye(3))},
    }
    dirs = _candidate_dirs(coords)
    res["dirs"] = len(dirs)

    def consider(M):
        """通过 tol 检验 → 精修 → 精修后仍需通过（且偏差不增大）→ 去重后收下。"""
        res["tested"] += 1
        if not _op_ok(coords, M, blocks, tol):
            return False
        R = _refine_op(coords, M, blocks, tol)
        if R is None:
            return False
        k = _mat_key(R)
        if k in res["_keys"]:
            return False
        res["_keys"].add(k)
        res["gens"].append(R)
        return True

    # 反演
    if consider(-np.eye(3)):
        res["inversion"] = True
    # 真转动
    rot_axes: dict = {}
    for u in dirs:
        for n in range(2, maxn + 1):
            if consider(_rot(u, 360.0 / n)):
                rot_axes.setdefault(_dirkey(u), []).append(n)
    res["n_rot_axes"] = len(rot_axes)
    # 反映面
    for u in dirs:
        if consider(_refl(u)):
            res["n_planes"] += 1
    # 映轴：只在已确认为真转动轴的方向上找（S_n^(n/2) 必含 C_(n/2) 型真转动）
    for u in dirs:
        if _dirkey(u) not in rot_axes:
            continue
        for n in range(3, maxn + 1):
            if consider(_srot(u, 360.0 / n)):
                res["n_improper"] += 1
    res.pop("_keys", None)
    return res


def _close_group(mats: list, coords: np.ndarray, blocks: list, tol: float, limit: int = 200) -> list:
    """群元闭包（队列式，O(h²)）：已检出的操作都合法，其乘积必然还是对称操作。

    去重用矩阵键（round 5 位）而不是逐个 allclose 扫描——否则是 O(h³)，
    Ih 那一档 120 阶会慢到不可用。
    """
    out: list = []
    keys: set = set()
    for m in mats:
        M = np.asarray(m, dtype=float)
        k = _mat_key(M)
        if k not in keys:
            keys.add(k)
            out.append(M)
    queue = list(range(len(out)))
    qi = 0
    while qi < len(queue) and len(out) < limit:
        i = queue[qi]
        qi += 1
        if i >= len(out):
            continue
        n_now = len(out)
        for j in range(n_now):
            for M in (out[i] @ out[j], out[j] @ out[i]):
                k = _mat_key(M)
                if k in keys:
                    continue
                if _op_ok(coords, M, blocks, tol):
                    keys.add(k)
                    out.append(M)
                    queue.append(len(out) - 1)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# 由点群名生成全部群元矩阵（compose_operations / normal_modes / 自校验共用）
# ---------------------------------------------------------------------------

_Z = np.array([0.0, 0.0, 1.0])
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])


def _c_merged_label(n: int, k: int) -> str:
    """循环群 C_n 教材合并约定下的类标签（k 与 n-k 合并）。"""
    k %= n
    if k == 0:
        return "E"
    m = n // math.gcd(n, k)
    if m == 2:
        return "C2"
    kk = min(k, n - k)
    if m < n:
        return "2C%d" % m
    return "2C%d%s" % (n, "" if kk == 1 else _sup(kk))


def _s_merged_label(total: int, k: int) -> str:
    """映轴类（σh·C^k 或 S_total^k）在合并约定下的标签。

    统一先取类内代表 j = min(k, total-k)（k 与 total-k 是同一类），再按阶给名；
    阶等于 total 时用教材的**奇数指数**写法（D5h 的 2S5³、D4d 的 2S8³ 都是这么来的）。
    """
    k %= total
    if k == 0:
        return "σh"
    j = min(k, total - k)
    m = total // math.gcd(total, j)
    if m == 2:
        return "i"
    if m < total:
        return "2S%d" % m
    e = j if j % 2 else j + total          # S_total^e，e 取奇数
    kk = min(e, 2 * total - e)             # 同一类里取角度 < 180° 的代表
    return "2S%d%s" % (total, "" if kk == 1 else _sup(kk))


def _cyc_el_label(total: int, k: int, proper: bool) -> str:
    """阿贝尔循环群（C_nh / S_2n / T 的循环子结构）里单个元素的标签。"""
    k %= total
    if k == 0:
        return "E"
    g = math.gcd(total, k)
    m = total // g
    e = k // g
    if not proper:
        if m == 2:
            return "i"
        if m == 1:
            return "σh"
        return "S%d%s" % (m, "" if e == 1 else _sup(e))
    return "C%d%s" % (m, "" if e == 1 else _sup(e))


def _parse_group_name(name: str):
    """→ (family, n)。family ∈ {C1,Cs,Ci,C,S,Cs_v,Cnh,D,Dnh,Dnd,T,Td,Th,O,Oh,I,Ih,Cinf,Dinf}"""
    s = (name or "").strip().replace(" ", "")
    if not s:
        return None, 0
    low = s.replace("σ", "s")
    if low.upper().startswith("C") and ("∞" in low or "INF" in low.upper()):
        return "Cinf", 0
    if low.upper().startswith("D") and ("∞" in low or "INF" in low.upper()):
        return "Dinf", 0
    if s in ("C1",):
        return "C1", 1
    if s in ("Cs", "CS"):
        return "Cs", 1
    if s in ("Ci", "CI"):
        return "Ci", 1
    for fam in ("Td", "Th", "Oh", "Ih", "T", "O", "I"):
        if s == fam:
            return fam, 0
    head = s[0].upper()
    tail = s[1:]
    if head in ("C", "D", "S") and tail.isdigit():
        n = int(tail)
        if n < 1 or n > 12:
            return None, 0
        if head == "S":
            if n % 2:
                return None, 0  # S_n(n 奇) 就是 C_nh，不单列
            return "S", n
        return head, n
    if tail.endswith("v") and tail[:-1].isdigit():
        return ("Cv" if head == "C" else None), int(tail[:-1])
    if tail.endswith("h") and tail[:-1].isdigit():
        n = int(tail[:-1])
        return ("Ch" if head == "C" else "Dh"), n
    if tail.endswith("d") and tail[:-1].isdigit():
        return ("Dd" if head == "D" else None), int(tail[:-1])
    return None, 0


_CUBIC = ("T", "Td", "Th", "O", "Oh", "I", "Ih")


def _signed_perm_groups():
    """O（24 个真转动）与 T（12 个）的全部矩阵。"""
    import itertools

    O = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(float(np.linalg.det(M)) - 1.0) < 1e-9:
                O.append(M)
    tet = [np.array(v, float) / math.sqrt(3.0) for v in
           ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    T = []
    for M in O:
        ok = True
        for v in tet:
            w = M @ v
            if not any(np.allclose(w, x, atol=1e-6) for x in tet):
                ok = False
                break
        if ok:
            T.append(M)
    return O, T


@lru_cache(maxsize=128)
def _build_group(name: str) -> tuple:
    """点群名 → 全部群元 [(3×3 矩阵, 类标签)]，标签与特征标表一致。"""
    fam, n = _parse_group_name(name)
    out: list = []

    def add(M, label):
        M = np.asarray(M, dtype=float)
        for M2, _l in out:
            if _same(M, M2):
                return
        out.append((M, label))

    if fam is None:
        return ()

    if fam == "C1":
        add(np.eye(3), "E")
    elif fam == "Ci":
        add(np.eye(3), "E")
        add(-np.eye(3), "i")
    elif fam == "Cs":
        add(np.eye(3), "E")
        add(_refl(_Y), "σh")
    elif fam == "C":
        for k in range(n):
            add(_rot(_Z, 360.0 * k / n), _c_merged_label(n, k))
    elif fam == "Cv":
        for k in range(n):
            add(_rot(_Z, 360.0 * k / n), _c_merged_label(n, k))
        for k in range(n):
            phi = 180.0 * k / n
            normal = np.array([math.cos(math.radians(phi + 90.0)),
                               math.sin(math.radians(phi + 90.0)), 0.0])
            if n == 2:
                lbl = "σv(xz)" if k == 0 else "σv'(yz)"
            elif n % 2 == 0:
                # C4v / C6v：含 C2' 轴的 2σv 与平分者的 2σd 是两类
                lbl = "%dσv" % (n // 2) if k % 2 == 0 else "%dσd" % (n // 2)
            else:
                lbl = "%dσv" % n
            add(_refl(normal), lbl)
    elif fam == "Ch":
        for k in range(n):
            add(_rot(_Z, 360.0 * k / n), _cyc_el_label(n, k, True))
        for j in range(n):
            M = _srot(_Z, 360.0 * j / n)          # σh·C_n^j 就是 S_n^j
            tr = float(np.trace(M))
            if j == 0:
                lbl = "σh"
            elif abs(tr + 3.0) < 1e-6:
                lbl = "i"                          # j = n/2 时 σh·C2 = i
            else:
                e = j if j % 2 else j + n          # 教材写法 S_n^e，e 取奇数
                lbl = "S%d%s" % (n, "" if e == 1 else _sup(e))
            add(M, lbl)
    elif fam == "S":
        for k in range(n):
            add(_sn(_Z, n, k), _cyc_el_label(n, k, k % 2 == 0))
    elif fam in ("D", "Dh", "Dd"):
        for k in range(n):
            lbl = _c_merged_label(n, k)
            if k == 1 and name in ("D2", "D2h"):
                lbl = "C2(z)"                      # 教材：D2/D2h 的主轴写作 C2(z)
            add(_rot(_Z, 360.0 * k / n), lbl)
        # 垂直 C2 轴：k=0 沿 x（与原子共线者，记作 C2'）
        for k in range(n):
            phi = 180.0 * k / n
            axis = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
            if n == 2 and fam != "Dd":
                lbl = "C2(x)" if k == 0 else "C2(y)"
            elif fam == "Dd":
                lbl = ("%dC2'" % n) if n % 2 == 0 else ("%dC2" % n)
            elif n % 2 == 0:
                lbl = "%dC2'" % (n // 2) if k % 2 == 0 else "%dC2''" % (n // 2)
            else:
                lbl = "%dC2" % n
            add(_rot(axis, 180.0), lbl)
        if fam == "Dh":
            for k in range(n):
                if n == 2:
                    lbl = "σ(xy)" if k == 0 else "i"   # D2h 的教材写法
                else:
                    lbl = _s_merged_label(n, k)
                add(_srot(_Z, 360.0 * k / n), lbl)
            # 含 C2' 轴的竖直面 = σv；平分者 = σd
            for k in range(n):
                phi = 180.0 * k / n
                normal = np.array([math.cos(math.radians(phi + 90.0)),
                                   math.sin(math.radians(phi + 90.0)), 0.0])
                if n == 2:
                    lbl = "σ(xz)" if k == 0 else "σ(yz)"
                elif n % 2 == 0:
                    lbl = "%dσv" % (n // 2) if k % 2 == 0 else "%dσd" % (n // 2)
                else:
                    lbl = "%dσv" % n
                add(_refl(normal), lbl)
        elif fam == "Dd":
            for k in range(1, 2 * n):           # S_2n 的奇次幂才是映轴，偶次幂是已有的 C_n^j
                add(_sn(_Z, 2 * n, k), _s_merged_label(2 * n, k))
            for k in range(n):
                phi = 180.0 * k / n + 90.0 / n      # 平分 C2' 轴
                normal = np.array([math.cos(math.radians(phi + 90.0)),
                                   math.sin(math.radians(phi + 90.0)), 0.0])
                add(_refl(normal), "%dσd" % n)
    elif fam in _CUBIC:
        O, T = _signed_perm_groups()
        if fam == "O":
            mats = O
        elif fam == "Oh":
            mats = O + [-np.eye(3) @ M for M in O]
        elif fam == "T":
            mats = T
        elif fam == "Td":
            sig = _refl(np.array([1.0, -1.0, 0.0]))
            mats = T + [sig @ M for M in T]
        elif fam == "Th":
            mats = T + [-np.eye(3) @ M for M in T]
        elif fam in ("I", "Ih"):
            gen = [_rot([0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0], 72.0),
                   _rot([1.0, 1.0, 1.0], 120.0)]
            cur = [np.eye(3)]
            changed = True
            while changed and len(cur) < 130:
                changed = False
                for M in list(cur):
                    for g in gen:
                        P = g @ M
                        if not any(_same(P, q) for q in cur):
                            cur.append(P)
                            changed = True
            mats = cur if fam == "I" else cur + [-np.eye(3) @ M for M in cur]
        else:  # pragma: no cover
            mats = []
        # 类标签：按 (size, kind, 元素阶, 有符号转角) 匹配特征标表
        for M in mats:
            add(M, None)
        labels = _cubic_labels(name, mats)
        out = [(M, labels.get(i)) for i, (M, _l) in enumerate(out)]
        return tuple(out)
    return tuple(out)


def _cubic_labels(name: str, mats: list) -> dict:
    """给立方/二十面体群的每个群元分配教材的类标签。"""
    tbl = _load_data().get(name)
    if not tbl:
        return {}
    want = [(c["label"], c["size"]) for c in tbl["classes"]]
    # 共轭类（用矩阵键查表，避免 O(h³) 的 allclose 三重循环：Ih 的 120³ 会卡死）
    key2idx = {}
    for i, M in enumerate(mats):
        key2idx.setdefault(_mat_key(M), i)
    classes = []
    used = set()
    for i, Mi in enumerate(mats):
        if i in used:
            continue
        cls = set()
        for Mk in mats:
            P = Mk @ Mi @ np.linalg.inv(Mk)
            j = key2idx.get(_mat_key(P))
            if j is None:
                for jj, Mj in enumerate(mats):  # 兜底
                    if _same(P, Mj):
                        j = jj
                        break
            if j is not None:
                cls.add(int(j))
        cls.add(i)
        for j in cls:
            used.add(j)
        classes.append(sorted(cls))
    # 每个类的签名
    sig = []
    for cls in classes:
        M = mats[cls[0]]
        det = float(np.linalg.det(M))
        sig.append({
            "size": len(cls),
            "proper": det > 0,
            "order": _elem_order(M),
            "sangle": round(_signed_angle(M), 3),
            "aangle": round(abs(_signed_angle(M)), 3),
            "trace": round(float(np.trace(M)), 4),
        })
    # 标签解析：'8C3' → size 8, kind C, order 3；'4S6⁵' → size 4, S, order 6(+)
    import re

    sup = {c: i for i, c in enumerate(_SUPS)}

    def parse_label(lbl):
        m = re.match(r"^(\d*)([CSE])(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)$", lbl)
        if not m:
            return None
        size = int(m.group(1)) if m.group(1) else 1
        kind = m.group(2)
        order = int(m.group(3))
        exp = "".join(str(sup[c]) for c in m.group(4)) if m.group(4) else ""
        return {"size": size, "kind": kind, "order": order, "exp": exp}

    out = {}
    remaining = list(range(len(classes)))
    for lbl, size in want:
        if lbl == "E":
            for i in remaining:
                if sig[i]["size"] == 1:
                    out[i] = lbl
                    remaining.remove(i)
                    break
            continue
        if lbl == "i":
            for i in remaining:
                if sig[i]["size"] == 1 and abs(sig[i]["trace"] + 3) < 1e-6:
                    out[i] = lbl
                    remaining.remove(i)
                    break
            continue
        # 反映面类：教材标签可能是 "σh" / "2σv" / "3σh" / "6σd" —— 带前导数字的写法
        # （Oh 的 3σh 与 6σd、Td 的 6σd）也曾被漏掉：旧代码只认 lbl.startswith("σ")，
        # 于是所有「数字开头的 σ 类」既不进这个分支、又过不了 parse_label 的正则，
        # 最后被兜底标成 "?"，normal_modes 报「找不到代表操作」。
        # 判据：反映面 = det<0 且元素阶为 1，再用 size 区分 σh / σv / σd。
        m_sig = re.match(r"^(\d*)σ", lbl)
        if m_sig:
            size_want = int(m_sig.group(1)) if m_sig.group(1) else 1
            for i in remaining:
                if ((not sig[i]["proper"]) and sig[i]["order"] == 1
                        and sig[i]["size"] == size_want):
                    out[i] = lbl
                    remaining.remove(i)
                    break
            continue
        p = parse_label(lbl)
        if p is None:
            continue
        cand = [i for i in remaining
                if sig[i]["size"] == size
                and sig[i]["proper"] == (p["kind"] in "CT")
                and sig[i]["order"] == p["order"]]
        if cand:
            # 同 size 同阶的类要靠**转角大小**分开：I 的 12C5(72°) 与 12C5²(144°)、
            # Ih 的 12S10(36°) 与 12S10³(108°)。指数 exp 换算成期望 |θ| 再看 sig 的实测值。
            exp = int(p["exp"]) if p["exp"] else 1
            raw = (exp * 360.0 / p["order"]) % 360.0
            want_ang = min(raw, 360.0 - raw)
            same = [i for i in cand if abs(sig[i]["aangle"] - want_ang) < 1e-2]
            if same:
                cand = same
        if not cand:
            continue
        # 角也相同的（Th 的 4C3 与 4C3²）才用有符号转角降序做最后区分
        cand.sort(key=lambda i: -sig[i]["sangle"])
        pick = cand[0]
        out[pick] = lbl
        remaining.remove(pick)
    for i in remaining:  # 兜底：未匹配置 None（自测会报出来）
        M = mats[classes[i][0]]
        out[i] = "?"
    # 展开到每个群元
    full = {}
    for ci, cls in enumerate(classes):
        for j in cls:
            full[j] = out.get(ci)
    return full


# ---------------------------------------------------------------------------
# 群元统计 + 判据树（从矩阵直接判定；point_group 与自测共用同一套逻辑）
# ---------------------------------------------------------------------------

def _mat_census(mats: list) -> tuple:
    """群元 → (有无 i, 真转动轴{dirkey:(方向,阶)}, 镜面法向[], 映轴{...}, n_max, 最高轴列表)。"""
    inv = any(_same(M, -np.eye(3)) for M in mats)
    rot_axes: dict = {}
    planes: list = []
    improper_axes: dict = {}
    for M in mats:
        tr = float(np.trace(M))
        if abs(tr - 3.0) < 1e-6:
            continue
        if abs(tr + 3.0) < 1e-6:
            continue
        if float(np.linalg.det(M)) > 0:
            ax = _axis_of(M)
            if ax is None:
                continue
            k = _dirkey(ax)
            n = _elem_order(M)
            if k not in rot_axes or rot_axes[k][1] < n:
                rot_axes[k] = (ax, n)
        else:
            if abs(tr - 1.0) < 1e-6:
                planes.append(_plane_normal(M))
            else:
                ax = _axis_of_improper(M)
                improper_axes[_dirkey(ax)] = (ax, _elem_order(M))
    nmax = max([v[1] for v in rot_axes.values()], default=1)
    times_nmax = [v for v in rot_axes.values() if v[1] == nmax]
    return inv, rot_axes, planes, improper_axes, nmax, times_nmax


def _tree(h: int, inv: bool, rot_axes: dict, planes: list, improper_axes: dict) -> tuple:
    """群论判据树：由群元统计定 Schoenflies 记号。→ (名字 或 None, 中文推理链)。

    判据顺序（教材口径）：
      无真转动：有 i → Ci；有 σ → Cs；都没有 → C1。
      多条 n≥3 的真转动轴 → 立方/二十面体分支（按阶 + 有无 i / σ 区分 T/Td/Th/O/Oh/I/Ih）。
      否则走轴向族：取主轴的轴次 n，数垂直 C2 → 有 σh / σv / σd → C_n, C_nv, C_nh, S_2n,
      D_n, D_nh, D_nd。**每个候选主轴都必须让「群阶 = 名字推出的阶」自洽**，否则否决。
    """
    n_hi = len([1 for v in rot_axes.values() if v[1] >= 3])
    if max([v[1] for v in rot_axes.values()], default=1) <= 1:
        if inv:
            return "Ci", "除 E 外只剩反演 i → Ci。"
        if planes:
            return "Cs", "除 E 外只有一个反映面、无真转动 → Cs。"
        return "C1", "除 E 外无任何对称操作 → C1。"
    if h in (12, 24, 48, 60, 120) and n_hi >= 2:
        if h == 12:
            return "T", "12 阶、4 条 C3 轴、无 σ 无 i → T。"
        if h == 24:
            if inv:
                return "Th", "24 阶、含 i（3 个 σh） → Th。"
            if planes:
                return "Td", "24 阶、含 6σd 与 6S4、无 i → Td。"
            return "O", "24 阶、4C3+3C4+6C2、无 σ 无 i → O。"
        if h == 48:
            return "Oh", "48 阶、含 i 与 3σh+6σd → Oh。"
        if h == 60:
            return "I", "60 阶、6 条 C5 轴、无 σ 无 i → I。"
        return "Ih", "120 阶、含 i 与 15σ → Ih。"
    cands = sorted([(v[1], v[0]) for v in rot_axes.values()], key=lambda t: -t[0])
    for n_princ, axis in cands:
        ua = _unit(axis)
        nperp = len([1 for _kk, (v, _nn) in rot_axes.items()
                     if abs(float(np.dot(_unit(v), ua))) < 1e-3])
        sh = None
        for nrm in planes:
            if abs(abs(float(np.dot(_unit(nrm), ua))) - 1.0) < 1e-3:
                sh = nrm
                break
        svert = [nrm for nrm in planes if abs(float(np.dot(_unit(nrm), ua))) < 1e-3]
        if nperp == n_princ:
            if sh is not None and h == 4 * n_princ:
                return "D%dh" % n_princ, (
                    "最高轴次 C%d → 有 %d 个垂直 C2 → 有 σh（⊥主轴）→ D%dh。"
                    % (n_princ, nperp, n_princ))
            if sh is None and svert and h == 4 * n_princ:
                return "D%dd" % n_princ, (
                    "最高轴次 C%d → 有 %d 个垂直 C2 → 无 σh 但有 %d 个 σd → D%dd。"
                    % (n_princ, nperp, len(svert), n_princ))
            if sh is None and not svert and h == 2 * n_princ:
                return "D%d" % n_princ, (
                    "最高轴次 C%d → 有 %d 个垂直 C2 → 无 σh/σd → D%d。"
                    % (n_princ, nperp, n_princ))
        else:
            if sh is not None and h == 2 * n_princ:
                return "C%dh" % n_princ, (
                    "最高轴次 C%d → 无垂直 C2 → 有 σh（⊥主轴）→ C%dh。"
                    % (n_princ, n_princ))
            if sh is None and svert and h == 2 * n_princ:
                return "C%dv" % n_princ, (
                    "最高轴次 C%d → 无垂直 C2 → 有 %d 个含轴镜面 σv → C%dv。"
                    % (n_princ, len(svert), n_princ))
            if not svert and h == 2 * n_princ:
                smax = max([v[1] for v in improper_axes.values()], default=1)
                if smax == 2 * n_princ:
                    return "S%d" % (2 * n_princ), (
                        "最高轴次 C%d → 无垂直 C2、无镜面 → 但有 S%d 映轴 → S%d。"
                        % (n_princ, 2 * n_princ, 2 * n_princ))
            if not svert and h == n_princ:
                return "C%d" % n_princ, (
                    "最高轴次 C%d → 无垂直 C2、无镜面、无映轴 → C%d。"
                    % (n_princ, n_princ))
    return None, ""


def _classify_mats(mats: list) -> dict:
    """由一群 3×3 矩阵判定点群（判据树本体，不碰坐标）。

    自测直接拿 `_build_group(名字)` 的矩阵喂进来做全量反验：大群（O / T / I / Ih）
    要上百个原子的一般位置轨道才存在，靠构造分子去验证代价不可接受。
    """
    h = len(mats)
    inv, rot_axes, planes, improper_axes, nmax, times_nmax = _mat_census(mats)
    name, reason = _tree(h, inv, rot_axes, planes, improper_axes)
    return {
        "ok": name is not None,
        "point_group": name,
        "reason": reason,
        "order": h,
        "inv": inv,
        "nmax": nmax,
        "n_axes_nmax": len(times_nmax),
        "n_planes": len(planes),
        "rot_axes": rot_axes,
        "planes": planes,
        "improper_axes": improper_axes,
        "zaxis": (times_nmax[0][0] if len(times_nmax) == 1 else None),
    }


# ---------------------------------------------------------------------------
# S1 point_group
# ---------------------------------------------------------------------------

_INTERNATIONAL = {
    "C1": "1", "Cs": "m", "Ci": "1̄", "C2": "2", "C3": "3", "C4": "4", "C5": "5", "C6": "6",
    "C2v": "mm2", "C3v": "3m", "C4v": "4mm", "C5v": "5m", "C6v": "6mm",
    "C2h": "2/m", "C3h": "6̄",
    "D2": "222", "D3": "32", "D4": "422", "D5": "52", "D6": "622",
    "D2h": "mmm", "D3h": "6̄m2", "D4h": "4/mmm", "D5h": "10̄m2", "D6h": "6/mmm",
    "D2d": "4̄2m", "D3d": "3̄m", "D4d": "4̄2m", "D5d": "1̄0m2",
    "S4": "4̄", "S6": "3̄",
    "T": "23", "Td": "4̄3m", "Th": "m3̄", "O": "432", "Oh": "m3̄m",
    "I": "532", "Ih": "m3̄5̄",
    "C∞v": "∞m", "D∞h": "∞/mm",
}


def _frame(v: np.ndarray, zaxis) -> np.ndarray:
    """把世界方向的向量表达到 (x,y,z) 局部坐标系里。"""
    if zaxis is None:
        return v
    z = _unit(zaxis)
    trial = _X if abs(float(np.dot(_X, z))) < 0.9 else _Y
    x = _unit(np.cross(trial, z))
    y = np.cross(z, x)
    return np.array([np.dot(v, x), np.dot(v, y), np.dot(v, z)])


def _axis_name(v, zaxis) -> str:
    """方向的教科书写法：(z) / (x) / (y) / (1,1,0) 之类。"""
    a = _frame(np.asarray(v, float), zaxis)
    a = a / max(1e-12, float(np.linalg.norm(a)))
    lab = {0: "x", 1: "y", 2: "z"}
    for i in range(3):
        if abs(abs(a[i]) - 1.0) < 1e-3:
            return lab[i]
    parts = []
    for c in a:
        if abs(c) < 1e-3:
            parts.append("0")
        elif abs(abs(c) - round(abs(c), 2)) < 1e-9:
            parts.append(("%.2f" % c).rstrip("0").rstrip("."))
        else:
            parts.append("%.2f" % c)
    return "(" + ",".join(parts) + ")"


def _plane_name(normal, zaxis) -> str:
    """平面按「所含两条轴」命名：法向 z → (xy)。"""
    a = _frame(np.asarray(normal, float), zaxis)
    a = a / max(1e-12, float(np.linalg.norm(a)))
    idx = [0, 1, 2]
    dropped = None
    for i in idx:
        if abs(abs(a[i]) - 1.0) < 1e-3:
            dropped = i
            break
    if dropped is not None:
        return "(" + "".join("xyz"[i] for i in idx if i != dropped) + ")"
    return "(%s)" % _axis_name(normal, zaxis).strip("()")


def _op_name(M, zaxis) -> str:
    """群元的教科书名字。"""
    tr = float(np.trace(M))
    det = float(np.linalg.det(M))
    if abs(tr - 3.0) < 1e-6:
        return "E"
    if abs(tr + 3.0) < 1e-6:
        return "i"
    if det > 0:
        n = _elem_order(M)
        axis = _axis_name(_axis_of(M), zaxis)
        if n <= 1:
            return "E"
        k = int(round(360.0 / _rot_angle(M)))
        return "C%d(%s)" % (n, axis)
    if abs(tr - 1.0) < 1e-6:
        nrm = _plane_normal(M)
        return "σ%s" % _plane_name(nrm, zaxis)
    n = _elem_order(M)
    axis = _axis_name(_axis_of_improper(M), zaxis)
    return "S%d(%s)" % (n, axis)


def _axis_of_improper(M) -> np.ndarray:
    """映轴的轴：S_n 的轴是 σ_h 的法向，即 M 的特征值 -1 之外……用 M² 的真转轴。"""
    P = M @ M
    ax = _axis_of(P)
    if ax is not None:
        return ax
    w, vecs = np.linalg.eig(M)
    return _unit(np.real(vecs[:, 0]))


def point_group(atoms, tol: float = 0.15) -> dict:
    """S1：由原子坐标真算分子点群。

    参数
    ----
    atoms : [{"sym":"H","xyz":[x,y,z]}, ...]  单位 Å，函数自己平移到质心
    tol   : 原子匹配容差（Å），默认 0.15（=15 pm）

    返回
    ----
    {ok, error, point_group, order, operations, elements, international,
     detected, detected_ops, n_atoms, tol, cubic, linear, steps, notes}
    """
    syms, coords, perr = _parse_atoms(atoms)
    if perr:
        return _err(perr)
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        return _err("tol 必须是数字（单位 Å，默认 0.15）")
    if not (tol > 0):
        return _err("tol 必须大于 0（单位 Å，默认 0.15）")
    n_at = len(syms)
    raw = coords.copy()
    coords = coords - coords.mean(axis=0)  # 平移到几何质心

    # 元素分块
    blocks = []
    for s in sorted(set(syms)):
        blocks.append(np.array([i for i, t in enumerate(syms) if t == s], dtype=int))

    steps: list = []
    notes: list = []
    steps.append("共 %d 个原子，先把坐标平移到几何质心（任一对称操作都固定质心，故它必为不动点）。" % n_at)

    if n_at == 1:
        return {
            "ok": True, "error": None, "point_group": "K_h", "order": "∞",
            "operations": ["E", "C∞(任意轴)", "i", "σ(任意面)"],
            "elements": ["全部旋转轴、全部镜面、反演中心"],
            "international": None, "detected": {}, "detected_ops": {},
            "n_atoms": 1, "tol": tol, "tol_pm": tol * PM_PER_A,
            "cubic": True, "linear": True,
            "steps": steps + ["只有一个原子 → 全旋转群 K_h（教材不讨论，此处仅作标注）。"],
            "notes": ["单原子的分子点群是 K_h，本工具的查表/约化接口不提供该群。"],
        }

    # 线形快速分支
    #
    # 陷阱（曾导致 SF6/CH4/BF3/丙二烯/CH2Cl2 全部误判成 C∞v）：
    # coords 是**相对质心**的坐标，而质心往往正好落在中心原子上（SF6 的 S、CH4 的 C、
    # CH2Cl2 的 C 都在质心 → coords 里有一项是零向量）。若拿 coords[0] 当叉积基准，
    # 一旦它是零向量，cross(0, v) ≡ 0，任何"中心原子 + 外围原子"的分子都会被判成共线。
    # 因此基准必须取**模长最大**的向量；零向量不参与共线判定（中心原子本来就在轴上）。
    if n_at >= 2:
        norms = [float(np.linalg.norm(v)) for v in coords]
        imax = int(np.argmax(norms))
        ref = coords[imax]
        collinear = norms[imax] > 1e-9
        if collinear:
            for v, nv in zip(coords, norms):
                if nv < 1e-9:
                    continue
                if float(np.linalg.norm(np.cross(ref, v))) > tol * max(1.0, norms[imax]):
                    collinear = False
                    break
        if collinear:
            # 质心在原点，最远原子的方向即主轴方向（符号不影响 C∞v/D∞h 的判定）
            zaxis = _unit(ref)
            # 有无反演中心：每个坐标 ±t 成对（同元素）
            has_i = True
            for i, v in enumerate(coords):
                t = float(np.dot(v, zaxis))
                mirror_pt = -t
                found = False
                for j, w in enumerate(coords):
                    if syms[j] != syms[i]:
                        continue
                    if abs(float(np.dot(w, zaxis)) - mirror_pt) < tol and \
                            float(np.linalg.norm(w - np.dot(w, zaxis) * zaxis)) < tol:
                        found = True
                        break
                if not found:
                    has_i = False
                    break
            pg = "D∞h" if has_i else "C∞v"
            steps.append("全部原子共线（偏差 < %.3g Å）→ 走线形分子分支：主轴是 C∞ 轴。" % tol)
            steps.append("逐个原子检查 ±t 是否都有同元素原子 → %s反演中心。" %
                         ("有" if has_i else "无"))
            steps.append("线形 + %s → %s。" % ("有 i" if has_i else "无反演中心", pg))
            ops = ["E", "C∞", "σv(∞个)"] + (["i", "S∞", "C2(∞个)"] if has_i else [])
            return {
                "ok": True, "error": None, "point_group": pg, "order": "∞",
                "operations": ops,
                "elements": (["C∞ 轴（所有原子共线）", "无穷多个含轴镜面 σv"] +
                             (["反演中心 i", "S∞ 轴"] if has_i else [])),
                "international": _INTERNATIONAL.get(pg), "detected": {},
                "detected_ops": {}, "n_atoms": n_at, "tol": tol, "tol_pm": tol * PM_PER_A,
                "cubic": False, "linear": True, "steps": steps,
                "notes": ["线形分子按 C∞v / D∞h 的 Σ/Π 计数法处理（见 normal_modes）。"],
            }

    # 常规：枚举 + 闭包
    det = _detect_ops(coords, blocks, tol)
    steps.append(
        "取候选方向 %d 个（原子径向 / 两原子和 / 两原子差 / 叉积 / 惯量主轴），"
        "对每个方向试算 C2–C8 与 σ、对已确认的转动轴再试 S3–S8，共 %d 次操作验证。"
        % (det["dirs"], det["tested"]))
    steps.append("判定判据：R·r_i 后每个原子在 %.3g Å 内找到**同元素**原子，且一一对应"
                 "（二部图完美匹配）。" % tol)

    gens = list(det["gens"])  # 已 Kabsch 精修 + 去重，可直接闭包
    steps.append("检出：真转动轴 %d 条、镜面 %d 个、映轴 %d 个、%s反演中心。"
                 % (det["n_rot_axes"], det["n_planes"], det["n_improper"],
                    "有" if det["inversion"] else "无"))
    mats = _close_group(gens, coords, blocks, tol)
    h = len(mats)
    steps.append("群元乘积闭包后共 %d 个对称操作 → 群阶 h = %d。" % (h, h))

    # 结构统计
    inv, rot_axes, planes, improper_axes, nmax, times_nmax = _mat_census(mats)
    steps.append("最高真转动轴次 n_max = %d（这样的轴 %d 条）。" % (nmax, len(times_nmax)))

    # ---- 判据树 ----
    name, reason = _tree(h, inv, rot_axes, planes, improper_axes)
    if name is None:
        return _err(
            "识别到 %d 个对称操作（最高 C%d 轴 %d 条、镜面 %d 个、%s），但落不进标准判据树："
            "可能是 tol=%g Å 偏小漏判了对称元素，或坐标精度太差。请把 tol 调到 0.2–0.3 再试。"
            % (h, nmax, len(times_nmax), len(planes), "有 i" if inv else "无 i", tol))
    steps.append(reason)

    # 名字、操作清单、元素清单
    zaxis = None
    if nmax >= 2:
        zaxis = times_nmax[0][0] if len(times_nmax) >= 1 else None
        if zaxis is not None and len(times_nmax) > 1:
            # 立方群没有唯一主轴，保持世界坐标
            zaxis = None
    operations, detected, detected_ops = [], {}, {}
    for M in mats:
        nm = _op_name(M, zaxis)
        if nm in detected_ops:
            continue
        operations.append(nm)
        detected[nm] = round(_op_dev(coords, M, blocks, tol), 6)
        detected_ops[nm] = [[round(float(x), 6) for x in row] for row in M]
    order_idx = {n: i for i, n in enumerate(operations)}
    operations.sort(key=lambda s: (0 if s == "E" else 1 if s.startswith("C") else
                                   2 if s.startswith("σ") else 3 if s == "i" else 4))
    elements, elements_detail = _elements_from_mats(mats, zaxis)

    steps.append("写全群元 %d 个，最大重合偏差 %.3g Å（越小越准）。"
                 % (len(operations), max(detected.values()) if detected else 0.0))

    return {
        "ok": True, "error": None,
        "point_group": name,
        "order": h,
        "operations": operations,
        "elements": elements,
        "elements_detail": elements_detail,
        "international": _INTERNATIONAL.get(name),
        "detected": detected,
        "detected_ops": detected_ops,
        "n_atoms": n_at,
        "tol": tol,
        "tol_pm": round(tol * PM_PER_A, 3),
        "cubic": name in _CUBIC,
        "linear": False,
        "steps": steps,
        "notes": notes,
    }


def _elements_from_mats(mats: list, zaxis) -> tuple:
    """群元 → 对称元素清单（中文可直接抄）。"""
    detail = []
    axes: dict = {}
    planes: dict = {}
    has_i = False
    s_types: dict = {}
    for M in mats:
        tr = float(np.trace(M))
        det = float(np.linalg.det(M))
        if abs(tr - 3.0) < 1e-6:
            continue
        if abs(tr + 3.0) < 1e-6:
            has_i = True
            continue
        if det > 0:
            ax = _axis_of(M)
            if ax is None:
                continue
            k = _dirkey(ax)
            n = _elem_order(M)
            if k not in axes or axes[k][1] < n:
                axes[k] = (ax, n)
        elif abs(tr - 1.0) < 1e-6:
            nrm = _plane_normal(M)
            planes[_dirkey(nrm)] = nrm
        else:
            ax = _axis_of_improper(M)
            k = _dirkey(ax)
            n = _elem_order(M)
            # 与 axes 一致地存 (轴方向, 阶次)：第 1174 行按 -t[1][1] 排序并解包成 (ax, n)。
            # 之前这里只存了 n，导致映轴存在的分子（苯、SF6 等）直接 TypeError。
            if k not in s_types or s_types[k][1] < n:
                s_types[k] = (ax, n)
    out = []
    for _k, (ax, n) in sorted(axes.items(), key=lambda t: -t[1][1]):
        desc = "C%d 轴（%s）" % (n, _axis_name(ax, zaxis))
        detail.append({"kind": "Cn", "n": n, "count": 1, "desc": desc})
        out.append(desc)
    for _k, (ax, n) in sorted(s_types.items(), key=lambda t: -t[1][1]):
        desc = "S%d 映轴（%s）" % (n, _axis_name(ax, zaxis))
        detail.append({"kind": "Sn", "n": n, "count": 1, "desc": desc})
        out.append(desc)
    if planes:
        h_norm = [nrm for nrm in planes.values()
                  if zaxis is not None and abs(abs(float(np.dot(_unit(nrm), _unit(zaxis)))) - 1.0) < 1e-3]
        if h_norm and len(planes) > 1:
            out.append("σh 面 1 个（⊥主轴）")
            detail.append({"kind": "sigma_h", "count": 1, "desc": "σh 面（⊥主轴）"})
            v_norm = [nrm for nrm in planes.values()
                      if zaxis is not None and abs(float(np.dot(_unit(nrm), _unit(zaxis)))) < 1e-3]
            if v_norm:
                out.append("竖直镜面 σ %d 个（含主轴）" % len(v_norm))
                detail.append({"kind": "sigma_v", "count": len(v_norm),
                               "desc": "竖直镜面 σ %d 个（含主轴）" % len(v_norm)})
        else:
            out.append("σ 镜面 %d 个" % len(planes))
            detail.append({"kind": "sigma", "count": len(planes),
                           "desc": "σ 镜面 %d 个" % len(planes)})
        # 主轴上的镜面：有几个 σv / σd（用是否含垂直 C2 轴区分）
        # 这里只给数量，具体归属由 D_nh / D_nd 记号体现
    if has_i:
        out.append("反演中心 i")
        detail.append({"kind": "i", "count": 1, "desc": "反演中心 i"})
    if not out:
        out.append("只有 E")
        detail.append({"kind": "E", "count": 1, "desc": "只有恒等操作"})
    return out, detail


# ---------------------------------------------------------------------------
# S2 symmetry_elements
# ---------------------------------------------------------------------------

def symmetry_elements(point_group_name: str) -> dict:
    """S2：点群 → 阶、类、对称元素、生成元。纯查群结构，不需要坐标。"""
    name = (point_group_name or "").strip()
    fam, _n = _parse_group_name(name)
    tbl = _load_data().get(name)
    if fam is None and not tbl:
        return _err("认不出点群名 %r。可用记号：C1 Cs Ci C2..C6 C2v..C6v C2h C3h D2..D6 "
                    "D2h D3h D4h D5h D6h D2d D3d D4d D5d S4 S6 S8 T Td Th O Oh I Ih C∞v D∞h。"
                    % point_group_name)
    steps = ["按 Schoenflies 记号 %s 分解群结构。" % name]
    if tbl is None:
        return _err("数据表里还没有 %s 的特征标表（本工具只收教材附表覆盖的点群）。" % name)
    infinite = bool(tbl.get("infinite")) or tbl.get("order") == "∞"
    classes = []
    for c in tbl["classes"]:
        classes.append({"name": c["label"], "size": c["size"], "sample_ops": c.get("ops", [])})
    steps.append("群阶 h = %s；共 %d 个共轭类（不可约表示数 = 类数 = %d）。"
                 % (tbl["order"], len(classes), len(classes)))
    mats = _build_group(name)
    elements, detail = ([], [])
    if mats:
        elements, detail = _elements_from_mats([m for m, _l in mats], None)
        steps.append("对称元素：%s。" % "、".join(elements[:6]))
    gens = _generators(name)
    steps.append("生成元取 %s 即可生成全部群元。" % "、".join(gens))
    return {
        "ok": True, "error": None,
        "point_group": name,
        "order": tbl["order"],
        "classes": classes,
        "elements": elements,
        "elements_detail": detail,
        "generators": gens,
        "infinite": infinite,
        "steps": steps,
    }


def _generators(name: str) -> list:
    fam, n = _parse_group_name(name)
    if fam in ("C1",):
        return ["E"]
    if fam == "Ci":
        return ["i"]
    if fam == "Cs":
        return ["σh"]
    if fam == "C":
        return ["C%d" % n]
    if fam == "Cv":
        return ["C%d" % n, "σv"]
    if fam == "Ch":
        return ["C%d" % n, "σh"]
    if fam == "S":
        return ["S%d" % n]
    if fam == "D":
        return ["C%d" % n, "C2'"]
    if fam == "Dh":
        return ["C%d" % n, "C2'", "σh"]
    if fam == "Dd":
        return ["S%d" % (2 * n), "C2'"]
    if fam == "T":
        return ["C3", "C2"]
    if fam == "Td":
        return ["C3", "S4"]
    if fam == "Th":
        return ["C3", "C2", "i"]
    if fam == "O":
        return ["C4", "C3"]
    if fam == "Oh":
        return ["C4", "C3", "i"]
    if fam == "I":
        return ["C5", "C3"]
    if fam == "Ih":
        return ["C5", "C3", "i"]
    if fam == "Cinf":
        return ["C∞"]
    if fam == "Dinf":
        return ["C∞", "i"]
    return []


# ---------------------------------------------------------------------------
# S3 character_table
# ---------------------------------------------------------------------------

def _tbl(name: str):
    return _load_data().get(name)


def character_table(point_group_name: str) -> dict:
    """S3：点群 → 特征标表（类、不可约表示、基函数）。"""
    name = (point_group_name or "").strip()
    tbl = _tbl(name)
    if tbl is None:
        fam, _n = _parse_group_name(name)
        if fam is None:
            return _err("认不出点群名 %r。可用：C1 Cs Ci C2 C3 C4 C5 C6 C2v C3v C4v C5v C6v "
                        "C2h C3h D2 D3 D4 D5 D6 D2h D3h D4h D5h D6h D2d D3d D4d D5d "
                        "S4 S6 S8 T Td Th O Oh I Ih C∞v D∞h。" % point_group_name)
        return _err("数据表里没有 %s 的特征标表（只收教材附表覆盖的点群；不编数值）。" % name)
    classes = [{"label": c["label"], "size": c["size"], "ops": c.get("ops", [])}
               for c in tbl["classes"]]
    irreps = [{"name": r["name"], "chars": list(r["chars"]),
               "merged": bool(r.get("merged", False)),
               "dim": int(round(float(r["chars"][0]))) if isinstance(r["chars"][0], (int, float)) else None}
              for r in tbl["irreps"]]
    basis = {k: list(v) for k, v in tbl["basis"].items()}
    rev = {}
    for irr, funcs in basis.items():
        for f in funcs:
            for single in _split_funcs(f):
                rev.setdefault(single, irr)
    n_cls = len(classes)
    n_irr = len(irreps)
    infinite = bool(tbl.get("infinite")) or tbl.get("order") == "∞"
    steps = ["%s：群阶 h = %s，共 %d 个共轭类。" % (name, tbl["order"], n_cls)]
    merged = [r["name"] for r in irreps if r["merged"]]
    if merged:
        steps.append("本表 %d 个不可约表示：教材把 %d 对共轭一维复表示合并写成 E，"
                     "故表示数 %d < 类数 %d 是**正常的**（合并约定）。"
                     % (n_irr, len(merged), n_irr, n_cls))
        steps.append("合并约定下的判据只有三条：全对称表示 Σ g_c·χ = h；不同表示正交；"
                     "非 merged Σ g_c·χ_i² = h、merged = 2h。")
    else:
        steps.append("本表 %d 个不可约表示 = 类数 %d（未使用合并约定，全部为实表示）。"
                     % (n_irr, n_cls))
    if not infinite:
        dims = [r["dim"] for r in irreps]
        steps.append("维数账：Σ dim² = %s = h = %s（merged 的 E 计作两个一维复表示）。"
                     % (" + ".join("%d²" % d for d in dims), tbl["order"]))
    steps.append("类顺序（reduce_representation 的 chars 必须按此次序）：%s。"
                 % " ".join(c["label"] for c in classes))
    if merged:
        steps.append("约化这两个 E 时用 n=(1/(2h))Σ g_c·χ_i·χ_r：%s。" % "、".join(merged))
    if infinite:
        steps.append("无限群用简化表：类为 E / 2C_φ / ∞σv（D∞h 再加 i / 2S_φ / 2C2），"
                     "非整数字符写作 2cos(kφ)；不做有限群约化。")
    return {
        "ok": True, "error": None,
        "point_group": name,
        "order": tbl["order"],
        "classes": classes,
        "irreps": irreps,
        "basis": basis,
        "basis_reverse": rev,
        "n_classes": n_cls,
        "n_irreps": n_irr,
        "infinite": infinite,
        "note": tbl.get("note"),
        "conventions": _META.get("conventions", {}),
        "steps": steps,
    }


def _split_funcs(f: str) -> list:
    """把 "(x,y)" 拆成 ["x","y"]；单个函数原样返回。"""
    s = str(f).strip()
    if s.startswith("(") and s.endswith(")"):
        return [t.strip() for t in s[1:-1].split(",") if t.strip()]
    return [s]


# ---------------------------------------------------------------------------
# S4 reduce_representation
# ---------------------------------------------------------------------------

def reduce_representation(point_group_name: str, chars) -> dict:
    """S4：约化可约表示 n_i = (1/h)Σ_c g_c·χ_i(c)·χ_r(c)。"""
    name = (point_group_name or "").strip()
    tbl = _tbl(name)
    if tbl is None:
        return _err("数据表里没有 %s 的特征标表；先用 character_table 查可用点群。" % name)
    infinite = bool(tbl.get("infinite")) or tbl.get("order") == "∞"
    if infinite:
        return _err("%s 是无限群，其特征标随转角连续变化，不能用有限的 (1/h)Σ 公式约化。"
                    "线形分子的 3N 分解请直接用 normal_modes(point_group, n, is_linear=True)，"
                    "它走 Σ/Π 计数分支。" % name)
    classes = tbl["classes"]
    try:
        vals = [float(c) for c in chars]
    except (TypeError, ValueError):
        return _err("chars 必须是数字列表，按类顺序给出（例：C2v 的 Γ_3N = [9,-1,3,1]）")
    if len(vals) != len(classes):
        return _err("chars 长度 %d 与 %s 的类数 %d 不一致。类顺序为：%s"
                    % (len(vals), name, len(classes),
                       " ".join("%s(%s)" % (c["label"], c["size"]) for c in classes)))
    h = int(tbl["order"])
    total_dim = int(round(vals[0]))
    if abs(vals[0] - round(vals[0])) > 1e-6:
        return _err("第一个类是 E，其 χ(E) 必须是整数（= 表示的维数），收到 %g" % vals[0])
    decomp, steps = [], []
    steps.append("约化公式：n_i = (1/h)·Σ_c g_c·χ_i(c)·χ_r(c)，h = %d。" % h)
    steps.append("χ_r 按类顺序：%s。"
                 % " ".join("%s=%g" % (c["label"], v) for c, v in zip(classes, vals)))
    for r in tbl["irreps"]:
        irr = list(r["chars"])
        merged = bool(r.get("merged", False))
        terms, acc = [], 0.0
        for c, chi, ch in zip(classes, irr, vals):
            g = c["size"]
            acc += g * float(chi) * ch
            terms.append("%g×(%g)×(%g)" % (g, chi, ch))
        denom = 2 * h if merged else h
        n = acc / denom
        n_int = int(round(n))
        decomp.append({
            "irrep": r["name"],
            "count": n_int,
            "contribution": round(float(n), 6),
            "dim": int(round(float(irr[0]))),
            "merged_convention": merged,
        })
        steps.append("n(%s) = (1/%d)[%s] = %s%s"
                     % (r["name"], denom, " + ".join(terms),
                        ("%g" % n) if abs(n - n_int) > 1e-6 else str(n_int),
                        "   ← 教材合并约定，除以 2h" if merged else ""))
    # 还原自校验
    recon = [0.0] * len(classes)
    for d, r in zip(decomp, tbl["irreps"]):
        for i, chi in enumerate(r["chars"]):
            recon[i] += d["count"] * float(chi)
    max_dev = max((abs(a - b) for a, b in zip(recon, vals)), default=0.0)
    dim_sum = sum(d["count"] * d["dim"] for d in decomp)
    sum_check = max_dev < 1e-6
    dim_check = dim_sum == total_dim
    steps.append("还原校验：Σ_i n_i·χ_i = [%s]，与原可约表示的最大偏差 %.3g → %s。"
                 % (", ".join("%g" % x for x in recon), max_dev,
                    "通过" if sum_check else "不通过"))
    steps.append("维数校验：Σ_i n_i·dim_i = %d，而 χ_r(E) = %d → %s。"
                 % (dim_sum, total_dim, "通过" if dim_check else "不通过"))
    steps.append("结论：Γ = %s。" % " + ".join(
        "%d%s" % (d["count"], d["irrep"]) for d in decomp if d["count"]))
    return {
        "ok": True, "error": None,
        "point_group": name,
        "order": h,
        "chars": vals,
        "decomposition": decomp,
        "gamma": " + ".join("%d%s" % (d["count"], d["irrep"]) for d in decomp if d["count"]),
        "sum_check": bool(sum_check),
        "dimensions_check": bool(dim_check),
        "reconstructed": [round(x, 6) for x in recon],
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# S5 normal_modes
# ---------------------------------------------------------------------------

def _readable(pt: str) -> str:
    return {"Cinf": "C∞v", "Dinf": "D∞h"}.get(pt, pt)


def _linear_modes(pt: str, n_at: int) -> dict:
    """线形分子的 3N 分解（Σ/Π 计数法）。"""
    if n_at < 2:
        return _err("线形分子至少 2 个原子")
    n_vib = 3 * n_at - 5
    if pt == "C∞v":
        gvib = {"Σ⁺": n_at - 1, "Π": n_at - 1}
        ir = ["Σ⁺", "Π"]
        raman = ["Σ⁺", "Π", "Δ"]
        silent = []
        steps = [
            "线形（C∞v）：每个原子都在轴上，对任意转角 φ 都不动 → χ_3N(C_φ)=N(1+2cosφ)，"
            "χ_3N(σv)=N。",
            "与 Σ⁺(χ=1) 和 Π(χ=2cosφ) 对照得 Γ_3N = %dΣ⁺ ⊕ %dΠ（%d = 3×%d）。"
            % (n_at, n_at, 3 * n_at, n_at),
            "Γ_trans = Σ⁺(z) ⊕ Π(x,y) 共 3 维；Γ_rot = Π 共 2 维。",
            "Γ_vib = Γ_3N − Γ_trans − Γ_rot = %dΣ⁺ ⊕ %dΠ，模式数 %d + 2×%d = %d = 3N−5。"
            % (n_at - 1, n_at - 1, n_at - 1, n_at - 1, n_vib),
            "C∞v 选律：ΔΛ=0,±1 → Σ⁺ 与 Π 都红外活性；Raman 允许 Σ⁺、Π、Δ，故全模都拉曼活性。",
        ]
        return {
            "ok": True, "error": None, "point_group": pt, "n_atoms": n_at,
            "total_dof": 3 * n_at, "linear": True,
            "gamma_3N": {"Σ⁺": n_at, "Π": n_at}, "gamma_trans": {"Σ⁺": 1, "Π": 1},
            "gamma_rot": {"Π": 1}, "gamma_vib": gvib, "n_vib": n_vib,
            "ir_active": ir, "raman_active": raman, "silent": silent,
            "steps": steps,
            "notes": ["线形分子的 Π、Δ 是二重简并表示（每个计数算 2 个自由度）。"],
        }
    # D∞h
    n_c = 1 if n_at % 2 else 0
    a = (n_at - n_c) // 2
    c = (n_at + n_c) // 2
    gvib = {"Σg⁺": a, "Σu⁺": c - 1, "Πg": a - 1, "Πu": c - 1}
    gvib = {k: v for k, v in gvib.items() if v}
    n_modes = a + (c - 1) + 2 * (a - 1) + 2 * (c - 1)
    steps = [
        "线形 + 有 i（D∞h）：轴上原子被任意转动固定，被 σv 固定，但被 i 与 S_φ 移走"
        "（除位于中心的那个）。",
        "取 n_c = %d（%s），χ_3N: E=%d, C_φ=%d+%dcosφ, σv=%d, i=%d, S_φ=%d+2cosφ, C2'=%d。"
        % (n_c, "奇数个原子故有中心原子" if n_c else "偶数个原子故无中心原子",
           3 * n_at, n_at, 2 * n_at, n_at, -3 * n_c, -n_c, -n_c),
        "解得 Γ_3N = %dΣg⁺ ⊕ %dΣu⁺ ⊕ %dΠg ⊕ %dΠu。"
        % ((n_at - n_c) // 2, (n_at + n_c) // 2, (n_at - n_c) // 2, (n_at + n_c) // 2),
        "Γ_trans = Σu⁺(z) ⊕ Πu(x,y)；Γ_rot = Πg。",
        "Γ_vib = %s，模式数 %d = 3N−5。"
        % (" + ".join("%d%s" % (v, k) for k, v in gvib.items()), n_modes),
        "D∞h 互斥规则：g 模只 Raman 活性、u 模只红外活性；Σg⁺、Πg、Δg 拉曼；Σu⁺、Πu 红外。",
    ]
    return {
        "ok": True, "error": None, "point_group": pt, "n_atoms": n_at,
        "total_dof": 3 * n_at, "linear": True,
        "gamma_3N": {"Σg⁺": (n_at - n_c) // 2, "Σu⁺": (n_at + n_c) // 2,
                     "Πg": (n_at - n_c) // 2, "Πu": (n_at + n_c) // 2},
        "gamma_trans": {"Σu⁺": 1, "Πu": 1}, "gamma_rot": {"Πg": 1},
        "gamma_vib": gvib, "n_vib": n_modes,
        "ir_active": ["Σu⁺", "Πu"], "raman_active": ["Σg⁺", "Πg", "Δg"],
        "silent": [k for k in gvib if k not in ("Σu⁺", "Πu", "Σg⁺", "Πg")],
        "steps": steps,
        "notes": ["中心原子数按 N 的奇偶取 1/0（线形且有心时必然如此）。"],
    }


def _mk_atoms(groups):
    """groups: [(元素符号, [(x,y,z), ...]), ...] → [{sym, xyz}]。"""
    out = []
    for sym, pts in groups:
        for p in pts:
            out.append({"sym": sym, "xyz": [float(p[0]), float(p[1]), float(p[2])]})
    return out


_TET = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
_OCT = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
_EDGE = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
         (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
         (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]


def _cube_pts(vecs, d):
    n = math.sqrt(sum(c * c for c in vecs[0]))
    return [(v[0] * d / n, v[1] * d / n, v[2] * d / n) for v in vecs]


def _mk_ring(n, r, sym, z=0.0):
    return (sym, [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z)
                  for i in range(n)])


# (点群, 原子数) → 教科书标准构型。命中就跳过轨道搜索与反验 —— 这几套本身就是该点群的
# 典型结构（SF6 / CH4 / 八面体+棱心 / 立方体顶点 / 苯 / H2O）。
# 为什么需要：反验要跑完整 point_group 判定，Oh 单次就几秒；早期实现试满 16 个组合，
# 冷启动可到 30 s 以上，在 MCP 层表现为「调用超时 / 无响应」。
_PRESET_CANONICAL: dict = {}


def _init_presets() -> None:
    """只放「慢且坐标无歧义」的高对称立方/二十面体构型。

    刻意**不放** C2v / C3v / D6h 这些低对称群：它们本身就快（0–3 ms），
    而预置坐标会悄悄改变 B1/B2（乃至 A1/A2）的归属 —— 同一个分子换个平面取向，
    Γ_vib 里 B1 与 B2 就互换，标准答案随之对不上。这类群交给轨道搜索，
    其结果与「分子放在哪个平面」无关，反而更稳。
    """
    _PRESET_CANONICAL.update({
        ("Oh", 7): _mk_atoms([("M", [(0.0, 0.0, 0.0)]), ("X", _cube_pts(_OCT, 1.56))]),
        ("Oh", 6): _mk_atoms([("X", _cube_pts(_OCT, 1.56))]),
        ("Oh", 13): _mk_atoms([("M", [(0.0, 0.0, 0.0)]), ("X", _cube_pts(_EDGE, 2.2))]),
        ("Oh", 19): _mk_atoms([("M", [(0.0, 0.0, 0.0)]), ("X", _cube_pts(_OCT, 1.56)),
                               ("Y", _cube_pts(_EDGE, 2.2))]),
        ("Td", 5): _mk_atoms([("M", [(0.0, 0.0, 0.0)]), ("X", _cube_pts(_TET, 1.09))]),
        ("Td", 4): _mk_atoms([("X", _cube_pts(_TET, 1.09))]),
    })


_init_presets()


@lru_cache(maxsize=512)
def _canonical_atoms(pt: str, n_at: int):
    """给 (点群, 原子数) 造一套规范构型（用于在无坐标时求 3N 分解）。

    规则：从点群自身的群元生成若干条轨道（不动点 / 主轴上的点 / 镜面内的点 /
    一般位置），挑出总和等于 n_at 的轨道组合，并**用 point_group 反验**它真的是该点群。
    这是「无坐标时的规范模型」，不是猜——反验不过就报错。

    性能注记（实测踩过）：反验要跑完整点群判定，Oh 单次就几秒；早期实现对 Oh 试满
    16 个组合，冷启动能到 30 s 以上，在 MCP 里表现为「调用超时/无响应」。
    因此这里对教科书上最常见的几种构型**直接查表**（它们本身就是标准结构，不需要反验），
    其余组合的反验上限也从 16 收到 8。
    """
    preset = _PRESET_CANONICAL.get((pt, n_at))
    if preset is not None:
        return preset
    mats = [m for m, _l in _build_group(pt)]
    if not mats:
        return None
    seeds = [np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 0.0, 1.31]),
             np.array([0.0, 0.0, 2.17]),
             np.array([1.37, 0.0, 0.0]),
             np.array([0.0, 1.53, 0.0]),
             np.array([1.19, 0.0, 0.63]),
             np.array([0.0, 1.41, 0.71]),
             np.array([1.07, 0.61, 0.83]),
             np.array([0.93, 0.37, 1.61]),
             np.array([1.71, 0.83, 1.29]),
             np.array([0.71, 1.31, 1.73]),
             np.array([2.03, 1.17, 0.53]),
             # 体对角线 (1,1,1) 方向：在 Td / T / Th 里是 C3 轴，由此得 4 重轨道，
             # 正是 CH4（1 个中心 + 4 个顶点 = n_at 5）需要的那个组合。缺了它就凑不出 Td 的 5 原子。
             np.array([0.92, 0.92, 0.92]),
             np.array([1.28, 1.28, 1.28])]
    # 再从群自身的对称元素取种子：轴向点（落在 C_n 轴上）、镜面内的点。
    # 二十面体群的 C5 轴不沿 x/y/z，"盲种子"只会给出 30/60/120 的轨道，
    # 拿不到 C60 之外的 12 顶点那类小轨道（Ih 的 12 原子构型就凑不出来）。
    seen_seed = set()
    axis_seeds, plane_seeds = [], []
    for M in mats:
        tr = float(np.trace(M))
        if abs(tr - 3.0) < 1e-6:
            continue
        if float(np.linalg.det(M)) > 0:
            ax = _axis_of(M)
            if ax is not None:
                k = _dirkey(ax)
                if k not in seen_seed:
                    seen_seed.add(k)
                    axis_seeds.append(_unit(ax) * 1.37)
                    axis_seeds.append(_unit(ax) * 1.93)
        elif abs(tr - 1.0) < 1e-6:
            nrm = _plane_normal(M)
            if nrm is None:
                continue
            k = _dirkey(nrm)
            if k in seen_seed:
                continue
            seen_seed.add(k)
            t = np.cross(nrm, _X if abs(float(np.dot(nrm, _X))) < 0.9 else _Y)
            plane_seeds.append(_unit(t) * 1.49)
    seeds = seeds + axis_seeds[:24] + plane_seeds[:24]
    orbits = []
    for s in seeds:
        pts = []
        for M in mats:
            w = M @ s
            if not any(np.allclose(w, q, atol=1e-6) for q in pts):
                pts.append(w)
        orbits.append((len(pts), np.array(pts)))
        if len(pts) > n_at:
            continue
    # 搜索：轨道条数最少优先，其次优先「先在轴上/镜面内」的小轨道
    best = None
    order = sorted(range(len(orbits)), key=lambda i: (orbits[i][0], i))
    lim = len(orbits)

    def rec(start, left, picked):
        nonlocal best
        if left == 0:
            if best is None or len(picked) < len(best):
                best = list(picked)
            return
        if best is not None and len(picked) + 1 >= len(best):
            return
        for i in range(start, lim):
            sz = orbits[i][0]
            if sz > left:
                continue
            picked.append(i)
            rec(i + 1, left - sz, picked)
            picked.pop()

    rec(0, n_at, [])
    if best is None:
        return None
    # 由「小轨道优先」的组合试起，逐个用 point_group 反验
    combos = []

    def rec2(start, left, picked):
        if left == 0:
            combos.append(list(picked))
            return
        for i in range(start, lim):
            sz = orbits[i][0]
            if sz > left:
                continue
            picked.append(i)
            rec2(i + 1, left - sz, picked)
            picked.pop()
        return

    rec2(0, n_at, [])
    combos.sort(key=lambda c: (len(c),
                               -max(orbits[i][0] for i in c) if c else 0,
                               sum(orbits[i][0] * i for i in c)))
    def _try(combo):
        pts = np.vstack([orbits[i][1] for i in combo])
        if pts.shape[0] != n_at:
            return None
        # 同一轨道给同一个元素符号（便于匹配检验）
        syms = []
        for k, i in enumerate(combo):
            syms += [chr(ord("A") + k % 26)] * orbits[i][0]
        atoms = [{"sym": s, "xyz": list(map(float, p))} for s, p in zip(syms, pts)]
        r = point_group(atoms)
        return atoms if (r.get("ok") and r.get("point_group") == pt) else None

    # 第一轮：小轨道/小索引优先（对轴上有原子的群最快，如 H2O、SF6）。
    for combo in combos[:16]:
        got = _try(combo)
        if got is not None:
            return got
    # 第二轮（兜底）：**反向**取组合，即优先用末尾那些「3-D 一般位置」种子。
    # 为什么需要：C1 / Ci / Cs / C3 / C2h 这类群只有 size-1、size-2 的小轨道，
    # 而小索引种子恰好是共线/共面的（(0,0,1.31)、(0,0,2.17)、(1.37,0,0)…），
    # 前 16 个组合拼出来全是线形或平面退化构型，反验必然被判成 C∞v / D∞h / 更高群，
    # 于是「最平凡的群反而算不出」。反向取就能拿到三维错开的原子。
    for combo in combos[-16:]:
        got = _try(combo)
        if got is not None:
            return got
    return None


def normal_modes(point_group_name: str, n_atoms: int, is_linear: bool = False, atoms=None) -> dict:
    """S5：3N 自由度分解 Γ_3N = Γ_trans + Γ_rot + Γ_vib，并给 IR/Raman 活性。"""
    name = (point_group_name or "").strip()
    fam, _n = _parse_group_name(name)
    if fam is None:
        return _err("认不出点群名 %r。" % point_group_name)
    try:
        n_at = int(n_atoms)
    except (TypeError, ValueError):
        return _err("n_atoms 必须是正整数")
    if n_at < 1:
        return _err("n_atoms 必须 ≥ 1")
    canon = name
    if fam == "Cinf":
        canon = "C∞v"
    elif fam == "Dinf":
        canon = "D∞h"
    if is_linear or fam in ("Cinf", "Dinf"):
        if fam in ("Cinf", "Dinf"):
            return _linear_modes(canon, n_at)
        return _err("%s 不是线形点群，却收到 is_linear=True。" % name)
    tbl = _tbl(canon)
    if tbl is None:
        return _err("数据表里没有 %s 的特征标表。" % canon)
    if tbl.get("order") == "∞":
        return _linear_modes(canon, n_at)
    h = int(tbl["order"])
    classes = tbl["classes"]
    # 规范构型（可传 atoms 覆盖）
    if atoms is not None:
        syms, coords, perr = _parse_atoms(atoms)
        if perr:
            return _err(perr)
        if len(syms) != n_at:
            return _err("atoms 的原子数 %d 与 n_atoms=%d 不一致" % (len(syms), n_at))
        coords = coords - coords.mean(axis=0)
        blocks = []
        for s in sorted(set(syms)):
            blocks.append(np.array([i for i, t in enumerate(syms) if t == s], dtype=int))
        used = "atoms 实坐标"
    else:
        built = _canonical_atoms(canon, n_at)
        if built is None:
            return _err("找不到「%d 个原子且反验为 %s」的规范构型。两种原因："
                        "① 轨道大小凑不上；② 该群在这么少原子上**几何上就不存在**——"
                        "例如任意 3 个原子必共面因而必含 σ（不可能是 C1/Ci/C3），"
                        "2 个原子必线形（不可能是 Ci），1 个离轴原子 + n 个落在 C2' 轴上的原子"
                        "必共面因而必成 D_nh（凑不出 D_n/D_nd）。"
                        "请改用 normal_modes(%r, %d, atoms=[...]) 直接给真实分子坐标（最稳）。"
                        % (n_at, canon, canon, n_at))
        syms, coords, _ = _parse_atoms(built)
        coords = coords - coords.mean(axis=0)
        blocks = []
        for s in sorted(set(syms)):
            blocks.append(np.array([i for i, t in enumerate(syms) if t == s], dtype=int))
        used = "规范构型（%d 原子，已用 point_group 反验为 %s）" % (n_at, canon)

    # 每个类的代表操作：从真实群元里按类标签取第一个
    rep = {}
    for M, lbl in _build_group(canon):
        if lbl not in rep:
            rep[lbl] = M
    chars_3N, unmoved, detail = [], [], []
    missing = [c["label"] for c in classes if c["label"] not in rep]
    if missing:
        return _err("内部错误：%s 的类 %s 在群元表里找不到代表操作。" % (canon, "、".join(missing)))
    for c in classes:
        M = rep[c["label"]]
        moved = sum(1 for p in coords if float(np.linalg.norm(M @ p - p)) > 1e-6)
        nm = n_at - moved
        tr = float(np.trace(M))
        chars_3N.append(nm * tr)
        unmoved.append(nm)
        detail.append({"class": c["label"], "size": c["size"], "unmoved": nm,
                       "trace": round(tr, 6), "chi": nm * tr})
    red = reduce_representation(canon, chars_3N)
    if not red.get("ok"):
        return red
    # 平动/转动表示：**每个不可约表示贡献 1 个**，不是"函数出现几次"。
    # 原因：x、y、z 同属一个三维表示时（Oh 的 T1u、Td 的 T2、D3h 的 E'），
    # 三者合起来是 1 个 T1u（重数 1，维数 3），正好覆盖 3 个平动自由度。
    # 旧代码按函数个数累加 → trans[Oh]={T1u:3}，做减法时 1−3=−2，
    # 于是所有含简并表示的群（Oh/Td/D3h/D4h 的高维表示）全部报「分量不足」。
    trans, rot = {}, {}
    for irr, funcs in tbl["basis"].items():
        singles = [s for f in funcs for s in _split_funcs(f)]
        if any(s in ("x", "y", "z") for s in singles):
            trans[irr] = 1
        if any(s in ("Rx", "Ry", "Rz") for s in singles):
            rot[irr] = 1
    used_irr: dict = {}
    for d in red["decomposition"]:
        used_irr[d["irrep"]] = d["count"]
    vib = dict(used_irr)
    for src in (trans, rot):
        for irr, cnt in src.items():
            vib[irr] = vib.get(irr, 0) - cnt
    if any(v < 0 for v in vib.values()):
        return _err("内部错误：%s 的 Γ_3N 里平动/转动分量不足（数据或规范构型有问题）。" % canon)
    vib = {k: v for k, v in vib.items() if v}
    n_vib = sum(v * int(round(float(dict((d["irrep"], d["dim"]) for d in red["decomposition"])[k])))
                for k, v in vib.items())
    # 选律
    raman_set, ir_set = set(), set()
    for irr, funcs in tbl["basis"].items():
        for f in funcs:
            for single in _split_funcs(f):
                if single in ("x", "y", "z"):
                    ir_set.add(irr)
                else:
                    raman_set.add(irr)
    for irr in ("Rx", "Ry", "Rz"):
        pass
    ir_set = set(trans.keys())
    ir_active = sorted([k for k in vib if k in ir_set],
                       key=lambda s: (len(s), s))
    raman_active = sorted([k for k in vib if k in raman_set], key=lambda s: (len(s), s))
    silent = sorted([k for k in vib if k not in ir_set and k not in raman_set],
                    key=lambda s: (len(s), s))
    dims = dict((d["irrep"], d["dim"]) for d in red["decomposition"])
    steps = [
        "3N = 3×%d = %d 个自由度；%s，h = %d。" % (n_at, 3 * n_at, used, h),
        "按类求 χ_3N：不动的原子贡献 tr(R)（真转动 1+2cosθ、σ 为 1、i 为 −3、S_n 为 2cosθ−1），"
        "动了的原子贡献 0。",
    ]
    for d in detail:
        steps.append("  %s：不动原子 %d 个 × tr=%g → χ_3N=%g" %
                     (d["class"], d["unmoved"], d["trace"], d["chi"]))
    steps.append("Γ_3N = %s。" % red["gamma"])
    steps.append("Γ_trans = %s（x,y,z 所在表示）。" % _fmt_rep(trans))
    steps.append("Γ_rot = %s（Rx,Ry,Rz 所在表示）。" % _fmt_rep(rot))
    steps.append("Γ_vib = Γ_3N − Γ_trans − Γ_rot = %s，共 %d 个振动模式（3N−6）。"
                 % (_fmt_rep(vib), n_vib))
    steps.append("选律：IR 活性 ⇔ 与 x,y,z 同不可约表示 → %s；"
                 "Raman 活性 ⇔ 与二次函数同表示 → %s；都不行的是静默模 %s。"
                 % (", ".join(ir_active) or "无", ", ".join(raman_active) or "无",
                    ", ".join(silent) or "无"))
    return {
        "ok": True, "error": None,
        "point_group": canon,
        "n_atoms": n_at,
        "total_dof": 3 * n_at,
        "linear": False,
        "config_source": used,
        "chi_3N": chars_3N,
        "class_detail": detail,
        "gamma_3N": {k: v for k, v in used_irr.items() if v},
        "gamma_trans": trans,
        "gamma_rot": rot,
        "gamma_vib": vib,
        "n_vib": n_vib,
        "ir_active": ir_active,
        "raman_active": raman_active,
        "silent": silent,
        "reduce_check": {"sum_check": red["sum_check"], "dimensions_check": red["dimensions_check"]},
        "steps": steps,
        "notes": ["非线形分子振动模式数 = 3N−6；同时有 IR 与 Raman 活性的模必须落在"
                  "与 x,y,z 及二次函数都同表示的不可约表示上。"],
    }


def _fmt_rep(d: dict) -> str:
    if not d:
        return "0"
    return " + ".join("%d%s" % (v, k) for k, v in d.items() if v) or "0"


# ---------------------------------------------------------------------------
# S6 selection_rules
# ---------------------------------------------------------------------------

def selection_rules(point_group_name: str) -> dict:
    """S6：偶极矩、旋光性、允许的 IR/Raman 基函数。"""
    name = (point_group_name or "").strip()
    fam, n = _parse_group_name(name)
    tbl = _tbl(name)
    if fam is None and tbl is None:
        return _err("认不出点群名 %r。" % point_group_name)
    steps = []
    # 偶极矩：只有 C1 / Cs / Cn / Cnv 有永久偶极矩
    if fam == "C1":
        dip = True
        direction = "任意方向（C1 无对称性约束）"
    elif fam == "Cs":
        dip = True
        direction = "在镜面内（垂直于镜面法向的分量被禁，故 μ 落在镜面内）"
    elif fam in ("C", "Cv"):
        dip = True
        direction = "沿 C%d 主轴（z）" % n
    elif fam == "Cinf":
        dip = True
        direction = "沿 C∞ 主轴（z）"
    else:
        dip = False
        if fam in ("Ci",):
            direction = "无（有反演中心 i，μ 必须为 0）"
        elif fam in ("D", "Dh", "Dd", "Dinf"):
            direction = "无（D 族有 ≥2 条不平行的 C2 轴，任何 μ 都被迫为 0）"
        elif fam in _CUBIC:
            direction = "无（T/O/I 系有 ≥2 条不平行的 C_n 轴，μ 必须为 0）"
        elif fam == "S":
            direction = "无（S%d 群只含映轴，无极性方向）" % n
        elif fam in ("Ch",):
            direction = "无（C%dh 含 σh，μ ⊥ 主轴的分量被禁、沿主轴分量又被 C%d 转走）" % (n, n)
        else:
            direction = "无"
    steps.append("偶极矩判据：永久偶极矩只可能出现在 C1、Cs、C_n、C_nv（及 C∞v）；"
                 "有 i、或属于 D/T/O/I 系、或含 σh 的群，μ 必须为 0。")
    steps.append("%s → %s。" % (name, "有偶极矩：" + direction if dip else "无偶极矩：" + direction))

    # 旋光性：无 i、无 σ（即只含真转动）才可能有手性
    has_i, has_s = False, False
    mats = _build_group(name)
    for M, _l in mats:
        tr = float(np.trace(M))
        if abs(tr + 3.0) < 1e-6:
            has_i = True
        elif float(np.linalg.det(M)) < 0:
            has_s = True
    chiral = (not has_i) and (not has_s)
    if chiral:
        why = "群内既无反演 i 也无任何反映面/映轴（只含真转动）→ 可以手性，可能有旋光性"
    elif has_i and has_s:
        why = "既有 i 又有 σ/S_n → 分子与镜像重合，必无旋光性"
    elif has_i:
        why = "有反演中心 i → 分子与镜像重合，必无旋光性"
    else:
        why = "含反映面 σ 或映轴 S_n → 分子与镜像重合，必无旋光性"
    steps.append("旋光性判据：只有「无 i 且无 σ」的点群才可能有手性。%s → %s。" % (name, why))

    ir_basis, raman_basis = [], []
    if tbl is not None:
        for irr, funcs in tbl["basis"].items():
            for f in funcs:
                for single in _split_funcs(f):
                    if single in ("x", "y", "z"):
                        ir_basis.append("%s: %s" % (irr, single))
                    elif single not in ("Rx", "Ry", "Rz"):
                        raman_basis.append("%s: %s" % (irr, single))
        steps.append("IR 允许的基函数（与 x,y,z 同表示）：%s。"
                     % "、".join(ir_basis))
        steps.append("Raman 允许的基函数（与二次函数同表示）：%s。"
                     % "、".join(raman_basis))
    else:
        steps.append("数据表里没有 %s，无法列出基函数。" % name)
    return {
        "ok": True, "error": None,
        "point_group": name,
        "dipole": {"has": dip, "direction": direction},
        "chiral": {"has": chiral, "reason": why},
        "allowed_ir_basis": ir_basis,
        "allowed_raman_basis": raman_basis,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# S7 compose_operations
# ---------------------------------------------------------------------------

_OP_RE = None


def _parse_op(txt: str, pt: str):
    """把操作名解析成 3×3 矩阵；认不出返回 (None, 错误串)。"""
    import re

    s = (txt or "").strip().replace(" ", "").replace("－", "-")
    if not s:
        return None, "操作名为空"
    if s in ("E", "e", "1"):
        return np.eye(3), None
    if s in ("i", "I"):
        return -np.eye(3), None
    sg = s.replace("σ", "s").replace("Σ", "s").replace("sigma", "s")
    if sg.lower().startswith("s") and not re.match(r"^[sS]\d", sg):
        # σ 形式：σh / σv / σd / σxy / σ(xz) / σv(xz)
        body = sg[1:]
        body = re.sub(r"^(h|v|d|H|V|D)", "", body)
        body = body.strip("()")
        if body == "":
            # 无方向：按点群的规范约定取第一个
            mats = _build_group(pt)
            key = "σh" if sg[1:2].lower() == "h" else ("σd" if sg[1:2].lower() == "d" else "σv")
            for M, lbl in mats:
                if lbl and str(lbl).startswith(key):
                    return M, None
            for M, lbl in mats:
                if lbl and str(lbl).startswith("σ"):
                    return M, None
            return None, "点群 %s 里没有 σ 面" % pt
        normal = _dir_from_plane(body)
        if normal is None:
            return None, "认不出镜面写法 %r：请写 σxy / σ(xz) / σh / σv / σd" % txt
        return _refl(normal), None
    m = re.match(r"^[Cc]([1-9]\d*)((\^)(\d+))?(\(([^)]*)\))?$", s)
    if m:
        n = int(m.group(1))
        k = int(m.group(4)) if m.group(4) else 1
        d = m.group(6)
        axis = _dir_from_axis(d, pt) if d else _default_axis(pt)
        if axis is None:
            return None, "认不出 C%d 的轴写法 %r：请写 C2(z) / C3(1,1,1) 之类" % (n, d)
        return _rot(axis, 360.0 * k / n), None
    m = re.match(r"^[Ss]([1-9]\d*)((\^)(\d+))?(\(([^)]*)\))?$", s)
    if m:
        n = int(m.group(1))
        k = int(m.group(4)) if m.group(4) else 1
        d = m.group(6)
        axis = _dir_from_axis(d, pt) if d else _default_axis(pt)
        if axis is None:
            return None, "认不出 S%d 的轴写法 %r" % (n, d)
        return _srot(axis, 360.0 * k / n), None
    return None, "认不出操作 %r：可用 E / i / Cn(z) / Cn^k(z) / Sn(z) / σxy / σh / σv / σd" % txt


def _default_axis(pt: str):
    mats = _build_group(pt)
    best, bestn = None, 0
    for M, _l in mats:
        if float(np.linalg.det(M)) < 0 or abs(float(np.trace(M)) - 3.0) < 1e-9:
            continue
        n = _elem_order(M)
        if n > bestn:
            best, bestn = _axis_of(M), n
    return best if best is not None else _Z


def _dir_from_axis(txt: str, pt: str):
    if txt is None:
        return _default_axis(pt)
    t = txt.strip().lower().replace(" ", "")
    if t in ("x", "y", "z"):
        return {"x": _X, "y": _Y, "z": _Z}[t]
    t = t.strip("()")
    try:
        parts = [float(v) for v in t.split(",")]
    except ValueError:
        return None
    if len(parts) != 3:
        return None
    v = np.array(parts)
    if float(np.linalg.norm(v)) < 1e-9:
        return None
    return v


def _dir_from_plane(txt: str):
    """镜面写法（xy / xz / yz）→ 法向。"""
    t = txt.strip().lower().replace(" ", "").strip("()")
    if t in ("xy", "yx"):
        return _Z
    if t in ("xz", "zx"):
        return _Y
    if t in ("yz", "zy"):
        return _X
    try:
        parts = [float(v) for v in t.split(",")]
    except ValueError:
        return None
    if len(parts) != 3:
        return None
    v = np.array(parts)
    if float(np.linalg.norm(v)) < 1e-9:
        return None
    return v


def compose_operations(point_group_name: str, op1: str, op2: str) -> dict:
    """S7：操作乘法 op1·op2（约定：先做 op2 再做 op1，即矩阵 M1·M2）。"""
    name = (point_group_name or "").strip()
    fam, _n = _parse_group_name(name)
    if fam is None:
        return _err("认不出点群名 %r。" % point_group_name)
    mats = _build_group(name)
    if not mats:
        return _err("点群 %s 的群元表不可用（数据缺失）。" % name)
    M1, e1 = _parse_op(op1, name)
    if e1:
        return _err("op1：%s" % e1)
    M2, e2 = _parse_op(op2, name)
    if e2:
        return _err("op2：%s" % e2)
    in1 = any(_same(M1, M) for M, _l in mats)
    in2 = any(_same(M2, M) for M, _l in mats)
    if not in1:
        return _err("%s 不是 %s 的对称操作（它不在该点群内）。请先用 symmetry_elements 看该群的群元。"
                    % (op1, name))
    if not in2:
        return _err("%s 不是 %s 的对称操作（它不在该点群内）。" % (op2, name))
    P = M1 @ M2
    det = float(np.linalg.det(P))
    eq = None
    for M, lbl in mats:
        if _same(P, M):
            eq = _op_name(M, None)
            break
    steps = [
        "约定：%s·%s 表示**先做 %s、再做 %s**，对应矩阵相乘 M(%s)·M(%s)。"
        % (op1, op2, op2, op1, op1, op2),
        "M(%s) = %s" % (op1, _fmt_mat(M1)),
        "M(%s) = %s" % (op2, _fmt_mat(M2)),
        "乘积 = %s，det = %+.4g（%s）。" % (_fmt_mat(P), det,
                                            "+1 真转动" if det > 0 else "−1 含反演或反映"),
        ("乘积等于该群里的操作 %s。" % eq) if eq else "乘积不在该点群内（内部异常，请报告）。",
    ]
    return {
        "ok": True, "error": None,
        "point_group": name,
        "op1": op1, "op2": op2,
        "product_name": eq,
        "matrix": [[round(float(x), 9) for x in row] for row in P],
        "determinant": round(det, 9),
        "is_proper": det > 0,
        "steps": steps,
    }


def _fmt_mat(M) -> str:
    rows = []
    for r in M:
        rows.append("[" + ", ".join("%.4g" % float(x) for x in r) + "]")
    return "[" + "; ".join(rows) + "]"


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _mk(sym_xyz):
    return [{"sym": s, "xyz": list(map(float, v))} for s, v in sym_xyz]


def _benzene():
    at = []
    for k in range(6):
        a = math.radians(60.0 * k)
        at.append(("C", [1.397 * math.cos(a), 1.397 * math.sin(a), 0.0]))
    for k in range(6):
        a = math.radians(60.0 * k)
        at.append(("H", [2.481 * math.cos(a), 2.481 * math.sin(a), 0.0]))
    return _mk(at)


def _naphthalene():
    """两个稠合正六边形 + 8 个径向 H，整体 D2h。"""
    at = []
    c1 = np.array([0.0, 0.0])
    c2 = np.array([2.432, 0.0])
    r = 1.404
    ring1 = [c1 + r * np.array([math.cos(math.radians(30 + 60 * k)),
                                math.sin(math.radians(30 + 60 * k))]) for k in range(6)]
    ring2 = [c2 + r * np.array([math.cos(math.radians(30 + 60 * k)),
                                math.sin(math.radians(30 + 60 * k))]) for k in range(6)]
    shared = [p for p in ring1 if abs(p[0] - 1.216) < 1e-2]
    carbons = ring1 + [p for p in ring2 if not any(np.allclose(p, q, atol=1e-2) for q in shared)]
    for p in carbons:
        at.append(("C", [p[0], p[1], 0.0]))
    for p in carbons:
        if any(np.allclose(p, q, atol=1e-2) for q in shared):
            continue
        center = c1 if p[0] < 1.216 else c2
        d = p - center
        d = d / np.linalg.norm(d)
        h = p + 1.08 * d
        at.append(("H", [h[0], h[1], 0.0]))
    return _mk(at)


def _chair():
    at = []
    for k in range(6):
        a = math.radians(60.0 * k)
        at.append(("C", [1.46 * math.cos(a), 1.46 * math.sin(a), 0.25 if k % 2 == 0 else -0.25]))
    return _mk(at)


def _validate_tables() -> tuple:
    """特征标表的三条硬校验（合并约定下唯一成立的口径）。

    ① 全对称表示各字符为 1，且 Σ_c g_c·χ_全对称(c) = h；
    ② 任意两个不同表示 Σ_c g_c·χ_i(c)·χ_j(c) = 0；
    ③ 自身归一：非 merged 表示 Σ_c g_c·χ_i² = h，merged 表示 = 2h。
    附加：Σ 类大小 = h；维数账 Σ dim²（merged 的 E 按两个一维复表示计） = h。

    注意：**不校验「不可约表示数 = 类数」**——教材把一对共轭一维复表示合并写成一个
    E 之后，表示数必然小于类数（S4: 4类3表示、S8: 8类5表示、T: 4类3表示…），
    强行要求相等会把 10 个点群全部误报。
    """
    msgs = []
    bad = 0
    nt = 0
    for name, tbl in sorted(_load_data().items()):
        order = tbl.get("order")
        classes = tbl["classes"]
        irreps = tbl["irreps"]
        n_cls = len(classes)
        if order == "∞":
            continue
        nt += 1
        h = int(order)
        sizes = [c["size"] for c in classes]
        if sum(sizes) != h:
            msgs.append("  ✗ %s：Σ 类大小 = %d ≠ h = %d" % (name, sum(sizes), h))
            bad += 1
        # ① 全对称表示
        tsm = [r for r in irreps if all(abs(float(c) - 1.0) < 1e-9 for c in r["chars"])]
        if len(tsm) != 1:
            msgs.append("  ✗ %s：全对称表示个数 = %d（应恰为 1）" % (name, len(tsm)))
            bad += 1
        else:
            s = sum(sizes[k] * float(tsm[0]["chars"][k]) for k in range(n_cls))
            if abs(s - h) > 5e-3:
                msgs.append("  ✗ %s：Σ g_c·χ_全对称 = %.4f ≠ h = %d" % (name, s, h))
                bad += 1
        # 维数账
        acc = 0
        for r in irreps:
            d = int(round(float(r["chars"][0])))
            acc += d if r.get("merged") else d * d
        if acc != h:
            msgs.append("  ✗ %s：维数账 Σ dim² = %d ≠ h = %d（merged 的 E 按两个一维复表示计）"
                        % (name, acc, h))
            bad += 1
        # ②③ 正交性 + 归一化
        for i in range(len(irreps)):
            mi = 2 if irreps[i].get("merged") else 1
            for j in range(i, len(irreps)):
                s = sum(sizes[k] * float(irreps[i]["chars"][k]) * float(irreps[j]["chars"][k])
                        for k in range(n_cls))
                if i == j:
                    want = h * mi
                    if abs(s - want) > 5e-3:
                        msgs.append("  ✗ %s：<χ(%s)|χ(%s)> = %.4f，应 = %d（%s）"
                                    % (name, irreps[i]["name"], irreps[j]["name"], s, want,
                                       "合并约定 2h" if mi == 2 else "h"))
                        bad += 1
                elif abs(s) > 5e-3:
                    msgs.append("  ✗ %s：<χ(%s)|χ(%s)> = %.4f ≠ 0"
                                % (name, irreps[i]["name"], irreps[j]["name"], s))
                    bad += 1
    msgs.insert(0, "三条判据校验：%d 个有限点群，%d 处不合格。" % (nt, bad))
    return msgs, bad


def _selftest_group_closure() -> list:
    """群元与表的类标签必须一一对上；且 point_group(规范构型) 必须还原该点名。"""
    msgs = []
    bad = 0
    for name, tbl in sorted(_load_data().items()):
        if tbl.get("order") == "∞":
            continue
        mats = _build_group(name)
        want = sorted([c["label"] for c in tbl["classes"]])
        got = sorted([str(l) for _m, l in mats])
        if len(mats) != int(tbl["order"]):
            msgs.append("  ✗ %s: 群元 %d 个 ≠ h = %s" % (name, len(mats), tbl["order"]))
            bad += 1
        if want != got:
            # 两者语义不同，不能直接比列表：
            #   表里的 classes 是**每个类一条**（"2C3" 一条代表 2 个操作），
            #   群元列表是**每个操作一条**（C3 的 E/C3/C3² → 'E','2C3','2C3'）。
            # 因此只比「标签集合」是否一致。早期版本比列表，导致 13 个含合并类的
            # 点群（C3/C3v/C4/C4v/C5/C5v/C6/C6v/D2d/D3/D3d/D3h/D4）全部误报。
            miss = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            if miss or extra:
                msgs.append("  ✗ %s: 类标签不匹配（表有未匹：%s；群多出：%s）"
                            % (name, miss, extra))
                bad += 1
    msgs.insert(0, "群元/类标签自校验：%d 处不合格。" % bad)
    return msgs, bad


def _func_kind(f: str) -> str:
    """基函数分类：vec（x,y,z）/ rot（Rx,Ry,Rz）/ quad（二次齐次）/ other（三次及以上）。"""
    s = str(f).strip()
    if s in ("x", "y", "z"):
        return "vec"
    if s in ("Rx", "Ry", "Rz"):
        return "rot"
    if "²" in s or s in ("xy", "yz", "xz"):
        return "quad"
    return "other"


def _selftest_basis() -> list:
    """用真实群元算 Γ_vec / Γ_rot / Γ_quad，约化后必须与表里的 basis 一致。"""
    msgs = []
    bad = 0
    nchk = 0
    for name, tbl in sorted(_load_data().items()):
        if tbl.get("order") == "∞":
            continue
        classes = tbl["classes"]
        mats = _build_group(name)
        rep = {}
        for M, lbl in mats:
            rep.setdefault(lbl, M)
        if not all(c["label"] in rep for c in classes):
            msgs.append("  ✗ %s：缺类代表，跳过基函数校验" % name)
            bad += 1
            continue
        vec, rot, quad = [], [], []
        for c in classes:
            M = rep[c["label"]]
            vec.append(float(np.trace(M)))
            det = float(np.linalg.det(M))
            rot.append(det * float(np.trace(M)))
            quad.append((float(np.trace(M)) ** 2 + float(np.trace(M @ M))) / 2.0)
        basis = tbl["basis"]
        want = {"vec": set(), "rot": set(), "quad": set(), "other": set()}
        for irr, funcs in basis.items():
            for f in funcs:
                for single in _split_funcs(f):
                    want[_func_kind(single)].add(irr)
        for tag, chars in (("vec", vec), ("rot", rot), ("quad", quad)):
            got = _reduce_names(tbl, chars)
            nchk += 1
            if got is None:
                msgs.append("  ✗ %s：Γ_%s 约化不出整数" % (name, tag))
                bad += 1
                continue
            # got 是「按重数展开」的表，want 是集合：只比较落在哪些不可约表示上
            if sorted(set(got)) != sorted(want[tag]):
                msgs.append("  ✗ %s：Γ_%s 落在 %s，表里 basis 说 %s"
                            % (name, tag, sorted(set(got)), sorted(want[tag])))
                bad += 1
    msgs.insert(0, "基函数一致性自校验：%d 项检验，%d 处不合格。" % (nchk, bad))
    return msgs, bad


def _reduce_names(tbl, chars):
    h = int(tbl["order"])
    out = []
    for r in tbl["irreps"]:
        acc = sum(c["size"] * float(r["chars"][i]) * chars[i]
                  for i, c in enumerate(tbl["classes"]))
        denom = 2 * h if r.get("merged") else h
        n = acc / denom
        if abs(n - round(n)) > 1e-4:
            return None
        k = int(round(n))
        out += [r["name"]] * k
    return sorted(out)


def _main() -> int:
    fails = []

    def chk(label, got, want):
        ok = got == want
        print(("  [%s] %-46s %s%s" % ("OK " if ok else "FAIL", label, got,
                                     "" if ok else "   ← 期望 %s" % (want,))))
        if not ok:
            fails.append("%s: %s != %s" % (label, got, want))

    print("=" * 78)
    print("symmetry.py 自测（教材口径：《结构化学基础》第 5 版）")
    print("=" * 78)

    print("\n[1] 点群判定（真算：枚举矩阵 → 集合匹配 → 群闭包 → 判据树）")
    cases = [
        ("CH2Cl2", _mk([("C", [0, 0, 0]),
                        ("H", [0.904, 0, 0.609]), ("H", [-0.904, 0, 0.609]),
                        ("Cl", [0, 1.458, -1.002]), ("Cl", [0, -1.458, -1.002])]), "C2v"),
        ("苯 C6H6", _benzene(), "D6h"),
        ("SF6", _mk([("S", [0, 0, 0])] +
                    [("F", list(map(float, v))) for v in
                     ([1.56, 0, 0], [-1.56, 0, 0], [0, 1.56, 0], [0, -1.56, 0],
                      [0, 0, 1.56], [0, 0, -1.56])]), "Oh"),
        ("丙二烯 CH2=C=CH2", _mk([("C", [0, 0, 0]), ("C", [0, 0, 1.31]), ("C", [0, 0, -1.31]),
                                   ("H", [0.926, 0, 1.866]), ("H", [-0.926, 0, 1.866]),
                                   ("H", [0, 0.926, -1.866]), ("H", [0, -0.926, -1.866])]), "D2d"),
        ("椅式环己烷（C 骨架）", _chair(), "D3d"),
        ("萘 C10H8", _naphthalene(), "D2h"),
        ("BF3", _mk([("B", [0, 0, 0])] +
                    [("F", [1.31 * math.cos(math.radians(120 * k)),
                            1.31 * math.sin(math.radians(120 * k)), 0.0]) for k in range(3)]), "D3h"),
        ("CH4", _mk([("C", [0, 0, 0])] +
                    [("H", [0.629 * a, 0.629 * b, 0.629 * c]) for a, b, c in
                     ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]), "Td"),
        ("H2O2（二面角 111°）", _mk([("O", [0.7375, 0, 0]), ("O", [-0.7375, 0, 0]),
                                      ("H", [0.9025, 0.530, 0.771]),
                                      ("H", [-0.9025, 0.530, -0.771])]), "C2"),
        ("HCN", _mk([("H", [0, 0, -1.065]), ("C", [0, 0, 0]), ("N", [0, 0, 1.156])]), "C∞v"),
        ("CO2", _mk([("O", [0, 0, -1.16]), ("C", [0, 0, 0]), ("O", [0, 0, 1.16])]), "D∞h"),
        ("反式二氯乙烯", _mk([("C", [0.67, -0.11, 0]), ("C", [-0.67, 0.11, 0]),
                              ("H", [1.23, -1.02, 0]), ("H", [-1.23, 1.02, 0]),
                              ("Cl", [0.06, 1.63, 0]), ("Cl", [-0.06, -1.63, 0])]), "C2h"),
        ("环丙烯正离子 C3H3+", _mk([("C", [1.40 * math.cos(math.radians(120 * k)),
                                           1.40 * math.sin(math.radians(120 * k)), 0.0])
                                     for k in range(3)] +
                                   [("H", [2.48 * math.cos(math.radians(120 * k)),
                                           2.48 * math.sin(math.radians(120 * k)), 0.0])
                                    for k in range(3)]), "D3h"),
        ("NH3", _mk([("N", [0, 0, 0.36])] +
                    [("H", [0.94 * math.cos(math.radians(120 * k)),
                            0.94 * math.sin(math.radians(120 * k)), 0.0]) for k in range(3)]), "C3v"),
        ("H2O", _mk([("O", [0, 0, 0.22]), ("H", [0.76, 0, -0.44]), ("H", [-0.76, 0, -0.44])]), "C2v"),
        ("XeF4", _mk([("Xe", [0, 0, 0])] +
                     [("F", list(map(float, v))) for v in
                      ([1.95, 0, 0], [-1.95, 0, 0], [0, 1.95, 0], [0, -1.95, 0])]), "D4h"),
        ("正八面体 6 个同种原子", _mk([("X", list(map(float, v))) for v in
                                      ([1.0, 0, 0], [-1.0, 0, 0], [0, 1, 0], [0, -1, 0],
                                       [0, 0, 1], [0, 0, -1])]), "Oh"),
        ("正四面体 4 个同种原子", _mk([("X", [a, b, c]) for a, b, c in
                                      ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]), "Td"),
        ("正二十面体 12 个顶点", _mk([("X", list(map(float, v))) for v in
                                      ([0, 1, (1 + 5 ** 0.5) / 2], [0, -1, (1 + 5 ** 0.5) / 2],
                                       [0, 1, -(1 + 5 ** 0.5) / 2], [0, -1, -(1 + 5 ** 0.5) / 2],
                                       [1, (1 + 5 ** 0.5) / 2, 0], [-1, (1 + 5 ** 0.5) / 2, 0],
                                       [1, -(1 + 5 ** 0.5) / 2, 0], [-1, -(1 + 5 ** 0.5) / 2, 0],
                                       [(1 + 5 ** 0.5) / 2, 0, 1], [(1 + 5 ** 0.5) / 2, 0, -1],
                                       [-(1 + 5 ** 0.5) / 2, 0, 1], [-(1 + 5 ** 0.5) / 2, 0, -1])]),
         "Ih"),
        ("SF4（跷跷板）", _mk([("S", [0, 0, 0.2]),
                               ("F", [1.55, 0, -0.55]), ("F", [-1.55, 0, -0.55]),
                               ("F", [0, 1.65, 0.62]), ("F", [0, -1.65, 0.62])]), "C2v"),
        ("S8 环（S4 型盘状）", _mk([("X", [1.0 * math.cos(math.radians(90 * k + 45 * (k % 2))),
                                          1.0 * math.sin(math.radians(90 * k + 45 * (k % 2))),
                                          0.35 * (1 if k < 4 else -1)]) for k in range(4)]), None),
    ]
    for label, atoms, want in cases:
        r = point_group(atoms)
        if want is None:
            print("  [info] %-40s → %s" % (label, r.get("point_group")))
            continue
        chk(label, r.get("point_group") if r.get("ok") else r.get("error"), want)
        if not r.get("ok"):
            continue
        if label in ("CH2Cl2", "H2O"):
            print("         steps: " + " / ".join(r["steps"][-2:]))

    print("\n[2] 特征标表（阶 / 类数 / 不可约表示数 / 正交性 / 基函数）")
    tbl = character_table("D4h")
    chk("character_table(D4h).order", tbl["order"], 16)
    chk("character_table(D4h) 类数", tbl["n_classes"], 10)
    chk("character_table(D4h) 不可约表示数", tbl["n_irreps"], 10)
    chk("D4h 十个类标签", " ".join(c["label"] for c in tbl["classes"]),
        "E 2C4 C2 2C2' 2C2'' i 2S4 σh 2σv 2σd")
    oh = character_table("Oh")
    chk("Oh 阶/类数/不可约表示数", (oh["order"], oh["n_classes"], oh["n_irreps"]), (48, 10, 10))
    td = character_table("Td")
    chk("Td 阶/类数/不可约表示数", (td["order"], td["n_classes"], td["n_irreps"]), (24, 5, 5))
    h6 = character_table("D6h")
    chk("D6h 阶/类数/不可约表示数", (h6["order"], h6["n_classes"], h6["n_irreps"]), (24, 12, 12))
    ih = character_table("Ih")
    chk("Ih 阶/类数/不可约表示数", (ih["order"], ih["n_classes"], ih["n_irreps"]), (120, 10, 10))
    chk("C2v 的 Γ(z) 表示", character_table("C2v")["basis_reverse"].get("z"), "A1")
    chk("D4h z 的表示", character_table("D4h")["basis_reverse"].get("z"), "A2u")
    chk("Oh xyz 的表示", character_table("Oh")["basis_reverse"].get("xyz"), "A2u")
    chk("S4 类数/表示数（教材合并约定）",
        (character_table("S4")["n_classes"], character_table("S4")["n_irreps"]), (4, 3))
    msgs, bad = _validate_tables()
    print("\n".join(msgs[:14]) if bad else "  " + msgs[0])
    if bad:
        fails.append("特征标表自校验 %d 处不合格" % bad)
    msgs, bad = _selftest_basis()
    print("  " + msgs[0] if not bad else "\n".join(msgs[:14]))
    if bad:
        fails.append("基函数一致性 %d 处不合格" % bad)
    msgs, bad = _selftest_group_closure()
    print("  " + msgs[0] if not bad else "\n".join(msgs[:14]))
    if bad:
        fails.append("群元/类标签 %d 处不合格" % bad)

    print("\n[3] 约化 + 还原自校验")
    r = reduce_representation("C2v", [9, -1, 3, 1])
    chk("reduce(C2v, H2O 的 Γ_3N)", r["gamma"], "3A1 + 1A2 + 3B1 + 2B2")
    chk("reduce(C2v) 还原校验", r["sum_check"], True)
    r = reduce_representation("D6h", [36, 0, 0, 0, -4, 0, 0, 0, 0, 12, 0, 4])
    chk("reduce(D6h, 苯的 Γ_3N) 还原校验", (r["sum_check"], r["dimensions_check"]), (True, True))
    r = reduce_representation("C3", [3, 0])
    chk("reduce(C3, A+E, 合并约定)", (r["gamma"], r["sum_check"]), ("1A + 1E", True))
    r = reduce_representation("C2v", [9, -1, 3])
    chk("reduce(chars 长度错) 应报错", r["ok"], False)
    r = reduce_representation("C∞v", [3])
    chk("reduce(C∞v) 应提示无限群", r["ok"], False)

    print("\n[4] 3N 分解")
    r = normal_modes("C2v", 3)
    chk("normal_modes(C2v, 3) Γ_vib", _fmt_rep(r["gamma_vib"]), "2A1 + 1B1")
    chk("normal_modes(C2v, 3) IR/Raman/静默",
        (r["ir_active"], sorted(r["raman_active"]), r["silent"]),
        (["A1", "B1"], ["A1", "B1"], []))
    r = normal_modes("Oh", 7)
    want = "1A1g + 1Eg + 2T1u + 1T2g + 1T2u"
    # 比集合而非字符串：_fmt_rep 的表示顺序不固定，内容对就够了（SF6 的标准答案就是这 5 项）
    chk("normal_modes(Oh, 7) Γ_vib",
        tuple(sorted(r["gamma_vib"].items())),
        tuple(sorted({"A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1}.items())))
    r = normal_modes("D6h", 12)
    print("  [info] normal_modes(D6h, 12) Γ_vib = %s" % _fmt_rep(r["gamma_vib"]))
    r = normal_modes("D3d", 6)
    print("  [info] normal_modes(D3d, 6) Γ_vib = %s" % _fmt_rep(r["gamma_vib"]))
    r = normal_modes("C∞v", 3, is_linear=True)
    chk("normal_modes(C∞v, 3) Γ_vib", _fmt_rep(r["gamma_vib"]), "2Σ⁺ + 2Π")
    r = normal_modes("D∞h", 3, is_linear=True)
    chk("normal_modes(D∞h, 3)=CO2", _fmt_rep(r["gamma_vib"]), "1Σg⁺ + 1Σu⁺ + 1Πu")
    r = normal_modes("D∞h", 4, is_linear=True)
    chk("normal_modes(D∞h, 4)=C2H2", _fmt_rep(r["gamma_vib"]), "2Σg⁺ + 1Σu⁺ + 1Πg + 1Πu")

    print("\n[5] 选律")
    r = selection_rules("C2v")
    chk("selection_rules(C2v) 偶极矩/旋光",
        (r["dipole"]["has"], r["chiral"]["has"]), (True, False))
    r = selection_rules("D2d")
    # D2d 含 2 个 σd 与 S4 映轴 → 第二类操作在手，构型必与镜像重合 → **无**旋光性。
    # 早先这里期望 (False, True) 是错的（1,3,5,7-四甲基环辛四烯属 S4，正是教材用来说明
    # "有 S 轴即无旋光"的例子）。手性只在纯旋转群 C1 / Cn / Dn / T / O / I 中出现。
    chk("selection_rules(D2d) 偶极矩/旋光",
        (r["dipole"]["has"], r["chiral"]["has"]), (False, False))
    r = selection_rules("D2")
    # 纯旋转群 D2（扭曲联苯类）：无 i 无 σ → 可手性；D 族有 ≥2 条不平行 C2 → 无偶极矩。
    chk("selection_rules(D2) 偶极矩/旋光",
        (r["dipole"]["has"], r["chiral"]["has"]), (False, True))
    r = selection_rules("S4")
    chk("selection_rules(S4) 旋光", r["chiral"]["has"], False)
    r = selection_rules("C2")
    chk("selection_rules(C2) 手性", (r["dipole"]["has"], r["chiral"]["has"]), (True, True))
    r = selection_rules("Td")
    chk("selection_rules(Td) 偶极矩", r["dipole"]["has"], False)

    print("\n[6] 操作乘法")
    for pt, a, b, want in [("C2v", "C2(z)", "σ(xz)", None),
                           ("C2h", "C2(z)", "σ(xy)", "i"),
                           ("C3v", "C3(z)", "σv", None),
                           ("D6h", "C6(z)", "C2'", None),
                           ("C2v", "C3(z)", "σv", None)]:
        r = compose_operations(pt, a, b)
        if want is not None or r.get("ok"):
            print("  [info] %s: %s·%s = %s (det=%s)" %
                  (pt, a, b, r.get("product_name") if r.get("ok") else r.get("error"),
                   r.get("determinant")))
        if want is not None:
            chk("%s: %s·%s" % (pt, a, b), r.get("product_name"), want)

    print("\n[7] 非法输入必须优雅报错（返回 ok=False，不抛异常）")
    bad_inputs = [
        ("point_group([])", lambda: point_group([])),
        ("point_group('x')", lambda: point_group("x")),
        ("point_group 缺 xyz", lambda: point_group([{"sym": "H"}])),
        ("point_group tol=0", lambda: point_group(_benzene(), tol=0)),
        ("character_table('Xx')", lambda: character_table("Xx")),
        ("symmetry_elements('Q9')", lambda: symmetry_elements("Q9")),
        ("normal_modes('C2v', 500)", lambda: normal_modes("C2v", 500)),
        ("compose('C2v','C3(z)','σ(xz)')", lambda: compose_operations("C2v", "C3(z)", "σ(xz)")),
    ]
    for label, fn in bad_inputs:
        try:
            res = fn()
            ok = (res.get("ok") is False) and bool(res.get("error"))
        except Exception as exc:  # noqa: BLE001
            ok = False
            res = {"error": "抛异常：%r" % (exc,)}
        print("  [%s] %-34s %s" % ("OK " if ok else "FAIL", label,
                                   str(res.get("error"))[:64]))
        if not ok:
            fails.append(label + " 未优雅报错")

    print("\n[8] 判据树全量反验：39 个有限点群的群元 → 判据树必须还原点名")
    okcnt = 0
    for name, tb in sorted(_load_data().items()):
        if tb.get("order") == "∞":
            continue
        mats = [m for m, _l in _build_group(name)]
        info = _classify_mats(mats)
        ok = info["ok"] and info["point_group"] == name and info["order"] == int(tb["order"])
        if ok:
            okcnt += 1
        else:
            print("  [FAIL] %-6s 判据树 → %s（h=%d，期望 h=%s）"
                  % (name, info["point_group"], info["order"], tb["order"]))
            print("         %s" % info["reason"])
            fails.append("%s 判据树反验失败" % name)
    print("  %d 个点群的判据树反验通过。" % okcnt)

    print("\n" + "=" * 78)
    if fails:
        print("自测失败 %d 项：" % len(fails))
        for f in fails:
            print("  - " + f)
        print("=" * 78)
        return 1
    print("全部自测通过 ✓")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
