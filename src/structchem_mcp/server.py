# -*- coding: utf-8 -*-
"""structchem MCP server —— 结构化学课程答题的确定性工具层。

传输：stdio（纯本地，无网络、无 API key）。
    python -m structchem_mcp.server
自测（不经过客户端）：
    python -m structchem_mcp.server --selftest

设计边界（说清楚，别把它当万能解题器）：
- 这里只做**可确定性计算**的事：点群判定、特征标查表、约化表示、3N-6 分解、
  势箱/类氢/光谱项数值、Hückel 久期行列式、晶体几何、晶体场 CFSE 与磁矩。
- 判据选择、作答文风、采分点、图形怎么画属于化学判断，由模型按 skill 的方法论
  落笔，插件只提供数值与过程核对项（每个工具都带 steps）。
- 输入不足时返回 ok=False + 中文原因，绝不猜一个"看起来对"的数值。
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as _exc:  # pragma: no cover
    raise SystemExit(
        "本插件需要 MCP Python SDK 1.x（FastMCP）。\n"
        "修复：pip install 'mcp>=1.0,<2'\n"
        f"（原始错误：{_exc}）"
    ) from _exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structchem_mcp import core  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

mcp = FastMCP("structchem")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    with open(os.path.join(_DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _safe(mod: str, fn: str):
    """把「模块缺失 / 函数没实现 / 内部异常」统一折成 ok=False 的结构化回执。

    模型拿到的永远是可读的中文原因，而不是 MCP 层面的调用失败。
    """

    def deco(factory):
        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            try:
                m = core.get_module(mod)
            except ImportError as exc:
                return {
                    "ok": False,
                    "error": f"模块 {mod} 不可用：{exc}",
                    "hint": "检查插件安装是否完整（src/structchem_mcp/core/）",
                }
            impl = getattr(m, fn, None)
            if impl is None:
                return {
                    "ok": False,
                    "error": f"{mod}.{fn} 未实现",
                    "hint": "该工具尚在开发中；请用可用的相邻工具，或告知用户该项暂缺。",
                }
            try:
                out = factory(impl, *args, **kwargs)
            except TypeError as exc:
                return {"ok": False, "error": f"参数不对：{exc}"}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if not isinstance(out, dict):
                return {"ok": True, "result": out}
            out.setdefault("ok", True)
            return out

        # ── 关键：把注入用的第一个形参 impl 从对外签名里剔掉 ──────────────────
        # FastMCP 用 inspect.signature() 生成 JSON Schema，而 inspect.signature 默认
        # 跟随 functools.wraps 设的 __wrapped__ 去取**原始 factory** 的签名 ——
        # 于是工具会多出一个必填的 "impl": {"type": "string"}。
        # 真实后果（装进 DSH 后实测）：tools/list 里 24 个计算工具的 required 全变成
        # ["impl", ...]，模型被迫给 impl 传一个字符串，调用必然失败，整个插件不可用。
        # __signature__ 的优先级高于 __wrapped__，因此这里显式改写即可，不需放弃 wraps。
        _sig = inspect.signature(factory)
        wrapper.__signature__ = _sig.replace(
            parameters=list(_sig.parameters.values())[1:]
        )
        return wrapper

    return deco


# ---------------------------------------------------------------------------
# 量子化学基础与原子结构（quantum）
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe("quantum", "particle_box")
def sc_particle_box(impl, n: int = 1, L_pm: float | list = 100.0, dim: int = 1,
                    electrons: int | None = None, n1: int | None = None,
                    n2: int | None = None) -> dict:
    """粒子在势箱（一维/二维/三维）：能级、节点数、简并度、跃迁波长、零点能。

    L_pm：一维给箱长（pm）标量；三维给 [a,b,c] 可算长方体（三边不等时简并自动解除）。
    dim=1/2/3。给 electrons 则自动算 HOMO→LUMO 跃迁；给 n1/n2 则显式指定跃迁两端。
    返回 levels / zero_point_eV / transition / steps，steps 可直接抄进答题纸。
    """
    return impl(n=n, L_pm=L_pm, dim=dim, electrons=electrons, n1=n1, n2=n2)


@mcp.tool()
@_safe("quantum", "hydrogen_orbital")
def sc_hydrogen_orbital(impl, n: int, l: int, m: int = 0) -> dict:
    """类氢原子轨道（n,l,m）：径向/角向节面数、能量、|L|、Lz、简并度、最可几半径、平均半径。

    画径向分布图、判断节面数、算角动量都用它。含 steps。
    """
    return impl(n=n, l=l, m=m)


@mcp.tool()
@_safe("quantum", "slater")
def sc_slater(impl, config: str, target: str) -> dict:
    """Slater 规则：屏蔽常数 σ、有效核电荷 Z*、轨道能量。

    config 例："Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"，target 例："3d"。
    返回 groups（每个电子的贡献明细）、sigma、z_eff、E_eV、steps。
    """
    return impl(config=config, target=target)


@mcp.tool()
@_safe("quantum", "spectroscopic_terms")
def sc_spectroscopic_terms(impl, config: str) -> dict:
    """L-S 耦合光谱项：全部光谱项、微观状态数自校验、Hund 规则基项与 J 值。

    config 例："d2" / "d6" / "p2" / "d5" / "3d6" / "p d"（非同科）。
    返回 terms（含每个项的 L/S/多重度/J 列表/微观状态数）、microstates_total、
    check（自校验是否通过）、ground_term。微观状态数校验不通过会直接标红。
    """
    return impl(config=config)


@mcp.tool()
@_safe("quantum", "superposition")
def sc_superposition(impl, terms: list) -> dict:
    """波函数线性叠加：能量平均值、各本征值出现几率、L² 与 Lz 的平均值与取值几率。

    terms 例：[{"coef":0.8,"n":2,"l":1,"m":0},{"coef":0.6,"n":3,"l":1,"m":1}]。
    正是"ψ=C1ψ210+C2ψ311 求能量平均值/角动量"这类题。
    """
    return impl(terms=terms)


# ---------------------------------------------------------------------------
# 分子对称性与群论（symmetry）
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe("symmetry", "point_group")
def sc_point_group(impl, atoms: list, tol: float = 0.15) -> dict:
    """分子点群判定（真算：枚举并检验全部对称操作后定 Schoenflies 记号）。

    atoms 例：[{"sym":"O","xyz":[0,0,0]},{"sym":"H","xyz":[75.7,0,0]},
             {"sym":"H","xyz":[-24.0,71.7,0]}]（单位 Å，函数自己平移质心）。
    返回 point_group / order / operations / elements / international / steps。
    steps 是可抄进答题纸的判定链。查不出就报错，不猜。
    """
    return impl(atoms=atoms, tol=tol)


@mcp.tool()
@_safe("symmetry", "symmetry_elements")
def sc_symmetry_elements(impl, point_group: str) -> dict:
    """点群 → 对称元素与对称操作清单、群的阶、类的划分、生成元。

    写"某分子有哪些对称元素/操作"、"列出全部对称操作"这类题用它。
    """
    return impl(point_group_name=point_group)


@mcp.tool()
@_safe("symmetry", "character_table")
def sc_character_table(impl, point_group: str) -> dict:
    """特征标表（含基函数归属：x,y,z / Rx,Ry,Rz / 二次函数）。

    查基函数、判断某运动属哪个不可约表示、做投影算符都从这里取。
    返回 classes / irreps / basis / basis_reverse / order / n_classes / n_irreps。
    """
    return impl(point_group_name=point_group)


@mcp.tool()
@_safe("symmetry", "reduce_representation")
def sc_reduce_representation(impl, point_group: str, chars: list) -> dict:
    """可约表示约化：n_i = (1/h)Σ g_c·χ_i(c)·χ_r(c)，含逐项代入算式。

    chars 必须按该点群类的顺序给（先用 sc_character_table 看类顺序）。
    返回 decomposition / sum_check / dimensions_check / steps。
    """
    return impl(point_group_name=point_group, chars=chars)


@mcp.tool()
@_safe("symmetry", "normal_modes")
def sc_normal_modes(impl, point_group: str, n_atoms: int, is_linear: bool = False,
                    with_rotation: bool = True) -> dict:
    """3N 自由度分解：Γ_3N → Γ_平动 + Γ_转动 + Γ_振动，再判 IR/Raman 活性。

    "求分子振动的不可约表示并说明哪些是红外活性/拉曼活性"这类题的标准工具。
    返回 gamma_3N / gamma_vib / ir_active / raman_active / silent / steps。
    """
    return impl(point_group_name=point_group, n_atoms=n_atoms, is_linear=is_linear)


@mcp.tool()
@_safe("symmetry", "selection_rules")
def sc_selection_rules(impl, point_group: str) -> dict:
    """偶极矩与旋光性判据（按点群）。

    返回 dipole{has,direction,reason} / chiral{has,reason} / allowed_ir_basis /
    allowed_raman_basis / steps。"判断有无偶极矩和旋光性"直接用它。
    """
    return impl(point_group_name=point_group)


@mcp.tool()
@_safe("symmetry", "compose_operations")
def sc_compose_operations(impl, point_group: str, op1: str, op2: str) -> dict:
    """对称操作乘法：C2(z)·σ(xy) = ? 先算矩阵再给等价操作名。

    用于"用表示矩阵证明 C2(z)σxy = i"这类题。返回 matrix / det / equivalent /
    is_proper / steps。
    """
    return impl(point_group_name=point_group, op1=op1, op2=op2)


# ---------------------------------------------------------------------------
# 分子轨道与化学键（orbital）
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe("orbital", "huckel")
def sc_huckel(impl, topology: str = "linear", n_atoms: int | None = None,
              n_electrons: int | None = None, adj: list | None = None) -> dict:
    """Hückel 分子轨道法（HMO）：真解久期行列式。

    topology 取 "linear"（共轭多烯链）/ "cyclic"（环）/ "custom"（配 adj 邻接矩阵）。
    返回 orbitals（能级 α+xβ、系数、占据）、pi_energy_total、delocalization_energy、
    bond_orders、charge_densities、free_valence、steps（含久期行列式与 x 的解）。
    键级/电荷密度/自由价/分子图数据一次给全。
    """
    return impl(topology=topology, n_atoms=n_atoms, n_electrons=n_electrons, adj=adj)


@mcp.tool()
@_safe("orbital", "aromaticity")
def sc_aromaticity(impl, n_pi: int, n_atoms: int, ring: bool = True,
                   planar: bool = True) -> dict:
    """Hückel 4n+2 芳香性判定（区分芳香 / 反芳香 / 非芳香）。"""
    return impl(n_pi=n_pi, n_atoms=n_atoms, ring=ring, planar=planar)


@mcp.tool()
@_safe("orbital", "diatomic_mo")
def sc_diatomic_mo(impl, n_electrons: int, period: int = 2,
                   homo: str | None = None) -> dict:
    """第二周期双原子分子 MO：电子组态、键级、未成对电子、磁性。

    n_electrons 是**总电子数**（O2 = 16，N2 = 14，NO = 15，F2 = 18），
    内部自动折算价电子数并在 steps 里写明折算过程。
    返回 config / bond_order / unpaired_electrons / magnetism / steps。
    O2 顺磁、N2 键级 3 这类结论由计算给出；MO 能级顺序两套（N2 及以前 σ2pz 高于 π2p，
    O2/F2 相反）由函数自行判定。
    """
    kwargs = {"n_electrons": n_electrons, "period": period}
    if homo is not None:
        kwargs["homo"] = homo
    return impl(**kwargs)


@mcp.tool()
@_safe("orbital", "vsepr")
def sc_vsepr(impl, central: str, ligands: list, lone_pairs: int = 0,
             ligand_charges: list | None = None) -> dict:
    """VSEPR：价层电子对数 → 构型、杂化、键角、点群、VSEPR 记号。

    返回 steric_number / vsepr_type / geometry / hybrid / bond_angles /
    point_group / examples / steps。
    """
    return impl(central=central, ligands=ligands, lone_pairs=lone_pairs,
                ligand_charges=ligand_charges)


# ---------------------------------------------------------------------------
# 晶体结构（crystal）
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe("crystal", "unit_cell")
def sc_unit_cell(impl, a: float, b: float, c: float, alpha: float = 90.0,
                 beta: float = 90.0, gamma: float = 90.0) -> dict:
    """晶胞参数 → 晶系判定、布拉维格子、晶胞体积（pm³ 与 Å³）。"""
    return impl(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


@mcp.tool()
@_safe("crystal", "miller_indices")
def sc_miller_indices(impl, intercepts: list) -> dict:
    """截距 → 晶面指标（Miller 指数）。

    intercepts 例：["a/2","b/3","c/4"] 或 [0.5, 1/3, 0.25] 或 ["1/2","1/3","1/4"]；
    截距 ∞ 或 0 表示与该轴平行。返回 h,k,l / miller / steps。
    """
    return impl(intercepts=intercepts)


@mcp.tool()
@_safe("crystal", "d_spacing")
def sc_d_spacing(impl, h: int, k: int, l: int, a: float, b: float | None = None,
                 c: float | None = None, alpha: float = 90.0, beta: float = 90.0,
                 gamma: float = 90.0) -> dict:
    """晶面间距 d(hkl)：按晶系自动选公式，返回 d_pm / d_A / formula_used / steps。"""
    return impl(h=h, k=k, l=l, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


@mcp.tool()
@_safe("crystal", "packing")
def sc_packing(impl, kind: str) -> dict:
    """堆积方式（fcc / hcp / bcc / sc / diamond）：原子数、配位数、空间利用率、
    半径-晶胞参数关系、空隙数与典型物质。"""
    return impl(kind=kind)


@mcp.tool()
@_safe("crystal", "radius_ratio")
def sc_radius_ratio(impl, r_plus_pm: float, r_minus_pm: float) -> dict:
    """半径比规则 → 配位数与配位多面体形状（含临界值区间与 steps）。"""
    return impl(r_plus_pm=r_plus_pm, r_minus_pm=r_minus_pm)


@mcp.tool()
@_safe("crystal", "cell_content")
def sc_cell_content(impl, formula: str, a_pm: float,
                    density_gcm3: float | None = None, z: int | None = None,
                    r_plus_pm: float | None = None,
                    r_minus_pm: float | None = None,
                    b_pm: float | None = None, c_pm: float | None = None,
                    volume_pm3: float | None = None) -> dict:
    """晶胞内容物：由密度反算 Z（或由 Z 算密度）、离子空间利用率。

    ρ = Z·M/(N_A·V)。给 z 正算密度；给 density_gcm3 反算 Z。
    体积默认按立方 V=a³；给 b_pm / c_pm 按直角晶胞 a·b·c；
    六方/单斜/三斜直接传 volume_pm3（优先于 a³）。
    给 r_plus_pm / r_minus_pm 会额外算离子接触与空间利用率（仅 AB 型成立）。
    """
    return impl(formula=formula, a_pm=a_pm, density_gcm3=density_gcm3, z=z,
                r_plus=r_plus_pm, r_minus=r_minus_pm,
                b_pm=b_pm, c_pm=c_pm, volume_pm3=volume_pm3)


# ---------------------------------------------------------------------------
# 配位化合物（coordination）
# ---------------------------------------------------------------------------

@mcp.tool()
@_safe("coordination", "crystal_field")
def sc_crystal_field(impl, d_n: int, geometry: str = "octahedral",
                     pairing_energy_cm1: float | None = None,
                     delta_o_cm1: float | None = None,
                     metal: str | None = None, ligand: str | None = None) -> dict:
    """晶体场理论：d 轨道分裂、CFSE、高/低自旋、未成对电子、磁矩、磁性。

    d_n 取 1..10；geometry 取 "octahedral" / "tetrahedral"。
    给 delta_o_cm1 与 pairing_energy_cm1 则比较 Δo 与 P 定高低自旋；
    也可以只给 metal / ligand 让函数按光谱化学序列估计。
    返回 splitting / cfse_dq / cfse_cm1 / spin_state / unpaired_electrons /
    magnetic_moment_BM / magnetism / steps。

    平面四方（square_planar）不在本工具范围：其 d 轨道分组与 CFSE 基准（以八面体重心为
    零点时 dxz,dyz −5.14Dq / dxy −2.28Dq / dz² +2.28Dq / dx²−y² +12.28Dq）与八面体不是
    同一把尺，混算必然出错，需另行推导。
    """
    return impl(d_n=d_n, geometry=geometry, pairing_energy_cm1=pairing_energy_cm1,
                delta_o_cm1=delta_o_cm1, metal=metal, ligand=ligand)


@mcp.tool()
@_safe("coordination", "spectrochemical_series")
def sc_spectrochemical_series(impl, ligand: str) -> dict:
    """配体在光谱化学序列中的位置、Δo 相对大小、常见实例。"""
    return impl(ligand=ligand)


# ---------------------------------------------------------------------------
# 知识检索（数据层：答题卡 / 归因词典 / 教材索引 / 答案风格）
# ---------------------------------------------------------------------------

def _keyword_search(items: list, query: str, keys: list[str]) -> list:
    q = (query or "").strip().lower()
    if not q:
        return list(items)
    hits = []
    for it in items:
        blob = " ".join(
            str(it.get(k, "")) if not isinstance(it.get(k), list)
            else " ".join(str(x) for x in it.get(k, []))
            for k in keys
        ).lower()
        if q in blob:
            hits.append(it)
    if hits:
        return hits
    # 退化：按字符重合度打分
    scored = []
    for it in items:
        blob = " ".join(
            str(it.get(k, "")) if not isinstance(it.get(k), list)
            else " ".join(str(x) for x in it.get(k, []))
            for k in keys
        ).lower()
        score = sum(1 for ch in set(q) if ch in blob)
        if score >= max(2, len(set(q)) // 2):
            scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored]


@mcp.tool()
def sc_answer_card(query: str = "") -> dict:
    """题型答题卡：作答步骤、可直接抄的答题模板、阅卷采分点、常见陷阱、对应工具。

    query 例："点群"、"约化表示"、"振动分解"、"光谱项"、"HMO"、"晶面指标"、
    "半径比"、"晶体场"、"空间利用率"、"偶极矩"。不传则列出全部题型。
    """
    try:
        data = _load("answer_cards.json")
    except FileNotFoundError:
        return {"ok": False, "error": "answer_cards.json 未安装",
                "hint": "重装插件或检查 src/structchem_mcp/data/"}
    cards = data.get("cards", [])
    if not query:
        return {
            "ok": True,
            "count": len(cards),
            "cards": [{"id": c.get("id"), "name": c.get("name"),
                       "keywords": c.get("keywords", [])} for c in cards],
        }
    hits = _keyword_search(cards, query, ["id", "name", "keywords"])
    return {"ok": True, "query": query, "count": len(hits), "cards": hits[:4]}


@mcp.tool()
def sc_rationale(query: str = "") -> dict:
    """归因词典：把「为什么」类问题落到可命名的原理上。

    query 例："偶极矩"、"旋光性"、"红外活性"、"能级高低"、"键长"、"磁性"、
    "芳香性"、"颜色"、"配位数"、"氢键"。返回 criteria / template / common_wrong / example。
    """
    try:
        data = _load("rationale.json")
    except FileNotFoundError:
        return {"ok": False, "error": "rationale.json 未安装"}
    dims = data.get("dims", [])
    if not query:
        return {"ok": True, "count": len(dims),
                "dims": [{"id": d.get("id"), "name": d.get("name")} for d in dims]}
    hits = _keyword_search(dims, query, ["id", "name", "keywords"])
    return {"ok": True, "query": query, "count": len(hits), "dims": hits[:4]}


@mcp.tool()
def sc_book_index(chapter: str = "") -> dict:
    """教材索引（《结构化学基础》第 5 版）：章节 → 节、考点、概念定义、公式、对应工具。

    chapter 例："4"、"对称性"、"晶体"、"配位"。不传则列出十章目录。
    """
    try:
        data = _load("book_index.json")
    except FileNotFoundError:
        return {"ok": False, "error": "book_index.json 未安装"}
    chapters = data.get("chapters", [])
    if not chapter:
        return {"ok": True, "count": len(chapters),
                "chapters": [{"no": c.get("no"), "title": c.get("title")}
                             for c in chapters]}
    hits = _keyword_search(chapters, chapter, ["no", "title", "sections", "concepts"])
    if not hits:
        try:
            n = int(chapter)
            hits = [c for c in chapters if c.get("no") == n]
        except ValueError:
            pass
    return {"ok": True, "query": chapter, "count": len(hits), "chapters": hits[:2]}


@mcp.tool()
def sc_style_law() -> dict:
    """答案风格铁律：这套课的答案「怎么写才像标准答案」的实测结论（含原文证据）。

    动笔前读一次，能避免把 20 字的答案写成 200 字的作文。
    """
    for name in ("answer_style.json", "style_law.json"):
        try:
            return {"ok": True, **_load(name)}
        except FileNotFoundError:
            continue
    # data -> structchem_mcp -> src -> 包根
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(_DATA_DIR)))
    path = os.path.join(pkg_root, "skills", "structural-chem-answering",
                        "references", "answer-style.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {"ok": True, "source": path, "markdown": f.read()}
    return {"ok": False, "error": "答案风格文件未找到", "searched": path}


@mcp.tool()
def sc_notation(query: str = "") -> dict:
    """符号与排版规范：多义希腊字母的消歧、上下标双轨写法、单位与括号约定。

    query 例："σ"、"μ"、"点群"、"光谱项"、"t2g"、"单位"、"有效数字"、"晶面"、
    "ν"、"Δ"、"不可约表示"。不传则返回全部符号索引 + 排版条目清单 + 速查铁律。

    为什么要有它：结构化学里 σ、π、χ、μ、Δ、α、β、Z、φ 每个都至少三种含义，
    写错一个字母整道题就答成了另一道题。落笔前查一下，比事后改答案便宜。
    """
    try:
        data = _load("notation_rules.json")
    except FileNotFoundError:
        return {"ok": False, "error": "notation_rules.json 未安装"}

    symbols = data.get("symbols", [])
    typo = data.get("typography", [])
    rules = data.get("quick_rules", [])

    if not query:
        return {
            "ok": True,
            "core_principle": data.get("_meta", {}).get("core_principle"),
            "symbols": [{"sym": s["sym"],
                         "senses": [x["where"] + "：" + x["means"] for x in s["senses"]]}
                        for s in symbols],
            "typography_items": [t["item"] for t in typo],
            "quick_rules_count": len(rules),
        }

    q = query.strip()
    qlow = q.lower()

    def blob(d) -> str:
        parts = []
        for v in d.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        parts.append(x)
                    elif isinstance(x, dict):
                        parts.extend(str(y) for y in x.values() if isinstance(y, (str,)))
        return " ".join(parts).lower()

    # 1) 符号：精确字符命中优先，其次模糊
    sym_hits = [s for s in symbols if s["sym"] == q]
    if not sym_hits:
        sym_hits = [s for s in symbols if qlow in blob(s) or q in s["sym"]]
    # 2) 排版条目
    typo_hits = [t for t in typo if qlow in blob(t) or qlow in t["item"].lower()]
    # 3) 速查铁律
    rule_hits = [r for r in rules if any(tok and tok in r for tok in
                                        (q, q.replace("⁻¹", "-1"), q.replace("₂", "2")))]
    if not (sym_hits or typo_hits or rule_hits):
        return {
            "ok": False,
            "error": f"没有匹配「{query}」的符号或排版条目",
            "hint": "可查：" + "、".join(s["sym"] for s in symbols[:12]) +
                    "；或「点群」「光谱项」「单位」「有效数字」「晶面」。",
        }
    return {
        "ok": True,
        "query": query,
        "symbols": sym_hits[:4],
        "typography": typo_hits[:4],
        "quick_rules": rule_hits[:6],
        "note": "写答案前扫一眼 trap 字段：那是这类符号最高频的失分点。",
    }


@mcp.tool()
def sc_toolbox() -> dict:
    """列出本插件全部工具、所属模块与可用性（排障与自我介绍用）。

    返回每个模块的导入状态；某模块坏了会给出真实异常，便于定位。
    """
    avail = core.availability()
    tools_by_module = {
        "quantum": ["sc_particle_box", "sc_hydrogen_orbital", "sc_slater",
                    "sc_spectroscopic_terms", "sc_superposition"],
        "symmetry": ["sc_point_group", "sc_symmetry_elements", "sc_character_table",
                     "sc_reduce_representation", "sc_normal_modes",
                     "sc_selection_rules", "sc_compose_operations"],
        "orbital": ["sc_huckel", "sc_aromaticity", "sc_diatomic_mo", "sc_vsepr"],
        "crystal": ["sc_unit_cell", "sc_miller_indices", "sc_d_spacing",
                    "sc_packing", "sc_radius_ratio", "sc_cell_content"],
        "coordination": ["sc_crystal_field", "sc_spectrochemical_series"],
        "data": ["sc_answer_card", "sc_rationale", "sc_book_index", "sc_style_law",
                 "sc_notation"],
    }
    total = sum(len(v) for v in tools_by_module.values()) + 1  # +1 = sc_toolbox 自己
    broken = [k for k, v in avail.items() if not v["ok"]]
    return {
        "ok": True,
        "version": "1.0.0",
        "tool_count": total,
        "tools_by_module": tools_by_module,
        "module_health": avail,
        "broken_modules": broken,
        "note": "每个计算工具都返回 steps（逐步算式），可直接抄进答题纸。"
                "sc_toolbox 自身不在 tools_by_module 里，故 tool_count = 表内合计 + 1。",
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _selftest() -> int:
    avail = core.availability()
    print("== 模块健康 ==")
    for name, info in avail.items():
        flag = "OK  " if info["ok"] else "FAIL"
        extra = "" if info["ok"] else f"  <- {info['error']}"
        print(f"[{flag}] {name:14s} funcs={len(info['functions'])}{extra}")

    cases = [
        ("particle_box n=1 L=100pm", lambda: _call("quantum", "particle_box",
                                                   n=1, L_pm=100.0, dim=1)),
        ("hydrogen (3,1,0)", lambda: _call("quantum", "hydrogen_orbital", n=3, l=1, m=0)),
        ("slater Fe 3d", lambda: _call("quantum", "slater",
                                       config="Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2", target="3d")),
        ("terms d2", lambda: _call("quantum", "spectroscopic_terms", config="d2")),
        ("terms d6", lambda: _call("quantum", "spectroscopic_terms", config="d6")),
        ("point_group CH2Cl2", lambda: _call("symmetry", "point_group", atoms=_CH2CL2)),
        ("point_group 苯", lambda: _call("symmetry", "point_group", atoms=_BENZENE)),
        ("point_group SF6", lambda: _call("symmetry", "point_group", atoms=_SF6)),
        ("character_table D4h", lambda: _call("symmetry", "character_table",
                                              point_group_name="D4h")),
        ("reduce C2v", lambda: _call("symmetry", "reduce_representation",
                                     point_group_name="C2v", chars=[9, -1, 3, 1])),
        ("normal_modes C2v n=3", lambda: _call("symmetry", "normal_modes",
                                               point_group_name="C2v", n_atoms=3)),
        ("normal_modes Oh n=7", lambda: _call("symmetry", "normal_modes",
                                              point_group_name="Oh", n_atoms=7)),
        ("selection_rules C2v", lambda: _call("symmetry", "selection_rules",
                                              point_group_name="C2v")),
        ("huckel 苯", lambda: _call("orbital", "huckel", topology="cyclic",
                                    n_atoms=6, n_electrons=6)),
        ("huckel 丁二烯", lambda: _call("orbital", "huckel", topology="linear",
                                        n_atoms=4, n_electrons=4)),
        ("diatomic O2", lambda: _call("orbital", "diatomic_mo", n_electrons=16)),
        ("diatomic N2", lambda: _call("orbital", "diatomic_mo", n_electrons=14)),
        ("vsepr SF4", lambda: _call("orbital", "vsepr", central="S",
                                    ligands=["F", "F", "F", "F"], lone_pairs=1)),
        ("miller a/2,b/3,c/4", lambda: _call("crystal", "miller_indices",
                                             intercepts=["a/2", "b/3", "c/4"])),
        ("packing fcc", lambda: _call("crystal", "packing", kind="fcc")),
        ("radius_ratio 133/328", lambda: _call("crystal", "radius_ratio",
                                               r_plus_pm=133.0, r_minus_pm=328.0)),
        ("crystal_field d6 oct", lambda: _call("coordination", "crystal_field",
                                               d_n=6, geometry="octahedral")),
        ("answer_card 点群", lambda: sc_answer_card("点群")),
        ("rationale 偶极矩", lambda: sc_rationale("偶极矩")),
        ("book_index 4", lambda: sc_book_index("4")),
        ("style_law", lambda: sc_style_law()),
        ("toolbox", lambda: sc_toolbox()),
    ]

    ok = 0
    for name, fn in cases:
        try:
            r = fn()
            good = isinstance(r, dict) and r.get("ok")
            if good:
                ok += 1
            brief = json.dumps(r, ensure_ascii=False, default=str)[:160]
            print(f"[{'OK  ' if good else 'FAIL'}] {name}: {brief}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR ] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(cases)} passed")
    return 0 if ok == len(cases) else 1


def _call(mod: str, fn: str, **kwargs):
    m = core.get_module(mod)
    impl = getattr(m, fn)
    return impl(**kwargs)


_CH2CL2 = [
    {"sym": "C", "xyz": [0.0, 0.0, 0.0]},
    {"sym": "Cl", "xyz": [0.0, 1.466, -0.992]},
    {"sym": "Cl", "xyz": [0.0, -1.466, -0.992]},
    {"sym": "H", "xyz": [0.895, 0.0, 0.621]},
    {"sym": "H", "xyz": [-0.895, 0.0, 0.621]},
]

_BENZENE = [
    {"sym": "C", "xyz": [1.397 * __import__("math").cos(3.141592653589793 / 3 * i),
                         1.397 * __import__("math").sin(3.141592653589793 / 3 * i), 0.0]}
    for i in range(6)
] + [
    {"sym": "H", "xyz": [2.481 * __import__("math").cos(3.141592653589793 / 3 * i),
                         2.481 * __import__("math").sin(3.141592653589793 / 3 * i), 0.0]}
    for i in range(6)
]

_SF6 = [{"sym": "S", "xyz": [0.0, 0.0, 0.0]}] + [
    {"sym": "F", "xyz": v} for v in
    ([1.56, 0, 0], [-1.56, 0, 0], [0, 1.56, 0], [0, -1.56, 0], [0, 0, 1.56], [0, 0, -1.56])
]


def _preload() -> None:
    """启动时预热：导入全部 core 子模块，并把最重的一条算法路径实跑一遍。

    为什么要前置：`core.get_module()` 是惰性导入，第一个工具调用才会真正 import
    （symmetry.py 2700+ 行、特征标表首次加载、numpy 首次进线程池）。实测在 MCP 会话里
    首次 `sc_point_group` 能拖到 25 s 以上（第二次同样的调用只要几十毫秒）——
    客户端看到的就是「第一个工具超时」。server 是长驻进程，把这点成本挪到启动期是划算的。

    这里除了 import，还真实跑一遍 point_group / normal_modes / character_table，
    让特征标表加载、群元构造、二部图匹配、缓存全部就位。
    """
    for _m in core._MODULES:
        try:
            core.get_module(_m)
        except ImportError:
            continue  # 单模块坏了不该拦住整个 server，sc_toolbox 会报出来
    try:
        sym = core.get_module("symmetry")
        water = [
            {"sym": "O", "xyz": [0.0, 0.0, 0.117]},
            {"sym": "H", "xyz": [0.0, 0.757, -0.468]},
            {"sym": "H", "xyz": [0.0, -0.757, -0.468]},
        ]
        sym.point_group(water)
        sym.character_table("C2v")
        sym.normal_modes("C2v", 3)
        sym.normal_modes("Oh", 7)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _preload()
    mcp.run()


if __name__ == "__main__":
    main()
