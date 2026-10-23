# dsh-structchem 架构契约（v1）

> **本文件是全体开发成员的唯一接口契约。动手前必读，字段名一律以此为准。**
> 契约由 Lead 维护；成员不得自行改签名。发现契约有错，先在消息里报告 Lead，不要私改。

## 0. 项目定位

《结构化学》（周公度·段连运《结构化学基础》第 5 版体系）**课程答题的确定性工具层 + 作答规范层**。

设计边界（与 dsh-orgchem / dsh-mech 一致）：

- 插件只做**可确定性计算**的事：数值、判定、查表、约化、分解。
- 判据、文风、采分点、题型套路属于 skill，由模型按 skill 方法论落笔。
- 插件绝不"猜"：输入不足就报错并说明缺什么，不返回 plausible-but-wrong 的数值。

## 1. 目录与写域（严格 disjoint，越界写入即冲突）

```
D:\大肥鱼的实验室\dsh-structchem\
├── cordis.patch.yml               [Lead]
├── package.json                   [Lead]
├── README.md                      [Lead]
├── requirements.txt               [Lead]
├── LICENSE                        [Lead]
├── ARCHITECTURE.md                [Lead]  ← 本文件
├── src\structchem_mcp\
│   ├── __init__.py                [Lead]
│   ├── server.py                  [Lead]  ← 只 import core，不写算法
│   ├── core\
│   │   ├── __init__.py            [Lead]  ← 聚合导出
│   │   ├── quantum.py             [quantum-dev]
│   │   ├── symmetry.py            [symmetry-dev]
│   │   ├── orbital.py             [orbital-dev]
│   │   ├── crystal.py             [crystal-dev]
│   │   └── coordination.py        [crystal-dev]
│   └── data\
│       ├── character_tables.json  [symmetry-dev]
│       ├── answer_cards.json      [skill-author]
│       ├── rationale.json         [skill-author]
│       └── book_index.json        [knowledge-miner]
├── skills\structural-chem-answering\
│   ├── SKILL.md                   [skill-author]
│   ├── meta.yaml                  [skill-author]
│   └── references\*.md            [skill-author]
├── tests\                         [Lead]
└── tools\                         [Lead]

D:\大肥鱼的实验室\_structchem_work\   ← 原材料区
├── ocr\exercise_pages.jsonl       [Lead 产出，只读]
├── web\_extract.txt               [Lead 产出，只读]
└── knowledge\                     [knowledge-miner 写]
```

## 2. 通用约定（**所有模块必须遵守**）

### 2.1 返回结构

每个对外函数返回 `dict`，且**必须**含这两个键：

```python
{
  "ok": True,                  # 成功 True；输入非法 False
  "error": None,               # ok=False 时给中文可执行的原因
  ...                          # 业务字段
}
```

输入非法时：返回 `{"ok": False, "error": "<说人话：缺什么/哪里错/怎么改>"}`，**不抛裸异常**。
断言式失败要给出用户能直接抄进答案的那句提示。

### 2.2 数值与单位

- 一律返回**字符串化的过程核对项** + 浮点数值双份，例如
  `{"value": 2.5, "value_str": "2.5", "steps": ["..."]}`。
- `steps` 是**可直接抄进答题纸**的逐步算式（中文，含代入数字）。这是本插件的核心价值，
  没有 steps 的工具视为未完成。
- 长度默认 **pm**（同时给 Å 换算字段），能量默认 **eV**（同时给 `cm-1`、`kJ/mol`、`a.u.` 可选）。
- 角度 `°`；无量纲量（键级、电荷密度、自由价、配位数、空间利用率）直接给。
- 常数统一从 `core/_const.py` 引用（Lead 提供）。**实际可用名**（别凭记忆 import）：
  `H`, `HBAR`, `ME`, `E_CHARGE`, `NA`, `C_LIGHT`, `K_B`,
  `A0_PM`, `A0_M`, `HARTREE_EV`, `RYDBERG_EV`, `RYDBERG_CM1`,
  `EV_TO_CM1`, `EV_TO_KJMOL`, `J_TO_EV`, `MU_B`, `PM_PER_A`, `NM_PER_PM`；
  换算函数 `hartree_to_ev` / `ev_to_cm1` / `ev_to_kjmol` / `ev_to_nm` / `pm_to_a` /
  `spin_only_moment`（= √(n(n+2)) BM）。

### 2.3 代码规范

- Python 3.10+，只用 `numpy`（已装）。**不要**引入 sympy / scipy 之外的新依赖；
  scipy 可用（已装）但优先手写算法以便 steps 可读。
- 纯函数，无 IO，无网络，无全局可变状态。数据文件通过 `_load_data()` 惰性读。
- 中文 docstring，说明「算什么 / 边界在哪 / 不做什么」。
- 每个模块底部写 `if __name__ == "__main__":` 自测块，跑通契约里的验收断言。

## 3. 工具契约（16 个，函数名即 MCP 工具名去掉前缀后的实现名）

> MCP 工具名 = `sc_` + 函数名中的动词，由 server.py 包装。下面给的是 `core` 里的函数名。

### 3.1 quantum.py [quantum-dev]

**Q1 `particle_box(n, L_pm, dim=1, mass="electron", electrons=None)`**
粒子在势箱。dim ∈ {1,2,3}。
返回：
- `levels`: `[{label:"n=1", E_eV, E_str, degeneracy}]`（dim=1 时 n 到 n 输入值；dim=3 时按 n² 排序列出）
- `zero_point_eV`, `nodes`: n-1（一维）
- `transition`: 给 `n1,n2` 或自动取 HOMO→LUMO，含 `dE_eV`, `wavelength_nm`, `steps`
- `steps`: 从 E = n²h²/(8mL²) 起，代入数值。
- 三维等边长简并度：n²=3(111)、6、9、11…（用 n_x²+n_y²+n_z² 枚举）

**Q2 `hydrogen_orbital(n, l, m=0)`**
返回：`n_radial_nodes`=n-l-1、`n_angular_nodes`=l、`n_nodes_total`=n-1、
`energy_eV`/`energy_au`（=-13.6/n²、=-1/(2n²)）、`abs_L`（=√(l(l+1))ħ）、`Lz`（=mħ）、
`degeneracy`（=n²，含自旋 2n²）、`most_probable_r_pm`（1s 用 a0，其它给近似公式或数值扫描）、
`mean_r_pm`（=<r> = n²a0[1+½(1-l(l+1)/n²)]）、`steps`。

**Q3 `slater(config, target)`**
Slater 规则。`config` 例 `"Fe:1s2 2s2 2p6 3s2 3p6 3d6 4s2"`，`target` 例 `"3d"`。
返回：`sigma`, `z_eff`, `groups`（按 Slater 分组与贡献明细）、`E_eV`（=-13.6 Z*²/n*²）、`steps`。
必须正确处理：同组 0.35（1s 同组 0.30）、n-1 层 0.85、n-2 及更内 1.00、
(d/f 电子左侧全算 1.00)。

**Q4 `spectroscopic_terms(config)`**
L-S 耦合。输入 `"d2"` / `"p2"` / `"f3"` / `"3d6"` 等（等价电子）或非同科电子 `"p d"`。
返回：
- `terms`: `[{term:"3F", L:3, S:1, multiplicity:3, J:[2,3,4], microstates:21}]`
- `microstates_total` 与 `check`（=∏C(2(2l+1), k)，必须自校验通过）
- `ground_term`（Hund 三规则：S 最大 → L 最大 → 壳层半满前 J=|L-S|、超过半满 J=L+S）
- `steps`

**Q5 `superposition(terms)`**
波函数叠加。`terms=[{"coef":0.8,"n":2,"l":1,"m":0},{"coef":0.6,"n":3,"l":1,"m":1}]`。
返回：`norm_check`、`energy_mean_eV`、`energy_mean_au`、
`eigen_probabilities`: `[{eigen:"E(n=2)", value_eV, prob}]`、
`L2_mean`、`Lz_mean`、`Lz_values_prob`、`steps`。

### 3.2 symmetry.py [symmetry-dev]

**S1 `point_group(atoms, tol=0.15)`**
`atoms = [{"sym":"H","xyz":[x,y,z]}, ...]`（Å，质心已居中由函数自己平移）。
自己枚举并检验全部 24 个真转动 + 反演 + 反映 + 映轴操作（用矩阵作用于坐标后做
集合匹配，匹配判据：每个原子在 tol 内找到同元素原子，且一一对应）。
返回：`point_group`（Schoenflies）、`order`（阶 h）、`operations`（`["E","C2(z)","σ(xz)",...]`）、
`elements`（对称元素清单）、`international`（若已知映射）、`detected`（每组操作的重合度）、
`steps`（判定推理链：最高轴次 → 有无 nC2⊥ → 有无 σh/σd → …）。
**必须覆盖**：C1 Cs Ci C2 C3 C4 C5 C6 C2v C3v C4v C5v C6v C2h C3h D2 D3 D4 D5 D6 D2h D3h D4h D5h D6h D2d D3d D4d D5d S4 S6 S8 T Td Th O Oh I Ih C∞v D∞h。
线形与球对称单独走快速分支。

**S2 `symmetry_elements(point_group)`**
点群 → `{order, classes:[{name, size, sample_ops}], elements, generators, steps}`。

**S3 `character_table(point_group)`**
读 `data/character_tables.json`。
返回：`{point_group, order, classes:[{label, size}], irreps:[{name, chars:[...]}], basis:{...}, steps}`。
`basis` 给 x,y,z / Rx,Ry,Rz / 二次函数 各自所属不可约表示（做题查基函数必用）。
**数据要求**：至少覆盖 S1 列出的全部点群，含 C∞v/D∞h 的简化表。特征标必须与教材一致。
**类数与不可约表示数必须相等**（群论定理），核对表：

| 点群 | 阶 h | 类数 | 不可约表示数 |
|---|---|---|---|
| C2v | 4 | 4 | 4 |
| C3v | 6 | 3 | 3 |
| D3h | 12 | 6 | 6 |
| D4h | 16 | **10** | **10**（A1g A2g B1g B2g Eg A1u A2u B1u B2u Eu） |
| D6h | 24 | 12 | 12 |
| Td | 24 | 5 | 5 |
| Oh | 48 | **10** | **10**（A1g A2g Eg T1g T2g A1u A2u Eu T1u T2u） |
| Ih | 120 | 10 | 10 |

D4h 的 10 个类：E 2C4 C2 2C2' 2C2'' i 2S4 σh 2σv 2σd。
Oh 的 10 个类：E 8C3 6C2 6C4 3C2(=C4²) i 6S4 8S6 3σh 6σd。

**S4 `reduce_representation(point_group, chars)`**
`chars` 按该点群类的顺序给。
返回：`{decomposition:[{irrep, count, contribution}], sum_check, dimensions_check, steps}`。
steps 必须写出 n_i = (1/h)Σ g_c·χ_i(c)·χ_r(c) 的逐项代入。

**S5 `normal_modes(point_group, n_atoms, is_linear=False)`**
3N 自由度分解：Γ_3N = Γ_trans + Γ_rot + Γ_vib。
返回：`{total_dof, gamma_3N, gamma_trans, gamma_rot, gamma_vib, ir_active:[...],
raman_active:[...], silent:[...], steps}`。
选律：IR 活性 = 与 x,y,z 同不可约表示；Raman 活性 = 与二次函数同表示。

**S6 `selection_rules(point_group)`**
返回：`{dipole:{has:bool, direction:"沿 C2 轴"/"无"/..., reason}, chiral:{has:bool, reason},
allowed_ir_basis:[...], allowed_raman_basis:[...], steps}`。
判据：无 i 且无 σ → 可能有旋光性；有 Cn 或 Cs 或 Cnv 有偶极矩（Cn/Cnv 沿主轴，
Cs 在镜面内，C1 任意），有 i 或 D/ T/ O/ I 系无偶极矩。

**S7 `compose_operations(point_group, op1, op2)`**
操作乘法。返回等价操作名 + 3×3 矩阵 + 行列式（+1 真转动 / -1 含反演或反映）。
供「证明 C2(z)σxy = i」类题用。

### 3.3 orbital.py [orbital-dev]

**O1 `huckel(topology, n_electrons=None, n_atoms=None, alpha=0.0, beta=-1.0)`**
`topology` 支持：`"linear"`（共轭多烯链）、`"cyclic"`（环）、`"ring"` + 自定义邻接矩阵
`adj:[[0,1,1],[1,0,1],[1,1,0]]`。

**符号约定（必须写死，教材里有两套等价写法，混用就是反号）**：
取 **x ≡ (E − α)/β**，久期方程 **|xδ_ij − A_ij| = 0**，于是 x 就是邻接矩阵的本征值，
**E = α + xβ**。教材另一套是 y ≡ (α − E)/β、|yδ_ij + 1| = 0、E = α − yβ，两者 y = −x。
实现取 x 口径（数值最干净），`steps` 里可同时给两种多项式。**β 必须 < 0**（传正数报错），
否则能级高低整体反号。

返回：`{orbitals:[{index, E_alpha_beta:"α+1.414β", E_beta:1.414, coeffs:[...], occupation}],
pi_energy_total, delocalization_energy, bond_orders:[{i,j,order,order_str}],
charge_densities:[...], free_valence:[...], degeneracies, steps}`。
自由价 F_r = √3 − Σ_s P_rs（教材口径；用精确 √3，steps 里注明与 1.732 取值的第三位差）。
离域能 = 总 π 能 − 同数孤立双键 ×2β（苯 = 2β，丁二烯 = 0.472β）。
steps 要写出久期行列式与解出的 x 值。

**O2 `aromaticity(n_pi, n_atoms, ring=True)`**
4n+2 判定。返回 `{rule, aromatic, n, reason, examples, steps}`。
含 4n 反芳香、非平面非芳香的反例说明。

**O3 `diatomic_mo(n_electrons=None, period=2, homo=None)`**
双原子分子 MO 组态。`n_electrons` 按**总电子数**收（O₂=16、N₂=14、NO=15、F₂=18），
函数内部折算价电子数，并在 `steps` 里写明折算过程；也接受显式 `n_valence_electrons`。
**物种名只由电子数决定**（`species_source: "electron_count"`）；`homo` 默认 `None`，
只作对照标签，**绝不允许用默认值把物种名钉死成 O2**（曾经踩过：`n_electrons=14` 返回
`species="O2"`，而 SKILL 会把 species 写进答案）。价电子数无唯一解时 `species=None` +
`species_candidates`，组态与键级照常给。完全无参调用 → `ok=False`（输入不足不猜）。
**能级顺序两套，方向别写反**（教材 OCR p129–134 依据）：

| 分子 | 判据 | 顺序 |
|---|---|---|
| N₂ 及更轻（Li₂ Be₂ B₂ C₂ N₂，价电子 ≤10） | 2s–2p 混杂强，σ2pz 被推高 | σ2s < σ*2s < **π2p < σ2pz** < π*2p < σ*2pz |
| O₂ F₂ Ne₂（价电子 ≥11） | 混杂弱，σ2pz 落回 π2p 之下 | σ2s < σ*2s < **σ2pz < π2p** < π*2p < σ*2pz |

即：**N₂ 及以前 σ2p 高于 π2p；O₂、F₂ 反过来**。
N₂ 组态写作 (σ2s)²(σ*2s)²(π2p)⁴(σ2pz)²；O₂ 写作 (σ2s)²(σ*2s)²(σ2pz)²(π2p)⁴(π*2p)²。
返回：`{config, bond_order, unpaired_electrons, magnetism, mo_energies_schematic,
bond_length_trend, steps}`。

**O4 `vsepr(central, ligands, lone_pairs=0, ligand_charges=None)`**
`central="S", ligands=["F","F","F","F"], lone_pairs=1`。
返回：`{steric_number, vsepr_type:"AX4E", geometry, hybrid:"sp3d", bond_angles:"≈89°",
point_group, examples, steps}`。

### 3.4 crystal.py [crystal-dev]

**C1 `unit_cell(a, b, c, alpha=90, beta=90, gamma=90)`**
返回：`{crystal_system, bravais, volume_pm3, volume_A3, steps}`。用晶系判据（边长相等关系+角关系）。

**C2 `miller_indices(intercepts)`**
截距 → Miller 指标（倒数化简约）。`intercepts=["a/2","b/3","c/4"]` 或 `[0.5, 0.3333, 0.25]` 或 `["1/2","1/3","1/4"]`。
返回：`{h,k,l, miller:"(234)", steps}`。含负指数与 ∞（截距为 0 → 指数 0）处理。

**C3 `d_spacing(h,k,l, a,b,c, alpha,beta,gamma)`**
返回：`{d_pm, d_A, formula_used, steps}`，各晶系用对应公式（立方/四方/正交/六方/单斜）。

**C4 `packing(kind)`**
`kind` ∈ {`fcc`,`hcp`,`bcc`,`sc`,`diamond`}。
返回：`{atoms_per_cell, coordination_number, packing_fraction:"74.05%", r_vs_a:"r=√2a/4",
voids:{octahedral, tetrahedral}, examples, steps}`。

**hcp 口径（易错，必须给两个数）**：`atoms_per_cell = 2`（菱形棱柱原胞，Z=2），
另给 `atoms_per_hex_prism = 6`（六方棱柱 = 3 个原胞）。理由：只有「2 原子 + 原胞体积
√2·a³（理想 c/a=√(8/3)）」才得出 74.05%；按 6 原子/六方棱柱算会得到 2.22 > 1，物理不可能。
`notes` 要写明「74.05% 仅在理想 c/a 下成立，实测 c/a 偏离时空间利用率随之改变」——
教材题常给非理想 c/a，那时不能照抄 74.05%。

**C5 `radius_ratio(r_plus_pm, r_minus_pm)`**
返回：`{ratio, coordination_number, polyhedron, range:"0.414–0.732", steps}`。
临界值：<0.155 三角(2)、0.155–0.225 三角(3)、0.225–0.414 四面体(4)、
0.414–0.732 八面体(6)、0.732–1.0 立方体(8)、>1.0 立方体(12)。

**C6 `structure_factor_count(formula, cell_content, ...)` → 归入 C7**
改为 **C6 `cell_content(formula, a_pm, density_gcm3=None, z=None, r_plus=None, r_minus=None)`**
反算/正算晶胞内容物数量 Z、密度、空间利用率。返回 `{Z, density, packing_fraction, steps}`。

### 3.5 coordination.py [crystal-dev]

**D1 `crystal_field(d_n, geometry="octahedral", pairing_energy_cm1=None, metal=None, ligand=None)`**
返回：`{d_configuration, splitting:{t2g,eg or e,t2}, delta_o_cm1, cfse_dq, cfse_cm1,
spin_state:"高自旋"/"低自旋", unpaired_electrons, magnetic_moment_BM, magnetism, steps}`。
CFSE 用 Dq 单位（八面体：-0.4n(t2g)+0.6n(eg)）；含配对能时比较。
磁矩 μ=√(n(n+2)) BM。

**D2 `spectrochemical_series(ligand)`**
返回配体在光谱化学序列中的位置 + Δo 相对大小 + 例子。

## 4. 验收断言（每个模块自测块必须含这些）

见 `tests/contract_cases.json`（Lead 提供）。摘录关键几条：

| 工具 | 输入 | 期望 |
|---|---|---|
| particle_box | n=3, L=100 pm | E3/E1 = 9 |
| hydrogen_orbital | n=3,l=1 | 径向节点 1、角节面 1、总节面 2、E=-1.511 eV |
| spectroscopic_terms | "d2" | 5 个项：1G 3F 1D 3P 1S，微观状态总数 45，基项 3F |
| spectroscopic_terms | "3d6"（Fe 基态） | 基项 5D4 |
| point_group | CH2Cl2 (C2v) | C2v |
| point_group | 苯 C6H6 | D6h |
| point_group | SF6 | Oh |
| point_group | 丙二烯 CH2=C=CH2 | D2d |
| point_group | H2O2（扭曲） | C2 |
| character_table | D4h | 10 个类、阶 16、8 个不可约表示 |
| normal_modes | C2v, n=3 (H2O) | Γ_vib = 2A1 + B1，全 IR+Raman 活性 |
| normal_modes | Oh, n=7 (SF6) | Γ_vib = A1g+Eg+2T1u+T2g+T2u |
| selection_rules | C2v | 有偶极矩（沿 C2），无旋光 |
| huckel | 环丙烯正离子 3 元环 2π | 能级 α+2β, α-β, α-β；离域能 2β |
| huckel | 苯 6 元环 6π | 能级 α+2β, α+β, α+β, α-β, α-β, α-2β |
| diatomic_mo | O2 (16 e) | 键级 2，2 个单电子，顺磁 |
| diatomic_mo | N2 (14 e) | 键级 3，抗磁 |
| vsepr | SF4 | AX4E，跷跷板，C2v |
| miller_indices | a/2,b/3,c/4 | (234) |
| packing | fcc | 4 原子/胞，CN=12，74.05% |
| radius_ratio | r₊=133, r₋=**195** | 133/195=0.682 → 八面体 CN=6（KBr 真题：键长 328 须先减出 r₋） |
| radius_ratio | r₊=133, r₋=328 | 0.4055 → 四面体 CN=4（**不是**八面体；这条用来锁住边界） |
| crystal_field | d6 oct, Δo=10000, P=21000 | 弱场 → 4 个单电子，μ=4.90 BM |

## 5. 协作纪律

1. 只写自己写域内的文件。需要改别人文件 → 发消息给 Lead。
2. 每完成一个函数就自己跑 `python -m structchem_mcp.core.<模块>` 自测，**绿了再交**。
3. 交付消息必须包含：文件路径、函数清单、自测输出摘要（贴关键行）、未达成项与原因。
4. 不写没有 steps 的工具。不返回自相矛盾的数值。不自造特征标表数值。
5. 遇到契约里没覆盖的教材细节，**按教材口径实现并在交付里标注**（例如 HMO 自由价用
   教材的 √3−ΣP 还是 1.732−ΣP，选一个并说明）。

## 5.5 已知限制（诚实清单，不假装通过）

1. **大群（O/T/I/Ih）不走「构造分子 → point_group」的端到端反验**。一般位置轨道要上百
   原子，代价不成比例。替代验证是「判据树 × 群元全量反验 39/39」+ 分子用例
   （含 Ih 正二十面体 12 顶点）。
2. **`_canonical_atoms` 只对部分 (点群, 原子数) 组合能凑出规范构型**。已支持：C2v 等常见群
   n≤13、Oh n=6/7/8/9/12/13、Td n=4/5/8/9/10/11、D2d/D4h/D6h n≥6。
   **T / Th 目前一律凑不出**（这两群在课程里几乎不出现）。凑不出时会返回可执行的错误，
   指路 `normal_modes(point_group, n_atoms, atoms=[...])` 直接给坐标 —— **不猜假数据**。
3. **`point_group` 的容差默认 0.15 Å**，内部对每个候选操作做 Kabsch 精修到最近正交变换。
   不做精修时偏斜轴会让群闭包爆炸（椅式环己烷从 12 个群元膨胀到 132 个），这个坑写在这里
   免得有人去掉精修。
4. **C∞v / D∞h 不支持 `reduce_representation`**（无限群），会明确报错并指路
   `normal_modes(is_linear=True)`，后者用 Σ/Π 计数法。
5. **`basis` 的二次函数归属曾有 5 格错**（D3d/S6 把二次函数放进 Eu、D4d 把 x²-y² 放进 B1、
   T/Th 的 E 格空着）。已用「真实群元算 Γ_quad 再约化」的方式定位并修正，勘误记在
   `character_tables.json` 的 `_meta`。这类错误的根因是手抄特征标表时把「基函数归属」
   当成文字抄，**必须用程序验**。
6. **题库的结构性缺口（重要）**：1681 题的题库**完全没有教材 4.6 节**（群的表示 / 特征标 /
   约化 / 3N−6 分解 / IR-Raman 选律）的题 —— `特征标` 命中 0、`不可约表示` 0、`表示矩阵` 0、
   `基函数` 0。所以 **题库题量分布 ≠ 教学重点分布**：第 4 章 141 题全砸在点群判定(82)与
   偶极矩旋光(40)，而权重最高的群表示块一题没有。
   这决定了 `sc_character_table` / `sc_reduce_representation` / `sc_normal_modes`
   这三个工具不是锦上添花，而是题库无法覆盖的那块唯一补给 —— 它们算不准，这一整块就没有答案。

## 6. Python 环境

- 解释器：`C:\Users\HL\AppData\Local\Programs\Python\Python310\python.exe`
- 已装：numpy 2.2.6、scipy 1.15.3、mcp（FastMCP 1.x）、pymupdf、rapidocr-onnxruntime
- 自测命令（模块内）：
  `cd D:\大肥鱼的实验室\dsh-structchem\src && python -m structchem_mcp.core.symmetry`
