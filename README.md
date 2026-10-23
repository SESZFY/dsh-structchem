# dsh-structchem

结构化学答题的**确定性工具层 + 作答规范层**。面向周公度·段连运《结构化学基础》（第 5 版）体系


---

## 它解决什么

结构化学的题有个特点——**判断题和计算题各占一半，而判断题最容易出错**：

- 点群判定（`C2v` 还是 `D2d`？）—— 靠背是不行的，异构体和构象一变就翻车；
- 3N−6 振动分解与 IR/Raman 活性 —— 靠查表，但特征标表十类十表示抄错一个数全错；
- HMO 久期行列式 —— 手解 6 阶行列式，系数与键级一路错到底；
- 晶体几何 —— 键长当离子半径用、空间利用率少除一个 Z，都是 0 分；
- 晶体场 —— 高/低自旋判反，磁矩跟着全错。

这个插件把这些**全部变成可计算的**，每条都给 `steps`（可直接抄进答题纸的逐步算式）。真正的化学判断（选哪个判据、图怎么画、话怎么说）交给 skill。

配套数据的来源（全部可追溯，无一凭空生成）：

- **题库** —— 《结构化学题库》260 页原生文字层 PDF，**10 章 1681 题，其中 1636 题带标准答案**
  （97.3%）。逐题解析为结构化 JSON，按题干关键词自动打「题型 → 工具」标签，覆盖率 64.6%；
  含公式/矩阵的答案会被 PDF 排版拆成碎片，已逐条标记 `answer_noise`，提示按化学常识重组。
- **教材** —— 《结构化学基础》第 5 版扫描件 414 页 OCR（10 章各章末习题区 88 页），
  与 PDF 内置目录逐页校核过偏移。
- **真实考卷** —— 南开大学结构化学试卷（含标准答案）与模型实习答案，
  skill 的「答案风格铁律」全部从这两份实测原文反推，不是印象。
- **考点地图** —— `_structchem_work/knowledge/chapter-map.md`（878 行），
  每章给出题号区间、题量、主力题型、必考概念/公式与对应工具，数字全部由脚本从上述两份文件生成。

---

## 安装

DSH 安装插件的本质是「往 profile 的依赖里加这个包，再让它作为 bundle 层装载」。

**A. 图形界面（最省事）**

桌面端「设置 → 插件」里填这个 spec：

```
github:SESZFY/dsh-structchem
```

**B. 命令行 / 直接改依赖**

```powershell
dsh plugin --profile desktop add github:SESZFY/dsh-structchem
```

等价于在 profile 的 `package.json`（`~/.dsh/profiles/desktop/package.json`）里加一行：

```json
"dependencies": {
  "dsh-structchem": "github:SESZFY/dsh-structchem"
}
```

然后在该 profile 目录 `pnpm install`。装完**重启 DSH** —— MCP 进程在启动时拉起，
`dsh-mcp-client` 没有热重载，换了代码不重启跑的还是旧版本。

**没有 git 的机器**：pnpm 解析 `github:` 依赖要调用 `git`（会转成 codeload 的 tarball，
但仍需 git 做 `ls-remote`）。没装 git 就把依赖换成 tarball URL：

```
https://codeload.github.com/SESZFY/dsh-structchem/tar.gz/refs/heads/main
```

### 先装 Python 依赖

```bash
pip install -r requirements.txt      # numpy + mcp(1.x)，就这两个
```

纯本地 numpy 计算，**零网络、零 API key**、无 RDKit 之类重依赖。

### 装完怎么验

让 agent 判一个点群（水分子应得 `C2v`、氨应得 `C3v`），或在任意会话里直接调用：

```
mcp__structchem__sc_point_group(
  atoms=[{"sym":"O","xyz":[0,0,0]},
         {"sym":"H","xyz":[75.7,0,0]},
         {"sym":"H","xyz":[-24.0,71.7,0]}])
```

返回 `point_group: "C2v"`、`order: 4` 就说明 MCP 起来了。工具与模块健康状态用
`mcp__structchem__sc_toolbox()` 自查。

> 计算工具**都不带 `impl` 参数**。如果你在 tools/list 里看到 `required: ["impl", ...]`，
> 说明 MCP 子进程仍在跑旧代码 —— 重启 DSH 即可，不必去改配置。

### bundle 层做了什么

`cordis.patch.yml` 会自动做两件事：

1. 把随包的 `skills/` 注册成独立技能扫描源（providerName `structchem`）；
2. 把 `src/` 里的 stdio MCP 服务挂进 MCP 客户端，模型侧出现 `mcp__structchem__sc_*` 工具。

环境变量：`STRUCTCHEM_PYTHON`（Python 可执行文件，默认 `python`，需 3.10+）。

---

## 30 个工具

### 量子化学基础与原子结构（`quantum`）

| 工具 | 做什么 |
|---|---|
| `sc_particle_box` | 一/二/三维势箱能级、简并度、零点能、跃迁波长 |
| `sc_hydrogen_orbital` | 类氢轨道：径向/角向/总节面、能量、\|L\|、Lz、最可几半径、⟨r⟩ |
| `sc_slater` | Slater 规则：屏蔽常数 σ、有效核电荷 Z\*、轨道能量（含分组贡献明细） |
| `sc_spectroscopic_terms` | L-S 耦合全部光谱项 + 微观状态数自校验 + Hund 基项与 J |
| `sc_superposition` | 波函数叠加：⟨E⟩、各本征值几率、⟨L²⟩、⟨Lz⟩ |

### 分子对称性与群论（`symmetry`）

| 工具 | 做什么 |
|---|---|
| `sc_point_group` | **点群判定**：从原子坐标真算——枚举检验全部对称操作后定 Schoenflies 记号 |
| `sc_symmetry_elements` | 点群 → 对称元素、操作清单、阶、类划分、生成元 |
| `sc_character_table` | 特征标表 + 基函数归属（x,y,z / Rx,Ry,Rz / 二次函数） |
| `sc_reduce_representation` | 可约表示约化，逐项代入 n_i=(1/h)Σg_c·χ_i·χ_r |
| `sc_normal_modes` | 3N 自由度分解 → Γ_vib → IR/Raman 活性（选律） |
| `sc_selection_rules` | 偶极矩（有无 + 方向）、旋光性判据 |
| `sc_compose_operations` | 对称操作乘法（C2(z)·σ(xy)=i 这类），矩阵 + 行列式 + 等价名 |

### 分子轨道与化学键（`orbital`）

| 工具 | 做什么 |
|---|---|
| `sc_huckel` | HMO 法真解久期行列式：能级、系数、键级、电荷密度、自由价、离域能 |
| `sc_aromaticity` | 4n+2 芳香性（区分芳香/反芳香/非芳香） |
| `sc_diatomic_mo` | 双原子分子 MO：组态、键级、未成对电子、磁性（两套能级顺序自动判定） |
| `sc_vsepr` | VSEPR：价层电子对数 → 构型、杂化、键角、点群 |

### 晶体结构（`crystal`）

| 工具 | 做什么 |
|---|---|
| `sc_unit_cell` | 晶胞参数 → 晶系判定、布拉维格子、体积 |
| `sc_miller_indices` | 截距 → 晶面指标（含 ∞ 与负指数） |
| `sc_d_spacing` | 晶面间距 d(hkl)，按晶系自动选公式 |
| `sc_packing` | 堆积（fcc/hcp/bcc/sc/diamond）：原子数、CN、空间利用率、空隙 |
| `sc_radius_ratio` | 半径比 → 配位数与配位多面体（含临界值区间） |
| `sc_cell_content` | 晶胞内容物：密度反算 Z、空间利用率 |

### 配位化合物（`coordination`）

| 工具 | 做什么 |
|---|---|
| `sc_crystal_field` | 晶体场：分裂、CFSE、Δo vs P 定高低自旋、未成对电子、磁矩 |
| `sc_spectrochemical_series` | 配体在光谱化学序列中的位置与 Δo 相对大小 |

### 知识检索（数据层）

| 工具 | 做什么 |
|---|---|
| `sc_answer_card` | 题型答题卡：作答步骤 / 抄写模板 / 采分点 / 陷阱 / 对应工具 |
| `sc_rationale` | 归因词典：把「为什么」落到可命名的原理（判据 / 模板 / 常见错答） |
| `sc_book_index` | 教材索引：章 → 节 → 考点 / 概念定义 / 公式 / 对应工具 |
| `sc_style_law` | 答案风格铁律（从真卷实测提炼，含原文证据） |
| `sc_notation` | 符号与排版规范：19 个多义希腊字母消歧、上下标双轨、单位与括号约定 |
| `sc_toolbox` | 工具清单 + 各模块健康状态（排障用） |

---

## 设计边界（先说清楚，避免当成万能解题器）

**做的**：可确定性计算的数值与判定 —— 点群、特征标、约化、分解、久期行列式、晶体几何、CFSE。

**不做的**：

- 不猜。输入不足就返回 `ok=False` + 中文原因（缺哪个原子、哪个参数），**绝不返回 plausible-but-wrong 的数值**；
- 不判化学意图。哪个判据适用、图怎么画、话怎么写 — 属于 skill 的方法论，插件只供数值与过程核对项；
- 不涉网络。无 API、无在线数据库。

每个计算工具都带 `steps`。`steps` 不是调试信息，是**给答题纸用的**——没有 steps 的工具视为未完成。

---

## 验收

```bash
python tests\test_contract.py                  # 8 组契约断言，退出码 0 = 全绿
python -m structchem_mcp.server --selftest     # 端到端自测（27 项，含各模块工具各打一发）
python -m structchem_mcp.core.symmetry         # 各模块独立自测
```

断言里的期望值全部来自教材标准答案、群论定理、或可手算校验的物理量——**不迁就实现**。
算不出来就改实现，不改断言。实际开发中这些断言抓出过 9 个真 bug，其中有 4 个是
「静默给错答案」而不是崩溃（最典型：SF6 被判成 C∞v、双原子分子物种名写死成 O2）。

群论数据另有一层硬校验：对全部 39 个有限点群逐对验证正交性
`Σ_c g_c·χ_i·χ_j = h·δ_ij`（教材合并表示取 `2h`）。这条恒等式挡掉任何一格手抄错。

---

## 目录

```
dsh-structchem/
├── ARCHITECTURE.md          # 接口契约（16 个计算工具的签名、返回字段、验收断言）
├── cordis.patch.yml         # DSH bundle patch
├── src/structchem_mcp/
│   ├── server.py            # MCP 工具注册 + 统一错误折返
│   ├── core/                # 计算层（纯函数、无 IO）
│   │   ├── _const.py        # 物理常数唯一来源
│   │   ├── quantum.py  symmetry.py  orbital.py  crystal.py  coordination.py
│   └── data/                # 特征标表 / 答题卡 / 归因词典 / 教材索引
├── skills/structural-chem-answering/
│   ├── SKILL.md             # 作答主线（五步法 + 18 类题型总表）
│   └── references/          # 八个分册
└── tests/
```

## License

MIT
