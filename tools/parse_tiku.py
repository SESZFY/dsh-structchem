# -*- coding: utf-8 -*-
"""把《结构化学题库》解析成结构化题库 JSON。

PDF 结构（内置 TOC 已确认）：
    p1-17   第一章习题      p18-50   第一章习题答案
    p51-72  第二章习题      p73-96   第二章习题答案
    ... 每章都是「习题页 + 紧随其后的答案页」
题号是 4 位数字，按章分块：1xxx / 2xxx / ... / 9xxx（第九章 9001-94xx，第十章 9501 起）。

答案就在同一份 PDF 里，不需要外部答案。公式部分提取会碎（PDF 排版所致），
因此**原样保留**并在 note 里标注，不做「猜测性还原」。
"""

from __future__ import annotations

import json
import os
import re

import pymupdf

PDF = r"C:\Users\HL\Downloads\结构化学题库.pdf"
OUT_DIR = r"D:\大肥鱼的实验室\_structchem_work\knowledge"

# (章号, 标题, 习题目录范围 lo, hi, 习题起始页, 习题结束页, 答案起始页, 答案结束页)
# 题号按章分块，但**第九章与第十章共用 9xxx**（九章 9001-94xx，十章 9501 起），
# 所以题号区间必须显式给出，不能靠 ch*1000 推。
CH = [
    (1, "量子力学基础知识", 1001, 1999, 1, 17, 18, 50),
    (2, "原子的结构和性质", 2001, 2999, 51, 72, 73, 96),
    (3, "共价键和双原子分子的结构化学", 3001, 3999, 97, 112, 113, 131),
    (4, "分子的对称性", 4001, 4999, 132, 140, 141, 149),
    (5, "多原子分子的结构化学", 5001, 5999, 150, 165, 166, 182),
    (6, "配位化合物的结构化学", 6001, 6999, 183, 190, 191, 197),
    (7, "晶体的点阵结构和晶体的性质", 7001, 7999, 198, 210, 211, 222),
    (8, "金属的结构和性质", 8001, 8499, 223, 228, 229, 236),
    (9, "离子化合物的结构化学", 9001, 9499, 237, 244, 245, 252),
    (10, "次级键及超分子结构化学", 9501, 9999, 253, 255, 256, 260),
]

# 题号在 PDF 里可能出现在**行中**（答案区常年一行挤 3-5 题：
# "4001 C3+i; C3+σh 4002 (非) 4003 (非) 4004 不对"），所以不能只 match 行首。
NUM = re.compile(r"(?<![\d.])(\d{4})(?![\d.])")


def pages_text(doc, a: int, b: int) -> str:
    out = []
    for p in range(a, b + 1):
        out.append(doc[p - 1].get_text())
    return "\n".join(out)


def clean(txt: str) -> str:
    """压掉 PDF 排版噪声：多余空白、连字符、孤立符号行。"""
    t = txt.replace("\u3000", " ").replace("\xa0", " ")
    # 中文之间的软换行直接去掉（PDF 一行没写完）
    t = re.sub(r"([\u4e00-\u9fff，。；：、）])\s*\n\s*([\u4e00-\u9fff（])", r"\1\2", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def split_by_id(text: str, lo: int, hi: int) -> dict:
    """按 4 位题号切块（题号可在行中）。落在 [lo,hi] 且总体递增，挡住正文里的数值。"""
    hits = []
    for m in NUM.finditer(text):
        n = int(m.group(1))
        if lo <= n <= hi:
            hits.append((m.start(), m.end(), n))
    # 单调性过滤：题号应基本递增；允许小幅回退（PDF 跨栏），剔除倒退过多的
    kept, last = [], 0
    for st, en, n in hits:
        if n >= last or n - last <= 5:
            kept.append((st, en, n))
            last = max(last, n)
    blocks: dict = {}
    for k, (st, en, n) in enumerate(kept):
        end = kept[k + 1][0] if k + 1 < len(kept) else len(text)
        body = text[en:end].strip()
        if n in blocks and len(blocks[n]) >= len(body):
            continue
        blocks[n] = body
    return blocks


def main() -> None:
    doc = pymupdf.open(PDF)
    os.makedirs(OUT_DIR, exist_ok=True)

    items = []
    stats = []
    for ch, title, lo, hi, e0, e1, k0, k1 in CH:
        ex = split_by_id(pages_text(doc, e0, e1), lo, hi)
        key = split_by_id(pages_text(doc, k0, k1), lo, hi)
        got = sorted(set(ex) & set(key))
        only_ex = sorted(set(ex) - set(key))
        for n in sorted(ex):
            stem = clean(ex[n])
            ans = clean(key.get(n, ""))
            if len(stem) < 4:
                continue
            qtype, tool = classify(stem, ans)
            items.append({
                "id": str(n),
                "chapter": ch,
                "chapter_title": title,
                "type": qtype,
                "stem": stem,
                "answer": ans or None,
                "source": f"《结构化学题库》第{ch}章 题号 {n}",
                "answer_noise": bool(ans) and _noisy(ans),
                "tool": tool,
                "note": (
                    "答案区未匹配到该题号（可能跨页截断）" if not ans
                    else ("答案含公式/矩阵，PDF 提取会把符号拆成碎片，"
                          "请按化学常识重组，不要照抄碎片" if _noisy(ans) else None)
                ),
            })
        stats.append({
            "chapter": ch, "title": title,
            "exercises": len(ex), "answers": len(key),
            "matched": len(got), "exercise_only": only_ex,
        })

    payload = {
        "_meta": {
            "source": "《结构化学题库》.pdf（260 页，原生文字层，非扫描件）",
            "parsed_by": "tools/parse_tiku.py",
            "note": "题干与答案均为 PDF 原文提取；含公式/矩阵的条目在 answer_noise=True，"
                    "排版会把公式拆成碎片，作答时请按化学常识重组，不要照抄碎片。",
            "chapter_stats": stats,
        },
        "items": items,
    }
    path = os.path.join(OUT_DIR, "question-bank.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("题号总数 =", len(items))
    for s in stats:
        print(f"  第{s['chapter']:2d}章 {s['title'][:14]:16s} 题 {s['exercises']:3d} "
              f"答案 {s['answers']:3d} 匹配 {s['matched']:3d}"
              + (f"  仅题无答 {s['exercise_only']}" if s["exercise_only"] else ""))
    with_ans = sum(1 for i in items if i["answer"])
    noisy = sum(1 for i in items if i["answer_noise"])
    print(f"\n有答案 {with_ans}/{len(items)}；答案含排版噪声 {noisy}")
    print("WROTE", path)


def _noisy(ans: str) -> bool:
    """答案里符号碎片占比高 → 判定为排版噪声。"""
    if not ans:
        return False
    if len(ans) < 12:
        return False
    tokens = ans.split()
    short = sum(1 for t in tokens if len(t) <= 2)
    return len(tokens) >= 4 and short / len(tokens) > 0.6


# 题型 → (归因关键词, 题型名, 对应工具)。顺序即优先级：越具体的放前面，
# 否则「振动」会把含「对称」的振动题抢走、「点群」会把晶系题抢走。
RULES = [
    (r"特征标|不可约表示|基函数|归(属|于).*表示", "特征标与表示", "sc_character_table"),
    (r"约化|可约表示|分解为不可约", "约化表示", "sc_reduce_representation"),
    (r"3N|3N-6|3N-5|振动(模式|自由度|方式)|红外活性|拉曼活性|IR ?活性|Raman", "振动分解与选律", "sc_normal_modes"),
    (r"偶极矩|极性分子|有无极性|旋光(性|异构)?", "偶极矩与旋光性", "sc_selection_rules"),
    (r"点群|对称元素|对称操作|熊夫|Schoenflies|所属.*群|群的阶", "点群判定", "sc_point_group"),
    (r"C2\(z\)|σ\(xy\)|操作(乘积|相乘|乘法)|证明.{0,6}=", "群论演算", "sc_compose_operations"),
    (r"Schrödinger|Schrodinger|薛定谔|波动方程|哈密顿|Hˆ", "写 Schrödinger 方程", "sc_hydrogen_orbital"),
    (r"势箱|势阱|一维箱|三维箱|共轭多烯.{0,8}吸收|箱中粒子", "势箱粒子", "sc_particle_box"),
    (r"量子数|节面|径向分布|角度分布|最可几|平均半径|原子轨道|概率密度|电子云", "类氢原子状态", "sc_hydrogen_orbital"),
    (r"Slater|屏蔽|有效核电荷|钻穿|Z\*", "Slater 规则", "sc_slater"),
    (r"光谱项|光谱支项|基项|基谱项|微观状态|Hund|洪特|L-S|耦合", "光谱项与基项", "sc_spectroscopic_terms"),
    (r"叠加|线性组合|平均值|期望值|力学量的平均", "波函数叠加", "sc_superposition"),
    (r"HMO|休克尔|Hückel|Huckel|久期|分子图|键级|电荷密度|自由价|离域能", "HMO 与分子图", "sc_huckel"),
    (r"芳香性|芳香族|4n\+2|Hückel 规则|反芳香", "芳香性判定", "sc_aromaticity"),
    (r"双原子分子.{0,10}(组态|轨道)|分子轨道.{0,6}组态|键级|顺磁|抗磁|未成对电子|分子轨道能级", "双原子分子 MO", "sc_diatomic_mo"),
    (r"VSEPR|价层电子对|杂化|空间构型|分子构型|键角|几何构型", "VSEPR 与杂化", "sc_vsepr"),
    (r"晶系|晶胞参数|布拉威|布拉维|点阵型式|空间群|格子", "晶系与点阵", "sc_unit_cell"),
    (r"晶面(指标|符号)|Miller|密勒|截距|\(hkl\)|晶面指数", "晶面指标", "sc_miller_indices"),
    (r"面间距|衍射|Bragg|布拉格|sinθ|d\(hkl\)|衍射指标", "晶面间距与衍射", "sc_d_spacing"),
    (r"堆积|密堆积|空间利用率|空隙|四面体空隙|八面体空隙|配位数", "堆积与空隙", "sc_packing"),
    (r"半径比|r\+/r-|配位多面体|离子半径比", "半径比与配位多面体", "sc_radius_ratio"),
    (r"结构基元|晶胞中.{0,8}(数|个)|密度.{0,10}晶胞|Z ?=", "晶胞内容物", "sc_cell_content"),
    (r"晶体场|分裂能|CFSE|稳定化能|高自旋|低自旋|自旋态|磁矩|光谱化学序列|配位场", "晶体场与磁性", "sc_crystal_field"),
    (r"氢键|次级键|超分子|范德华|堆积.力|包合物", "次级键与超分子", "sc_rationale"),
]


def classify(stem: str, ans: str):
    """按题干（必要时题干+答案）关键词打题型与工具标签。"""
    blob = stem + " " + (ans or "")
    for pat, name, tool in RULES:
        if re.search(pat, blob, re.I):
            return name, tool
    # 兜底：按提问方式粗分
    if re.search(r"计算|求|试算|列式", stem):
        return "计算题（未自动归类）", None
    if re.search(r"试述|说明|解释|为什么|为何|简述|比较", stem):
        return "解释题（未自动归类）", None
    if re.search(r"\(\s*[A-D]\s*\)|（\s*[A-D]\s*）", stem):
        return "选择题（未自动归类）", None
    return "概念题（未自动归类）", None


if __name__ == "__main__":
    main()
