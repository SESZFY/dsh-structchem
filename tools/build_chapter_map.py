# -*- coding: utf-8 -*-
"""生成 chapter-map.md —— 十章考点地图。

数据来源（全部本地、可追溯）：
  * src/structchem_mcp/data/book_index.json  —— 教材章节 / 考点 / 概念 / 公式 / 工具
  * _structchem_work/knowledge/question-bank.json —— 题库统计（题量、题型、题号区间）

不做任何「凭印象补写」：地图里每个数字都来自上面两份文件。
"""

from __future__ import annotations

import collections
import json
import os

ROOT = r"D:\大肥鱼的实验室"
BOOK = os.path.join(ROOT, "dsh-structchem", "src", "structchem_mcp", "data",
                    "book_index.json")
BANK = os.path.join(ROOT, "_structchem_work", "knowledge", "question-bank.json")
OUT = os.path.join(ROOT, "_structchem_work", "knowledge", "chapter-map.md")


def _num(item) -> int:
    """题号取整。curated 追加题可能是字符串 id（如 "4101a"），排在数字题之后，不抛异常。"""
    try:
        return int(str(item.get("id", "")).strip())
    except (TypeError, ValueError):
        return 10 ** 9


def main() -> None:
    book = json.load(open(BOOK, encoding="utf-8"))
    bank = json.load(open(BANK, encoding="utf-8"))
    items = bank["items"]

    by_ch = collections.defaultdict(list)
    for it in items:
        by_ch[it["chapter"]].append(it)

    lines: list[str] = []
    w = lines.append

    w("# 《结构化学基础》十章考点地图")
    w("")
    w("> 由 `tools/build_chapter_map.py` 从 `book_index.json` 与 `question-bank.json` 生成，")
    w("> 每个数字都可回溯到源文件。题库是《结构化学题库》（260 页原生文字层 PDF，")
    w("> 10 章 1654 题，其中 1609 题带标准答案）。")
    w("")
    w("## 全书概览")
    w("")
    total = len(items)
    with_ans = sum(1 for i in items if i["answer"])
    noisy = sum(1 for i in items if i["answer_noise"])
    w(f"- 题库总量 **{total} 题**，带答案 **{with_ans} 题**"
      f"（{with_ans / total * 100:.1f}%），其中 {noisy} 题的答案含 PDF 排版碎片"
      f"（公式/矩阵被拆散，需按化学常识重组）。")
    w(f"- 十章题量分布：" + "、".join(
        f"第{c}章 {len(by_ch[c])} 题" for c in sorted(by_ch)))
    w("")
    w("| 章 | 标题 | 题量 | 有答案 | 题号区间 | 主力题型（前 3） |")
    w("|---|---|---|---|---|---|")
    for ch in sorted(by_ch):
        its = by_ch[ch]
        ids = sorted(_num(i) for i in its)
        top = collections.Counter(i["type"] for i in its).most_common(3)
        w(f"| {ch} | {its[0]['chapter_title']} | {len(its)} | "
          f"{sum(1 for i in its if i['answer'])} | {ids[0]}–{ids[-1]} | "
          + "、".join(f"{t}({c})" for t, c in top) + " |")
    w("")

    for ch in sorted(by_ch):
        its = by_ch[ch]
        meta = next((c for c in book["chapters"] if c.get("no") == ch), None)
        title = its[0]["chapter_title"]
        w(f"## 第 {ch} 章　{title}")
        w("")
        w(f"**题量** {len(its)} 题，题号 {min(_num(i) for i in its)}–"
          f"{max(_num(i) for i in its)}；"
          f"带答案 {sum(1 for i in its if i['answer'])} 题。")
        w("")

        if meta:
            secs = meta.get("sections") or []
            if secs:
                w("**教材节次**")
                w("")
                for s in secs:
                    kp = "；".join((s.get("key_points") or [])[:3])
                    w(f"- `{s.get('no')}` {s.get('title')}" + (f" —— {kp}" if kp else ""))
                w("")
            concepts = meta.get("concepts") or {}
            if concepts:
                w("**必考概念**（一句可抄的定义）")
                w("")
                for k, v in list(concepts.items())[:14]:
                    w(f"- **{k}**：{v}")
                w("")
            formulas = meta.get("formulas") or []
            if formulas:
                w("**必背公式**")
                w("")
                for f in formulas[:16]:
                    w(f"- `{f}`")
                w("")

        types = collections.Counter(i["type"] for i in its)
        w("**题型分布**（题库实测）")
        w("")
        for t, c in types.most_common():
            ex = next((i["id"] for i in its if i["type"] == t), "")
            w(f"- {t}：{c} 题" + (f"，例 {ex}" if ex else ""))
        w("")

        tools = collections.Counter(i["tool"] for i in its if i["tool"])
        if tools:
            w("**对应插件工具**")
            w("")
            for t, c in tools.most_common():
                w(f"- `{t}` —— 覆盖 {c} 题")
            if meta and (meta.get("tools")):
                extra = [t for t in meta["tools"] if t not in tools]
                if extra:
                    w(f"- 教材标注、题库未命中：`{'`、`'.join(extra)}`")
            w("")

        # 每章挑 3 道代表题（有答案、非噪声、题干清晰）
        cands = [i for i in its
                 if i["answer"] and not i["answer_noise"] and 20 <= len(i["stem"]) <= 160]
        if cands:
            w("**代表题**（题干与答案均无排版噪声）")
            w("")
            for i in cands[:3]:
                w(f"- `{i['id']}`（{i['type']}）{i['stem'][:100]}")
                w(f"  - 答：{i['answer'][:100]}")
            w("")
        w("---")
        w("")

    # 附：全书题型 × 工具矩阵
    w("## 附：题型 → 工具 → 题量（全书）")
    w("")
    w("| 题型 | 工具 | 题量 |")
    w("|---|---|---|")
    pair = collections.Counter((i["type"], i["tool"] or "—") for i in items)
    for (t, tl), c in pair.most_common():
        w(f"| {t} | `{tl}` | {c} |")
    w("")
    un = sum(1 for i in items if not i["tool"])
    w(f"未自动归到工具的题：{un} 题（{un / total * 100:.1f}%）。这类题多为"
      "概念辨析与综合计算，按章节与关键词检索，不必强行映射到单一工具。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE", OUT, f"({len(lines)} 行)")


if __name__ == "__main__":
    main()
