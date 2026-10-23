# -*- coding: utf-8 -*-
"""统计真实语料里的数学符号/角标书写习惯，为「符号规范化」提供实测依据。

语料（三份，全部本地）：
  1. knowledge/question-bank.json   1681 题（题干+标准答案）
  2. web/_extract.txt               南开真题 + 实习答案
  3. ocr/exercise_pages.jsonl       教材第 5 版各章末习题 88 页 OCR

只统计「怎么写」，不判断对错；结论用于写 references/notation-and-typography.md。
"""
from __future__ import annotations

import collections
import json
import os
import re

K = r"D:\大肥鱼的实验室\_structchem_work"


def load_corpus() -> dict[str, str]:
    out = {}

    bank = json.load(open(os.path.join(K, "knowledge", "question-bank.json"),
                          encoding="utf-8"))
    out["bank_stem"] = "\n".join(i.get("stem") or "" for i in bank["items"])
    out["bank_answer"] = "\n".join(i.get("answer") or "" for i in bank["items"])

    p = os.path.join(K, "web", "_extract.txt")
    out["nankai"] = open(p, encoding="utf-8").read()

    p = os.path.join(K, "ocr", "exercise_pages.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    out["textbook_ocr"] = "\n".join(r.get("text") or "" for r in rows)
    return out


SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ"
SUBS = "₀₁₂₃₄₅₆₇₈₉₊₋"

# 点群记号：是否用下标数字
PG_SUB = re.compile(r"[CDSTOIch]\s*[₀-₉]")           # C₂v（Unicode 下标）
PG_PLAIN = re.compile(r"\b[CDSTO]\d[hvdi]\b")         # C2v（平排）
PG_SPECIAL = re.compile(r"[CDSTOIch][₂₃₄₅₆]")        # C₂v 无 v

GREEK = "αβγδεζηθικλμνξοπρστυφχψω"
GREEK_UP = "ΓΔΘΛΞΠΣΦΨΩ"
UNIT_CM = re.compile(r"cm\s*(?:[-⁻]\s*1|⁻¹|\^\{?-?1\}?)")
UNIT_PM = re.compile(r"\bpm\b")
UNIT_A = re.compile(r"Å|\bA\b(?=\s|$)")

SPECIAL = {
    "Σ(求和)": r"Σ",
    "⊕(直和)": r"⊕",
    "⊗": r"⊗",
    "σ*": r"σ\s*\*",
    "π*": r"π\s*\*",
    "→": r"→",
    "⇌": r"⇌",
    "≈": r"≈",
    "≠": r"≠",
    "≤": r"≤",
    "⊥": r"⊥",
    "∥": r"∥|\|\|",
    "√": r"√",
    "∫": r"∫",
    "∂": r"∂",
    "ħ": r"ħ|ℏ",
    "°(度)": r"°",
    "×": r"×",
    "·(中点)": r"·",
    "∼/~": r"∼|(?<!\d)~(?!\d)",
    "(hkl)": r"\(h\s*k\s*l\)",
    "[hkl]": r"\[h\s*k\s*l\]",
    "⟨⟩": r"[⟨〉〈〉]",
}

SPECTRAL_TERM = [
    ("³F₂ 全上标", re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹][SPDFGHIKL]\s*[₀-₉]")),
    ("3F2 全平排", re.compile(r"\b[1-9][SPDFGHIKL][0-9]\b")),
    ("3F 上标多重度", re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹][SPDFGHIKL]\b")),
    ("3F 平排多重度", re.compile(r"\b[1-9][SPDFGHIKL]\b")),
]

ORB_SUB = [
    ("t₂g / e_g（下标记法）", re.compile(r"[te]\s*[₂₃]\s*g|[te]₂g|e_g")),
    ("t2g / eg（平排）", re.compile(r"\bt2g\b|\beg\b")),
    ("t2g 带下标数字", re.compile(r"t\s*2\s*g")),
]

QN = [
    ("ψ₂₁₀ 全下标", re.compile(r"[ψφ]\s*[₀-₉]{2,}")),
    ("ψ210 平排", re.compile(r"[ψφ]\s*\d{3}\b")),
    ("ψ(n,l,m)", re.compile(r"[ψφ]\s*\(\s*n\s*,")),
]


def count(corpus: dict[str, str], pat, name: str) -> int:
    n = 0
    for txt in corpus.values():
        n += len(pat.findall(txt))
    return n


def main() -> None:
    c = load_corpus()
    print("语料规模：")
    for k, v in c.items():
        print(f"  {k:14s} {len(v):>9,d} 字符")
    print()

    print("=" * 68)
    print("一、点群记号")
    for name, pat in (("C₂v 式（Unicode 下标）", PG_SUB),
                      ("C2v 式（平排）", PG_PLAIN),
                      ("C₂ 式（无对称性后缀）", PG_SPECIAL)):
        print(f"  {name:26s} {count(c, pat, name)}")

    print()
    print("=" * 68)
    print("二、光谱项与原子项符号")
    for name, pat in SPECTRAL_TERM:
        print(f"  {name:26s} {len(pat.findall(c['bank_answer'] + c['bank_stem']))}  (题库)")
        print(f"  {'':26s} {len(pat.findall(c['nankai']))}  (南开)")

    print()
    print("=" * 68)
    print("三、d 轨道 / 分子轨道记号")
    for name, pat in ORB_SUB:
        print(f"  {name}: {count(c, pat, name)}")

    print()
    print("=" * 68)
    print("四、波函数角标")
    for name, pat in QN:
        print(f"  {name}: {count(c, pat, name)}")

    print()
    print("=" * 68)
    print("五、希腊字母出现频次（前 12）")
    allt = "\n".join(c.values())
    g = collections.Counter(ch for ch in allt if ch in GREEK + GREEK_UP)
    for ch, n in g.most_common(12):
        print(f"  {ch}  {n}")

    print()
    print("=" * 68)
    print("六、数学与化学符号")
    for name, pat in SPECIAL.items():
        n = len(re.findall(pat, allt))
        if n:
            print(f"  {name:12s} {n}")

    print()
    print("=" * 68)
    print("七、单位写法")
    for name, pat in (("cm⁻¹ / cm-1 类", UNIT_CM), ("pm", UNIT_PM), ("Å", UNIT_A)):
        print(f"  {name:16s} {count(c, pat, name)}")
    # cm 的几种写法细分
    for variant in (r"cm-1", r"cm⁻¹", r"cm\^\{?-1\}?", r"cm\s*−\s*1"):
        print(f"    cm 变体 {variant:18s} {len(re.findall(variant, allt))}")

    print()
    print("=" * 68)
    print("八、Unicode 上标/下标字符总量（判断语料是否已用富文本符号）")
    sup = sum(1 for ch in allt if ch in SUPS)
    sub = sum(1 for ch in allt if ch in SUBS)
    print(f"  上标字符 {sup}，下标字符 {sub}")

    print()
    print("=" * 68)
    print("九、几个高频记号的真实上下文（各取 3 条，看教材怎么排版）")
    for pat, label in ((r".{18}[¹²³⁴⁵⁶⁷⁸⁹][SPDFGHIKL].{10}", "光谱项上标"),
                       (r".{12}t\s*[₂₃]?\s*g.{12}", "t2g/eg"),
                       (r".{12}[CDST]\s*[₂₃₄]?\d?\s*[hvdi].{10}", "点群"),
                       (r".{14}cm\s*[-⁻]\s*1.{8}", "波数单位")):
        print(f"\n  [{label}]")
        for m in re.finditer(pat, allt):
            print("    ", m.group(0).replace("\n", " / ")[:64])
            if len([1]) and m.start() > len(allt) // 3:
                break


if __name__ == "__main__":
    main()
