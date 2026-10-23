# -*- coding: utf-8 -*-
"""structchem_mcp.core —— 结构化学课程答题的确定性计算层。

模块划分（与 ARCHITECTURE.md 第 3 节一一对应）：

    quantum       量子化学基础与原子结构   particle_box / hydrogen_orbital / slater
                                          / spectroscopic_terms / superposition
    symmetry      分子对称性与群论         point_group / symmetry_elements / character_table
                                          / reduce_representation / normal_modes
                                          / selection_rules / compose_operations
    orbital       分子轨道与化学键         huckel / aromaticity / diatomic_mo / vsepr
    crystal       晶体结构                 unit_cell / miller_indices / d_spacing
                                          / packing / radius_ratio / cell_content
    coordination  配位化合物               crystal_field / spectrochemical_series

导入策略：各子模块**按需惰性导入**并缓存可用性。任何一个模块缺失或损坏，
其余工具仍可用 —— 单点故障不该让整个插件熄火。
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULES = ("quantum", "symmetry", "orbital", "crystal", "coordination")

_cache: dict[str, Any] = {}
_errors: dict[str, str] = {}


def get_module(name: str) -> Any:
    """取子模块；不可用时抛 ImportError 并带上真实原因。"""
    if name in _cache:
        return _cache[name]
    if name in _errors:
        raise ImportError(_errors[name])
    try:
        mod = importlib.import_module(f"{__name__}.{name}")
    except Exception as exc:  # noqa: BLE001
        _errors[name] = f"{type(exc).__name__}: {exc}"
        raise ImportError(_errors[name]) from exc
    _cache[name] = mod
    return mod


def availability() -> dict:
    """每个子模块能否导入 + 若不能，原因是什么。供 --selftest 与排障用。"""
    out: dict[str, dict] = {}
    for name in _MODULES:
        try:
            mod = get_module(name)
        except ImportError as exc:
            out[name] = {"ok": False, "error": str(exc), "functions": []}
            continue
        funcs = [f for f in dir(mod) if not f.startswith("_") and callable(getattr(mod, f))]
        funcs = [f for f in funcs if getattr(getattr(mod, f), "__module__", "") == mod.__name__]
        out[name] = {"ok": True, "error": None, "functions": sorted(funcs)}
    return out


__all__ = ["get_module", "availability", "_MODULES"]
