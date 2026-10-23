# -*- coding: utf-8 -*-
"""物理常数与单位换算（全插件唯一来源）。

来源：CODATA 2018 / 教材《结构化学基础》第 5 版附录口径。
所有模块必须从这里取常数，不要在业务代码里写魔法数字。
"""

from __future__ import annotations

# --- 基本常数（SI） ---
H = 6.62607015e-34          # Planck 常数 J·s（精确值）
HBAR = H / (2 * 3.141592653589793)  # 约化 Planck 常数 J·s
ME = 9.1093837015e-31       # 电子质量 kg
E_CHARGE = 1.602176634e-19  # 元电荷 C（精确值）
NA = 6.02214076e23          # Avogadro 常数 mol^-1（精确值）
C_LIGHT = 2.99792458e8      # 光速 m/s（精确值）
K_B = 1.380649e-23          # Boltzmann 常数 J/K（精确值）

# --- 原子单位 / 能量 ---
A0_PM = 52.9177210903       # Bohr 半径 pm
A0_M = 5.29177210903e-11    # Bohr 半径 m
HARTREE_EV = 27.211386245988  # 1 Hartree = 27.2114 eV
RYDBERG_EV = 13.605693122994  # 1 Ry = 13.6057 eV（= 氢原子基态能量绝对值）
RYDBERG_CM1 = 109737.31568160  # 1 Ry = 109737.3 cm^-1
EV_TO_CM1 = 8065.54400456   # 1 eV = 8065.544 cm^-1
EV_TO_KJMOL = 96.485332123  # 1 eV = 96.4853 kJ/mol
J_TO_EV = 1.0 / E_CHARGE

# --- 波尔磁子 ---
MU_B = 9.2740100783e-24     # J/T

# --- 常用换算 ---
PM_PER_A = 100.0            # 1 Å = 100 pm
NM_PER_PM = 1e-3            # 1 pm = 1e-3 nm


def hartree_to_ev(x: float) -> float:
    return x * HARTREE_EV


def ev_to_cm1(x: float) -> float:
    return x * EV_TO_CM1


def ev_to_kjmol(x: float) -> float:
    return x * EV_TO_KJMOL


def ev_to_nm(x: float) -> float:
    """能量(eV) → 对应光子波长(nm)。"""
    if x == 0:
        return float("inf")
    return 1239.841984 / x


def pm_to_a(x: float) -> float:
    return x / PM_PER_A


# 磁矩公式 μ = sqrt(n(n+2)) BM 的理论自由离子值
def spin_only_moment(n_unpaired: int) -> float:
    """自旋-only 磁矩（BM），n 为未成对电子数。"""
    return (n_unpaired * (n_unpaired + 2)) ** 0.5


__all__ = [
    "H", "HBAR", "ME", "E_CHARGE", "NA", "C_LIGHT", "K_B",
    "A0_PM", "A0_M", "HARTREE_EV", "RYDBERG_EV", "RYDBERG_CM1",
    "EV_TO_CM1", "EV_TO_KJMOL", "J_TO_EV", "MU_B",
    "PM_PER_A", "NM_PER_PM",
    "hartree_to_ev", "ev_to_cm1", "ev_to_kjmol", "ev_to_nm",
    "pm_to_a", "spin_only_moment",
]
