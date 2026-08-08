#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT: FUGER LAW - RESIDUE CHEMICAL COMPOSITION (ENGLISH VERSION)
Author: E. V. Chernoknizhny
Date: August 7, 2026
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("PLOT: FUGER LAW - RESIDUE COMPOSITION (EN)")
print("="*70)

# ============================================================
# DATA: Chemical composition of remnants
# ============================================================

molecules = ['AlO', 'TiO', 'NaCl', 'SiO', 'Silicates', 'Fe', 'H/He']

# 1 = present, 0 = absent
n6946 = [1, 1, 1, 1, 1, 1, 1]          # N6946-BH1 (silent collapse, confirmed)
vycma = [1, 1, 1, 1, 1, 1, 1]           # VY CMa (active RSG, predicted silent collapse)
sn2023 = [0, 0, 0, 1, 0, 1, 1]          # SN 2023ixf (normal supernova progenitor)

# ============================================================
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(molecules))
width = 0.25

bars1 = ax.bar(x - width, n6946, width, label='N6946-BH1 (Remnant)', color='darkred', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, vycma, width, label='VY CMa (Active RSG)', color='orange', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, sn2023, width, label='SN 2023ixf (Normal Progenitor)', color='green', alpha=0.8, edgecolor='black')

ax.set_xlabel('Molecules', fontsize=14)
ax.set_ylabel('Detected (1 = Yes, 0 = No)', fontsize=14)
ax.set_title('Fuger Law: Chemical Composition of Remnants vs Progenitors', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(molecules, fontsize=12)
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['No', 'Yes'])

ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Anomalous region
ax.axvspan(x[0] - 0.5, x[2] + 0.5, color='red', alpha=0.1)
ax.text(x[1], 1.1, 'Anomalous\n(AlO, TiO, NaCl)', ha='center', va='bottom', fontsize=10, color='red')

# Annotations
ax.annotate('Silent collapse →\nAlO, TiO, NaCl, silicates',
            xy=(x[2], 1.0), xytext=(x[2] + 1.5, 0.8),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('Active mass loss →\nSame anomalous composition',
            xy=(x[2], 1.0), xytext=(x[2] + 1.5, 0.5),
            arrowprops=dict(facecolor='orange', shrink=0.05),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('Normal supernova →\nNo AlO, TiO, NaCl',
            xy=(x[2], 0), xytext=(x[2] - 2.5, -0.2),
            arrowprops=dict(facecolor='green', shrink=0.05),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

summary = (
    "N6946-BH1 and VY CMa share the same anomalous chemical signature:\n"
    "AlO, TiO, NaCl, silicates, Fe, H/He.\n"
    "This confirms the Fuger Law: anomalous composition → silent collapse.\n"
    "SN 2023ixf (normal composition) → supernova explosion."
)

ax.text(0.02, 0.02, summary,
        transform=ax.transAxes, fontsize=10, va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('fuger_law_residue_EN.png', dpi=150, bbox_inches='tight')
print("\n✅ Graph saved: fuger_law_residue_EN.png")

plt.show()

print("\n" + "="*70)
print("PLOT GENERATED SUCCESSFULLY!")
print("File: fuger_law_residue_EN.png")
print("="*70)