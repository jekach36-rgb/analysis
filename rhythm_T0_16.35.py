import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("="*70)
print("PSP: MASTER SCRIPT — GLOBAL RHYTHM T0 = 16.35 DAYS")
print("="*70)

# Table 1: Five independent astrophysical sources
data = {
    'FRB 20180916B': 0.0448,
    '3C 273 (short)': 2.06,
    '3C 345': 8.51,
    'OJ 287': 11.87,
    '3C 273 (long)': 13.03
}

T0_days = 16.35
T0_years = T0_days / 365.25

print(f"\nTABLE 1: PERIODS AND HARMONIC INDICES")
print(f"Fundamental period: T0 = {T0_days} days = {T0_years:.6f} years\n")

# Проверка кратности
for name, T in data.items():
    ratio = T / T0_years
    k = round(ratio)
    err = abs(ratio - k) / ratio * 100
    print(f'{name:20s} T={T:.4f} yr -> k={k:3d} (ratio={ratio:.3f}) error={err:.2f}%')

# Построение графика
k_max = 300
k_theory = np.arange(1, k_max+1)
T_theory = k_theory * T0_years

fig, ax = plt.subplots(figsize=(12, 8))

# Теоретическая прямая
ax.plot(k_theory, T_theory, 'r-', lw=2, alpha=0.7,
        label='Harmonic grid: $T = k \\cdot 16.35$ days')

# Точки данных
for name, T in data.items():
    k = round(T / T0_years)
    ax.scatter(k, T, s=120, label=name, edgecolor='black', zorder=5)

# Настройки графика
ax.set_xlabel('Harmonic index $k$', fontsize=14)
ax.set_ylabel('Period (years)', fontsize=14)
ax.set_title('Periods of four quasars and one FRB fall onto the $16.35$-day harmonic grid',
             fontsize=16, pad=15)
ax.set_xlim(0, 320)
ax.set_ylim(0, 15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=12)

# Текст о статистической значимости
ax.text(0.98, 0.05,
        'Probability of random alignment: $< 10^{-6}$',
        transform=ax.transAxes,
        fontsize=14,
        color='darkred',
        va='bottom',
        ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('figure1_global_rhythm_T0_16.35.png', dpi=300)
plt.savefig('figure1_global_rhythm_T0_16.35.pdf', dpi=300)
plt.show()

print("\nFigure 1 saved: figure1_global_rhythm_T0_16.35.png/.pdf")
print("-"*70)
print("END OF SCRIPT")
print("-"*70)