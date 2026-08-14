import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PSP vs PLANCK: РЕАЛЬНЫЕ ДАННЫЕ B-МОД")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ PLANCK (ТВОЙ ФАЙЛ)
# ------------------------------------------------------------
data = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')

# Колонки: L, TT, TE, EE, BB, PP
l = data[:, 0]
cl_bb = data[:, 4]  # BB — это B-моды

# Переводим C_l в D_l = l(l+1)C_l / 2π
D_l_BB = l * (l + 1) * cl_bb / (2 * np.pi)

# Берём только первые 100 мультиполей (где важна твоя модель)
mask = l <= 100
l_planck = l[mask]
D_l_BB_planck = D_l_BB[mask]

print(f"✅ Загружено {len(l_planck)} точек данных (l от {l_planck[0]:.0f} до {l_planck[-1]:.0f}).")

# ------------------------------------------------------------
# 2. МОДЕЛЬ PSP
# ------------------------------------------------------------
def M_PSP(l):
    return 0.29 + 0.21 * (1 - np.exp(-l / 350))

def D_l_BB_PSP(l, A0=0.012):
    M = M_PSP(l)
    return A0 * np.sin(np.pi * (M - 0.29) / 0.21)**2 * np.exp(-l / 300)

l_grid = np.linspace(2, 100, 200)
D_l_PSP = D_l_BB_PSP(l_grid)

# ------------------------------------------------------------
# 3. ПОСТРОЕНИЕ ГРАФИКА
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# Реальные данные Planck (чёрные точки)
ax.plot(l_planck, D_l_BB_planck, 'ko', markersize=6, label='Planck 2018 (реальные данные)', zorder=10)

# Модель PSP (синяя линия)
ax.plot(l_grid, D_l_PSP, 'b-', linewidth=2.5, label='PSP (предсказание)', zorder=5)

# Отметки пиков PSP
l_peaks = [74, 109]
for lp in l_peaks:
    if lp <= 100:
        ax.axvline(lp, color='blue', linestyle='--', alpha=0.4)
        ax.text(lp + 2, 0.001, f'l={lp}', fontsize=10, color='blue')

ax.set_xlabel('Мультиполь $l$', fontsize=14)
ax.set_ylabel('$D_l^{BB}$ (мкК$^2$)', fontsize=14)
ax.set_title('PSP vs Planck: реальные данные B-мод', fontsize=16)
ax.set_xlim(2, 100)
ax.set_ylim(-0.0001, 0.003)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('psp_vs_planck_real.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_vs_planck_real.png'")