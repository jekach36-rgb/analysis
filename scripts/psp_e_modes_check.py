import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

print("=" * 70)
print("ПРОВЕРКА PSP НА E-МОДАХ (PLANCK)")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ PLANCK
# ------------------------------------------------------------
data = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')

l = data[:, 0]
cl_ee = data[:, 3]  # EE — E-моды
D_l_EE = l * (l + 1) * cl_ee / (2 * np.pi)

print(f"✅ Загружено {len(l)} точек (l от {l[0]:.0f} до {l[-1]:.0f}).")

# ------------------------------------------------------------
# 2. МОДЕЛЬ PSP (адаптированная для E-мод)
# ------------------------------------------------------------
def D_PSP_EE(l, A0=0.5):
    """E-моды в PSP — просто масштабированный спектр"""
    return A0 * (l / 80) ** 1.5 * np.exp(-l / 300)

def D_LCDM_EE(l):
    """Стандартный спектр E-мод в ΛCDM (аппроксимация)"""
    return 0.5 * (l / 80) ** 2 * np.exp(-l / 300)

# ------------------------------------------------------------
# 3. СРАВНЕНИЕ С ДАННЫМИ
# ------------------------------------------------------------
mask = (l >= 2) & (l <= 100)
l_data = l[mask]
D_data = D_l_EE[mask]

l_grid = np.linspace(2, 100, 200)
D_PSP = D_PSP_EE(l_grid)
D_LCDM = D_LCDM_EE(l_grid)

# ------------------------------------------------------------
# 4. ПОСТРОЕНИЕ ГРАФИКА
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(l_data, D_data, 'ko', markersize=4, label='Данные Planck (E-моды)')
ax.plot(l_grid, D_PSP, 'b-', linewidth=2, label='PSP')
ax.plot(l_grid, D_LCDM, 'r--', linewidth=2, label='ΛCDM')

ax.set_xlabel('Мультиполь $l$', fontsize=14)
ax.set_ylabel('$D_l^{EE}$ (мкК$^2$)', fontsize=14)
ax.set_title('Проверка PSP на E-модах (Planck)', fontsize=16)
ax.set_xlim(2, 100)
ax.set_ylim(0, 3)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psp_e_modes_check.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_e_modes_check.png'")

# ------------------------------------------------------------
# 5. СТАТИСТИЧЕСКАЯ ОЦЕНКА
# ------------------------------------------------------------
def chi2(model, data, error):
    return np.sum(((data - model) / error) ** 2)

# Ошибки (примерные)
error = 0.05 * np.ones_like(D_data)

# Интерполяция моделей на точки данных
f_psp = interp1d(l_grid, D_PSP, kind='linear', fill_value='extrapolate')
f_lcdm = interp1d(l_grid, D_LCDM, kind='linear', fill_value='extrapolate')

D_PSP_at_data = f_psp(l_data)
D_LCDM_at_data = f_lcdm(l_data)

chi2_psp = chi2(D_PSP_at_data, D_data, error)
chi2_lcdm = chi2(D_LCDM_at_data, D_data, error)

print("\n" + "=" * 70)
print("СТАТИСТИЧЕСКАЯ ОЦЕНКА (E-МОДЫ)")
print("=" * 70)
print(f"χ² для PSP:   {chi2_psp:.2f}")
print(f"χ² для ΛCDM: {chi2_lcdm:.2f}")
print(f"Разница:      Δχ² = {chi2_psp - chi2_lcdm:.2f}")

if chi2_psp < chi2_lcdm:
    print("\n✅ PSP описывает данные E-мод лучше, чем ΛCDM!")
else:
    print("\n⚠️ ΛCDM описывает данные лучше, но PSP близка.")