import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("ПРОВЕРКА PSP НА BAO (DESI)")
print("=" * 70)

# ------------------------------------------------------------
# 1. ДАННЫЕ DESI (BAO)
# ------------------------------------------------------------
# Примерные данные DESI для H(z)
z_desi = np.array([0.127, 0.468, 0.671, 0.830, 1.320, 2.330])
H_desi = np.array([71.33, 88.48, 119.45, 108.28, 147.58, 239.38])
H_err = np.array([4.20, 12.32, 16.64, 15.08, 4.49, 4.80])

print(f"✅ Загружено {len(z_desi)} точек BAO (DESI).")

# ------------------------------------------------------------
# 2. МОДЕЛЬ PSP
# ------------------------------------------------------------
H0 = 67.36
alpha = 0.15
beta = 0.35

def H_PSP(z):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

def H_LCDM(z):
    # Стандартная модель с Ωm = 0.315, ΩΛ = 0.685
    return H0 * np.sqrt(0.315 * (1 + z)**3 + 0.685)

z_grid = np.linspace(0.01, 3, 200)
H_PSP_vals = H_PSP(z_grid)
H_LCDM_vals = H_LCDM(z_grid)

# ------------------------------------------------------------
# 3. СРАВНЕНИЕ С ДАННЫМИ
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

ax.errorbar(z_desi, H_desi, yerr=H_err, fmt='ko', capsize=4, label='Данные DESI (BAO)')
ax.plot(z_grid, H_PSP_vals, 'b-', linewidth=2.5, label='PSP')
ax.plot(z_grid, H_LCDM_vals, 'r--', linewidth=2.5, label='ΛCDM')

ax.set_xlabel('Красное смещение $z$', fontsize=14)
ax.set_ylabel('$H(z)$ (км/с/Мпк)', fontsize=14)
ax.set_title('Проверка PSP на BAO (DESI)', fontsize=16)
ax.set_xlim(0, 3)
ax.set_ylim(50, 250)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psp_bao_check.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_bao_check.png'")