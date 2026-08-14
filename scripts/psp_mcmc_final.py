import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("АНАЛИЗ χ² ПО КОМПОНЕНТАМ")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------
data = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
l = data[:, 0]
cl_bb = data[:, 4]
cl_ee = data[:, 3]

D_BB = l * (l + 1) * cl_bb / (2 * np.pi)
D_EE = l * (l + 1) * cl_ee / (2 * np.pi)

mask = (l >= 2) & (l <= 100)
l_data = l[mask]
D_BB_data = D_BB[mask]
D_EE_data = D_EE[mask]

# Данные DESI (BAO)
z_desi = np.array([0.127, 0.468, 0.671, 0.830, 1.320, 2.330])
H_desi = np.array([71.33, 88.48, 119.45, 108.28, 147.58, 239.38])
H_err = np.array([4.20, 12.32, 16.64, 15.08, 4.49, 4.80])

# ------------------------------------------------------------
# 2. МОДЕЛЬ PSP (ИСХОДНАЯ)
# ------------------------------------------------------------
def M_PSP(l):
    return 0.29 + 0.21 * (1 - np.exp(-l / 350))

def D_BB_PSP(l, A0=0.012):
    M = M_PSP(l)
    return A0 * np.sin(np.pi * (M - 0.29) / 0.21)**2 * np.exp(-l / 300)

def D_EE_PSP(l, A1=0.5):
    return A1 * (l / 80)**1.5 * np.exp(-l / 300)

def H_PSP(z, H0=67.36, alpha=0.15, beta=0.35):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

# ------------------------------------------------------------
# 3. ВЫЧИСЛЕНИЕ χ² ПО КОМПОНЕНТАМ (С ЛУЧШИМИ ПАРАМЕТРАМИ)
# ------------------------------------------------------------
# Берём параметры из твоего MCMC (медианы)
A0_bb = 0.539
A1_ee = 0.461
H0 = 67.36
alpha = 0.49
beta = 0.502

model_bb = D_BB_PSP(l_data, A0_bb)
model_ee = D_EE_PSP(l_data, A1_ee)
model_bao = H_PSP(z_desi, H0, alpha, beta)

# Ошибки
err_bb = 0.0002 * np.ones_like(D_BB_data)
err_ee = 0.1 * np.ones_like(D_EE_data)

# χ² по компонентам
chi2_bb = np.sum(((D_BB_data - model_bb) / err_bb)**2)
chi2_ee = np.sum(((D_EE_data - model_ee) / err_ee)**2)
chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)

print(f"\nχ² для B-мод:  {chi2_bb:.1f}")
print(f"χ² для E-мод:  {chi2_ee:.1f}")
print(f"χ² для BAO:    {chi2_bao:.1f}")
print(f"Общий χ²:      {chi2_bb + chi2_ee + chi2_bao:.1f}")

# ------------------------------------------------------------
# 4. ВИЗУАЛИЗАЦИЯ ОТКЛОНЕНИЙ
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# График 1: B-моды
ax = axes[0, 0]
ax.errorbar(l_data, D_BB_data, yerr=err_bb, fmt='ko', capsize=3, label='Данные Planck')
ax.plot(l_data, model_bb, 'b-', linewidth=2, label='PSP модель')
ax.set_xlabel('Мультиполь $l$')
ax.set_ylabel('$D_l^{BB}$ (мкК$^2$)')
ax.set_title('B-моды: данные vs модель')
ax.legend()
ax.grid(True, alpha=0.3)

# График 2: E-моды
ax = axes[0, 1]
ax.errorbar(l_data, D_EE_data, yerr=err_ee, fmt='ko', capsize=3, label='Данные Planck')
ax.plot(l_data, model_ee, 'b-', linewidth=2, label='PSP модель')
ax.set_xlabel('Мультиполь $l$')
ax.set_ylabel('$D_l^{EE}$ (мкК$^2$)')
ax.set_title('E-моды: данные vs модель')
ax.legend()
ax.grid(True, alpha=0.3)

# График 3: BAO
ax = axes[1, 0]
ax.errorbar(z_desi, H_desi, yerr=H_err, fmt='ko', capsize=3, label='Данные DESI')
z_grid = np.linspace(0, 2.5, 100)
ax.plot(z_grid, H_PSP(z_grid, H0, alpha, beta), 'b-', linewidth=2, label='PSP модель')
ax.set_xlabel('Красное смещение $z$')
ax.set_ylabel('$H(z)$ (км/с/Мпк)')
ax.set_title('BAO: данные vs модель')
ax.legend()
ax.grid(True, alpha=0.3)

# График 4: Отношение данных к модели
ax = axes[1, 1]
ax.plot(l_data, D_BB_data / model_bb, 'ro', markersize=4, label='B-моды')
ax.plot(l_data, D_EE_data / model_ee, 'bo', markersize=4, label='E-моды')
ax.axhline(1, color='black', linestyle='--')
ax.set_xlabel('Мультиполь $l$')
ax.set_ylabel('Отношение (данные / модель)')
ax.set_title('Отклонения модели PSP от данных')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psp_chi2_analysis.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_chi2_analysis.png'")