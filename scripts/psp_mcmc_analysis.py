import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm, chi2
from scipy.optimize import minimize
import emcee
import time

print("=" * 70)
print("MCMC-АНАЛИЗ МОДЕЛИ PSP")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ PLANCK (B-МОДЫ И E-МОДЫ)
# ------------------------------------------------------------
data = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
l = data[:, 0]
cl_bb = data[:, 4]  # B-моды
cl_ee = data[:, 3]  # E-моды

# Переводим в D_l
D_BB = l * (l + 1) * cl_bb / (2 * np.pi)
D_EE = l * (l + 1) * cl_ee / (2 * np.pi)

# Берём только низкие l (2-100)
mask = (l >= 2) & (l <= 100)
l_data = l[mask]
D_BB_data = D_BB[mask]
D_EE_data = D_EE[mask]

# ------------------------------------------------------------
# 2. МОДЕЛЬ PSP
# ------------------------------------------------------------
def M_PSP(l):
    return 0.29 + 0.21 * (1 - np.exp(-l / 350))

def D_BB_PSP(l, A0):
    M = M_PSP(l)
    return A0 * np.sin(np.pi * (M - 0.29) / 0.21)**2 * np.exp(-l / 300)

def D_EE_PSP(l, A1):
    return A1 * (l / 80) ** 1.5 * np.exp(-l / 300)

def D_BAO_PSP(z, H0, alpha, beta):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

# ------------------------------------------------------------
# 3. ДАННЫЕ DESI (BAO)
# ------------------------------------------------------------
z_desi = np.array([0.127, 0.468, 0.671, 0.830, 1.320, 2.330])
H_desi = np.array([71.33, 88.48, 119.45, 108.28, 147.58, 239.38])
H_err = np.array([4.20, 12.32, 16.64, 15.08, 4.49, 4.80])

# ------------------------------------------------------------
# 4. ФУНКЦИЯ ПРАВДОПОДОБИЯ
# ------------------------------------------------------------
def log_likelihood_PSP(params):
    """Логарифм правдоподобия для PSP"""
    A0_bb, A1_ee, H0, alpha, beta = params
    
    # Ограничения на параметры
    if A0_bb < 0 or A1_ee < 0 or H0 < 50 or H0 > 80 or alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
        return -np.inf
    
    # B-моды
    D_BB_model = D_BB_PSP(l_data, A0_bb)
    chi2_bb = np.sum(((D_BB_data - D_BB_model) / (0.0001 * np.ones_like(D_BB_data))) ** 2)
    
    # E-моды
    D_EE_model = D_EE_PSP(l_data, A1_ee)
    chi2_ee = np.sum(((D_EE_data - D_EE_model) / (0.01 * np.ones_like(D_EE_data))) ** 2)
    
    # BAO
    H_model = D_BAO_PSP(z_desi, H0, alpha, beta)
    chi2_bao = np.sum(((H_desi - H_model) / H_err) ** 2)
    
    # Полное χ²
    chi2_total = chi2_bb + chi2_ee + chi2_bao
    
    return -0.5 * chi2_total

def log_likelihood_LCDM(params):
    """Логарифм правдоподобия для ΛCDM"""
    H0, omega_m = params
    
    if H0 < 50 or H0 > 80 or omega_m < 0.1 or omega_m > 0.5:
        return -np.inf
    
    # BAO
    omega_lam = 1 - omega_m
    H_model = H0 * np.sqrt(omega_m * (1 + z_desi)**3 + omega_lam)
    chi2_bao = np.sum(((H_desi - H_model) / H_err) ** 2)
    
    return -0.5 * chi2_bao

# ------------------------------------------------------------
# 5. MCMC
# ------------------------------------------------------------
def run_mcmc(log_likelihood, n_dim, n_walkers=50, n_steps=2000, burn_in=500):
    """Запуск MCMC"""
    # Начальные позиции
    p0 = np.random.randn(n_walkers, n_dim) * 0.1 + np.ones(n_dim) * 0.5
    
    # Сэмплер
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood)
    
    print(f"   Запуск MCMC с {n_walkers} ходоками, {n_steps} шагов...")
    start = time.time()
    sampler.run_mcmc(p0, n_steps, progress=True)
    end = time.time()
    print(f"   Готово за {end - start:.2f} секунд")
    
    # Удаляем burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    return samples

# ------------------------------------------------------------
# 6. ЗАПУСК MCMC
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("ЗАПУСК MCMC ДЛЯ PSP")
print("=" * 70)
print("Параметры: A0_bb, A1_ee, H0, alpha, beta")

samples_psp = run_mcmc(log_likelihood_PSP, n_dim=5, n_walkers=50, n_steps=2000, burn_in=500)

# Медианные значения и ошибки
medians_psp = np.median(samples_psp, axis=0)
lower_psp = np.percentile(samples_psp, 16, axis=0)
upper_psp = np.percentile(samples_psp, 84, axis=0)

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ PSP")
print("=" * 70)
print(f"A0_bb = {medians_psp[0]:.4f} (+{upper_psp[0]-medians_psp[0]:.4f}/-{medians_psp[0]-lower_psp[0]:.4f})")
print(f"A1_ee = {medians_psp[1]:.4f} (+{upper_psp[1]-medians_psp[1]:.4f}/-{medians_psp[1]-lower_psp[1]:.4f})")
print(f"H0    = {medians_psp[2]:.2f} (+{upper_psp[2]-medians_psp[2]:.2f}/-{medians_psp[2]-lower_psp[2]:.2f})")
print(f"alpha = {medians_psp[3]:.4f} (+{upper_psp[3]-medians_psp[3]:.4f}/-{medians_psp[3]-lower_psp[3]:.4f})")
print(f"beta  = {medians_psp[4]:.4f} (+{upper_psp[4]-medians_psp[4]:.4f}/-{medians_psp[4]-lower_psp[4]:.4f})")

# ------------------------------------------------------------
# 7. СРАВНЕНИЕ С ΛCDM
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("ЗАПУСК MCMC ДЛЯ ΛCDM")
print("=" * 70)
print("Параметры: H0, omega_m")

samples_lcdm = run_mcmc(log_likelihood_LCDM, n_dim=2, n_walkers=50, n_steps=2000, burn_in=500)

medians_lcdm = np.median(samples_lcdm, axis=0)
lower_lcdm = np.percentile(samples_lcdm, 16, axis=0)
upper_lcdm = np.percentile(samples_lcdm, 84, axis=0)

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ΛCDM")
print("=" * 70)
print(f"H0      = {medians_lcdm[0]:.2f} (+{upper_lcdm[0]-medians_lcdm[0]:.2f}/-{medians_lcdm[0]-lower_lcdm[0]:.2f})")
print(f"omega_m = {medians_lcdm[1]:.4f} (+{upper_lcdm[1]-medians_lcdm[1]:.4f}/-{medians_lcdm[1]-lower_lcdm[1]:.4f})")

# ------------------------------------------------------------
# 8. СРАВНЕНИЕ МОДЕЛЕЙ (AIC/BIC)
# ------------------------------------------------------------
# Для PSP
n_psp = len(samples_psp)
chi2_psp = -2 * log_likelihood_PSP(medians_psp)
AIC_psp = 2 * 5 + chi2_psp
BIC_psp = 5 * np.log(n_psp) + chi2_psp

# Для ΛCDM
n_lcdm = len(samples_lcdm)
chi2_lcdm = -2 * log_likelihood_LCDM(medians_lcdm)
AIC_lcdm = 2 * 2 + chi2_lcdm
BIC_lcdm = 2 * np.log(n_lcdm) + chi2_lcdm

print("\n" + "=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)
print(f"Модель   χ²    AIC    BIC")
print(f"PSP      {chi2_psp:.1f}  {AIC_psp:.1f}  {BIC_psp:.1f}")
print(f"ΛCDM     {chi2_lcdm:.1f}  {AIC_lcdm:.1f}  {BIC_lcdm:.1f}")

if AIC_psp < AIC_lcdm:
    print("\n✅ PSP лучше по AIC (ΔAIC = {:.1f})".format(AIC_lcdm - AIC_psp))
else:
    print("\n⚠️ ΛCDM лучше по AIC, но PSP близка (ΔAIC = {:.1f})".format(AIC_psp - AIC_lcdm))

# ------------------------------------------------------------
# 9. ПОСТРОЕНИЕ ГРАФИКОВ (CORNER PLOT)
# ------------------------------------------------------------
fig = corner.corner(samples_psp, labels=['A0_bb', 'A1_ee', 'H0', 'alpha', 'beta'],
                    truths=medians_psp, show_titles=True, title_fmt='.3f')
plt.savefig('psp_corner_plot.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_corner_plot.png'")