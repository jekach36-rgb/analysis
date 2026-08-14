import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MCMC-АНАЛИЗ МОДЕЛИ PSP (ИСПРАВЛЕННЫЙ)")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------
data = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
l = data[:, 0]
cl_bb = data[:, 4]  # B-моды
cl_ee = data[:, 3]  # E-моды

D_BB = l * (l + 1) * cl_bb / (2 * np.pi)
D_EE = l * (l + 1) * cl_ee / (2 * np.pi)

# Берём l = 2-100
mask = (l >= 2) & (l <= 100)
l_data = l[mask]
D_BB_data = D_BB[mask]
D_EE_data = D_EE[mask]

print(f"✅ Загружено {len(l_data)} точек данных (B-моды и E-моды)")

# Данные DESI (BAO)
z_desi = np.array([0.127, 0.468, 0.671, 0.830, 1.320, 2.330])
H_desi = np.array([71.33, 88.48, 119.45, 108.28, 147.58, 239.38])
H_err = np.array([4.20, 12.32, 16.64, 15.08, 4.49, 4.80])

# ------------------------------------------------------------
# 2. МОДЕЛИ
# ------------------------------------------------------------
def M_PSP(l):
    return 0.29 + 0.21 * (1 - np.exp(-l / 350))

def D_BB_PSP(l, A0):
    M = M_PSP(l)
    return A0 * np.sin(np.pi * (M - 0.29) / 0.21)**2 * np.exp(-l / 300)

def D_EE_PSP(l, A1):
    return A1 * (l / 80)**1.5 * np.exp(-l / 300)

def H_PSP(z, H0, alpha, beta):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

def H_LCDM(z, H0, omega_m):
    return H0 * np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))

# ------------------------------------------------------------
# 3. ФУНКЦИЯ ПРАВДОПОДОБИЯ
# ------------------------------------------------------------
def log_likelihood_PSP(params):
    A0_bb, A1_ee, H0, alpha, beta = params
    
    # Проверка границ
    if A0_bb < 0 or A0_bb > 1 or A1_ee < 0 or A1_ee > 1:
        return -1e10
    if H0 < 50 or H0 > 80 or alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
        return -1e10
    
    try:
        # B-моды
        model_bb = D_BB_PSP(l_data, A0_bb)
        chi2_bb = np.sum(((D_BB_data - model_bb) / (0.0002 * np.ones_like(D_BB_data)))**2)
        
        # E-моды
        model_ee = D_EE_PSP(l_data, A1_ee)
        chi2_ee = np.sum(((D_EE_data - model_ee) / (0.1 * np.ones_like(D_EE_data)))**2)
        
        # BAO
        model_bao = H_PSP(z_desi, H0, alpha, beta)
        chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)
        
        chi2_total = chi2_bb + chi2_ee + chi2_bao
        return -0.5 * chi2_total
    except:
        return -1e10

def log_likelihood_LCDM(params):
    H0, omega_m = params
    
    if H0 < 50 or H0 > 80 or omega_m < 0.1 or omega_m > 0.5:
        return -1e10
    
    try:
        model_bao = H_LCDM(z_desi, H0, omega_m)
        chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)
        return -0.5 * chi2_bao
    except:
        return -1e10

# ------------------------------------------------------------
# 4. ПОИСК ЛУЧШИХ ПАРАМЕТРОВ
# ------------------------------------------------------------
print("\nПоиск оптимальных параметров...")

# PSP
res_psp = minimize(lambda p: -log_likelihood_PSP(p), 
                   [0.5, 0.5, 67, 0.15, 0.35],
                   bounds=[(0.01, 1), (0.01, 1), (50, 80), (0.01, 1), (0.01, 1)])
best_psp = res_psp.x
print(f"✅ PSP: A0_bb={best_psp[0]:.3f}, A1_ee={best_psp[1]:.3f}, H0={best_psp[2]:.1f}")

# ΛCDM
res_lcdm = minimize(lambda p: -log_likelihood_LCDM(p), 
                    [67, 0.315],
                    bounds=[(50, 80), (0.1, 0.5)])
best_lcdm = res_lcdm.x
print(f"✅ ΛCDM: H0={best_lcdm[0]:.1f}, omega_m={best_lcdm[1]:.3f}")

# ------------------------------------------------------------
# 5. ВЫЧИСЛЕНИЕ χ², AIC, BIC
# ------------------------------------------------------------
def compute_chi2_psp(params):
    A0_bb, A1_ee, H0, alpha, beta = params
    model_bb = D_BB_PSP(l_data, A0_bb)
    model_ee = D_EE_PSP(l_data, A1_ee)
    model_bao = H_PSP(z_desi, H0, alpha, beta)
    
    chi2_bb = np.sum(((D_BB_data - model_bb) / (0.0002 * np.ones_like(D_BB_data)))**2)
    chi2_ee = np.sum(((D_EE_data - model_ee) / (0.1 * np.ones_like(D_EE_data)))**2)
    chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)
    return chi2_bb + chi2_ee + chi2_bao

def compute_chi2_lcdm(params):
    H0, omega_m = params
    model_bao = H_LCDM(z_desi, H0, omega_m)
    return np.sum(((H_desi - model_bao) / H_err)**2)

chi2_psp = compute_chi2_psp(best_psp)
chi2_lcdm = compute_chi2_lcdm(best_lcdm)

n_data = len(l_data) + len(z_desi)
n_params_psp = 5
n_params_lcdm = 2

AIC_psp = 2 * n_params_psp + chi2_psp
BIC_psp = n_params_psp * np.log(n_data) + chi2_psp

AIC_lcdm = 2 * n_params_lcdm + chi2_lcdm
BIC_lcdm = n_params_lcdm * np.log(n_data) + chi2_lcdm

# ------------------------------------------------------------
# 6. ВЫВОД РЕЗУЛЬТАТОВ
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО СРАВНЕНИЯ")
print("=" * 70)
print(f"\n{'Модель':<10} {'χ²':<12} {'AIC':<12} {'BIC':<12}")
print("-" * 50)
print(f"{'PSP':<10} {chi2_psp:<12.1f} {AIC_psp:<12.1f} {BIC_psp:<12.1f}")
print(f"{'ΛCDM':<10} {chi2_lcdm:<12.1f} {AIC_lcdm:<12.1f} {BIC_lcdm:<12.1f}")

delta_AIC = AIC_psp - AIC_lcdm
delta_BIC = BIC_psp - BIC_lcdm

print("\n" + "=" * 70)
print("ИНТЕРПРЕТАЦИЯ")
print("=" * 70)

if delta_AIC < -2:
    print(f"✅ PSP лучше по AIC: ΔAIC = {delta_AIC:.1f} (сильное преимущество)")
elif delta_AIC < 0:
    print(f"✅ PSP немного лучше по AIC: ΔAIC = {delta_AIC:.1f}")
elif delta_AIC < 2:
    print(f"⚠️ Модели статистически неразличимы: ΔAIC = {delta_AIC:.1f}")
else:
    print(f"⚠️ ΛCDM лучше по AIC: ΔAIC = {delta_AIC:.1f}")

if delta_BIC < -2:
    print(f"✅ PSP лучше по BIC: ΔBIC = {delta_BIC:.1f} (сильное преимущество)")
elif delta_BIC < 0:
    print(f"✅ PSP немного лучше по BIC: ΔBIC = {delta_BIC:.1f}")
elif delta_BIC < 2:
    print(f"⚠️ Модели статистически неразличимы: ΔBIC = {delta_BIC:.1f}")
else:
    print(f"⚠️ ΛCDM лучше по BIC: ΔBIC = {delta_BIC:.1f}")