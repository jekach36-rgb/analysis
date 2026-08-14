import numpy as np
import emcee
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MCMC ДЛЯ УЛУЧШЕННОЙ МОДЕЛИ PSP")
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

z_desi = np.array([0.127, 0.468, 0.671, 0.830, 1.320, 2.330])
H_desi = np.array([71.33, 88.48, 119.45, 108.28, 147.58, 239.38])
H_err = np.array([4.20, 12.32, 16.64, 15.08, 4.49, 4.80])

# ------------------------------------------------------------
# 2. МОДЕЛИ
# ------------------------------------------------------------
def M_PSP(l):
    return 0.29 + 0.21 * (1 - np.exp(-l / 350))

def D_BB_PSP_improved(l, A0, l0, gamma):
    M = M_PSP(l)
    return A0 * np.sin(np.pi * (M - 0.29) / 0.21)**2 * (l / l0)**gamma * np.exp(-l / 300)

def D_EE_PSP_improved(l, A1, alpha_e):
    return A1 * (l / 80)**alpha_e * np.exp(-l / 300)

def H_PSP_improved(z, H0, alpha, beta, gamma_z):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z + gamma_z * z**2))

# ------------------------------------------------------------
# 3. ФУНКЦИЯ ПРАВДОПОДОБИЯ
# ------------------------------------------------------------
def log_likelihood(params):
    A0, l0, gamma, A1, alpha_e, H0, alpha, beta, gamma_z = params
    
    if A0 < 0 or A0 > 1 or l0 < 10 or l0 > 200 or gamma < -2 or gamma > 2:
        return -1e10
    if A1 < 0 or A1 > 2 or alpha_e < 0 or alpha_e > 3:
        return -1e10
    if H0 < 50 or H0 > 80 or alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
        return -1e10
    if gamma_z < -1 or gamma_z > 1:
        return -1e10
    
    try:
        model_bb = D_BB_PSP_improved(l_data, A0, l0, gamma)
        model_ee = D_EE_PSP_improved(l_data, A1, alpha_e)
        model_bao = H_PSP_improved(z_desi, H0, alpha, beta, gamma_z)
        
        err_bb = 0.0002 * np.ones_like(D_BB_data)
        err_ee = 0.1 * np.ones_like(D_EE_data)
        
        chi2_bb = np.sum(((D_BB_data - model_bb) / err_bb)**2)
        chi2_ee = np.sum(((D_EE_data - model_ee) / err_ee)**2)
        chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)
        
        return -0.5 * (chi2_bb + chi2_ee + chi2_bao)
    except:
        return -1e10

# ------------------------------------------------------------
# 4. ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ
# ------------------------------------------------------------
print("\nПоиск оптимальных параметров...")

res = minimize(lambda p: -log_likelihood(p), 
               [0.012, 80, 2.0, 0.5, 1.5, 67.36, 0.15, 0.35, 0.0],
               bounds=[(0.001, 1), (10, 200), (-2, 2), (0.01, 2), (0.1, 3), (50, 80), (0.01, 1), (0.01, 1), (-1, 1)])
best = res.x

print(f"\n✅ Оптимальные параметры:")
print(f"   A0     = {best[0]:.4f}")
print(f"   l0     = {best[1]:.1f}")
print(f"   gamma  = {best[2]:.3f}")
print(f"   A1     = {best[3]:.3f}")
print(f"   alpha_e= {best[4]:.3f}")
print(f"   H0     = {best[5]:.1f}")
print(f"   alpha  = {best[6]:.4f}")
print(f"   beta   = {best[7]:.4f}")
print(f"   gamma_z= {best[8]:.4f}")

# ------------------------------------------------------------
# 5. ВЫЧИСЛЕНИЕ χ², AIC, BIC
# ------------------------------------------------------------
def compute_chi2(params):
    A0, l0, gamma, A1, alpha_e, H0, alpha, beta, gamma_z = params
    model_bb = D_BB_PSP_improved(l_data, A0, l0, gamma)
    model_ee = D_EE_PSP_improved(l_data, A1, alpha_e)
    model_bao = H_PSP_improved(z_desi, H0, alpha, beta, gamma_z)
    
    err_bb = 0.0002 * np.ones_like(D_BB_data)
    err_ee = 0.1 * np.ones_like(D_EE_data)
    
    chi2_bb = np.sum(((D_BB_data - model_bb) / err_bb)**2)
    chi2_ee = np.sum(((D_EE_data - model_ee) / err_ee)**2)
    chi2_bao = np.sum(((H_desi - model_bao) / H_err)**2)
    return chi2_bb + chi2_ee + chi2_bao

chi2_psp = compute_chi2(best)
n_params = 9
n_data = len(l_data) + len(z_desi)

AIC = 2 * n_params + chi2_psp
BIC = n_params * np.log(n_data) + chi2_psp

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ")
print("=" * 70)
print(f"χ² = {chi2_psp:.1f}")
print(f"AIC = {AIC:.1f}")
print(f"BIC = {BIC:.1f}")