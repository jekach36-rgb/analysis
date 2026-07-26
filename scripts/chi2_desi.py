import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky, solve

print("=" * 70)
print("ПРОВЕРКА PSP ПО ДАННЫМ DESI DR2 (BAO)")
print("=" * 70)

# --- 1. Функции H(z) ---
def H_PSP(z, H0=67.36, alpha=0.125, beta=0.35):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

def H_LCDM(z, H0=67.36, Om=0.315):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# --- 2. Расстояния ---
def D_M(z, H_func, **kwargs):
    """Поперечное расстояние (в Мпк)"""
    def integrand(zp):
        return 1.0 / H_func(zp, **kwargs)
    integral, _ = quad(integrand, 0, z)
    c = 299792.458
    return (c / 100.0) * integral

def D_H(z, H_func, **kwargs):
    """Радиальное расстояние"""
    return 2997.92458 / H_func(z, **kwargs)  # c/H в Мпк

def D_V(z, H_func, **kwargs):
    """Сферическое расстояние DV"""
    dm = D_M(z, H_func, **kwargs)
    dh = D_H(z, H_func, **kwargs)
    return (z * dm**2 * dh)**(1/3)

# --- 3. Данные DESI DR2 ---
# z, DM/rd, DH/rd
desi_data = [
    [0.510, 13.58758434, 21.86294686],
    [0.706, 17.35069094, 19.45534918],
    [0.934, 21.57563956, 17.64149464],
    [1.321, 27.60085612, 14.17602155],
    [1.484, 30.51190063, 12.81699964],
    [2.330, 38.98897396, 8.63154567],
]

# BGS точка
bgs_z = 0.295
bgs_DV_over_rs = 7.92512927

# rd (звуковой горизонт) — стандартное значение 147.5 Мпк
rd = 147.5

# --- 4. Функция для расчёта chi2 ---
def compute_chi2_bao(H_func, **kwargs):
    diff = []
    for z, dm_obs, dh_obs in desi_data:
        dm_th = D_M(z, H_func, **kwargs) / rd
        dh_th = D_H(z, H_func, **kwargs) / rd
        diff.append(dm_obs - dm_th)
        diff.append(dh_obs - dh_th)
    diff = np.array(diff)
    
    # Ковариационная матрица 12x12
    cov = np.array([
        [2.834737420000000077e-02, -3.260620069999999732e-02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3.260620069999999732e-02, 1.839280399999999871e-01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3.237524420000000014e-02, -2.374456460000000033e-02, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -2.374456460000000033e-02, 1.114691980000000054e-01, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2.617328160000000070e-02, -1.129380060000000074e-02, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1.129380060000000074e-02, 4.041838779999999687e-02, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1.053365160000000050e-01, -2.903084179999999848e-02, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -2.903084179999999848e-02, 5.042330920000000216e-02, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5.830202770000000312e-01, -1.952155620000000091e-01, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1.952155620000000091e-01, 2.683361930000000006e-01, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.021361940000000031e-02, -2.313952160000000077e-02],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.313952160000000077e-02, 2.826857790000000259e-01],
    ])
    
    try:
        L = cholesky(cov, lower=True)
        y = solve(L, diff)
        chi2 = np.dot(y, y)
    except:
        chi2 = diff.T @ np.linalg.pinv(cov) @ diff
    
    return chi2

# --- 5. Расчёт chi2 для PSP и LCDM ---
chi2_PSP = compute_chi2_bao(H_PSP, alpha=0.125, beta=0.35)
chi2_LCDM = compute_chi2_bao(H_LCDM, Om=0.315)

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ПО DESI DR2")
print("=" * 70)
print(f"chi2_PSP:   {chi2_PSP:.2f}")
print(f"chi2_LCDM:  {chi2_LCDM:.2f}")
print(f"Δchi2 = PSP - LCDM = {chi2_PSP - chi2_LCDM:.2f}")
print("=" * 70)