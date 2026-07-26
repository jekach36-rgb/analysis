import numpy as np
from scipy.integrate import quad

print("=" * 70)
print("ПРОВЕРКА: РАСЧЁТ chi2 НА ДИАГОНАЛЬНЫХ ОШИБКАХ")
print("=" * 70)

# --- 1. Функции H(z) ---
def H_PSP(z, H0=67.36, alpha=0.125, beta=0.35):
    return H0 * np.sqrt(1 + alpha * (z / 2.5)**2 * np.exp(beta * z))

def H_LCDM(z, H0=67.36, Om=0.315):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# --- 2. Светимость-расстояние ---
def D_L(z, H_func, **kwargs):
    def integrand(zp):
        return 1.0 / H_func(zp, **kwargs)
    integral, _ = quad(integrand, 0, z)
    c = 299792.458
    return (c / 100.0) * (1 + z) * integral

# --- 3. Чтение данных ---
def read_pantheon():
    z_list, mu_list, mu_err_list = [], [], []
    with open('Pantheon_SHOES.dat', 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 12:
            continue
        try:
            z = float(parts[2])
            mu = float(parts[10])
            mu_err = float(parts[11])
            if z > 0 and mu > 0 and mu_err > 0:
                z_list.append(z)
                mu_list.append(mu)
                mu_err_list.append(mu_err)
        except:
            continue
    return np.array(z_list), np.array(mu_list), np.array(mu_err_list)

# --- 4. Загрузка ---
z, mu_obs, mu_err = read_pantheon()
print(f"Всего SN: {len(z)}")

# --- 5. Подгонка M_abs ---
M_abs_values = np.linspace(-20, -18, 100)
chi2_PSP_list, chi2_LCDM_list = [], []

for M_abs in M_abs_values:
    mu_PSP = np.array([5 * np.log10(D_L(zi, H_PSP, alpha=0.125, beta=0.35)) + 25 + M_abs for zi in z])
    mu_LCDM = np.array([5 * np.log10(D_L(zi, H_LCDM, Om=0.315)) + 25 + M_abs for zi in z])
    
    chi2_PSP_list.append(np.sum(((mu_obs - mu_PSP) / mu_err)**2))
    chi2_LCDM_list.append(np.sum(((mu_obs - mu_LCDM) / mu_err)**2))

best_idx_PSP = np.argmin(chi2_PSP_list)
best_idx_LCDM = np.argmin(chi2_LCDM_list)
M_abs_PSP = M_abs_values[best_idx_PSP]
M_abs_LCDM = M_abs_values[best_idx_LCDM]

print(f"Лучшие M_abs: PSP = {M_abs_PSP:.3f}, LCDM = {M_abs_LCDM:.3f}")

# --- 6. Финальный расчёт ---
mu_PSP_final = np.array([5 * np.log10(D_L(zi, H_PSP, alpha=0.125, beta=0.35)) + 25 + M_abs_PSP for zi in z])
mu_LCDM_final = np.array([5 * np.log10(D_L(zi, H_LCDM, Om=0.315)) + 25 + M_abs_LCDM for zi in z])

chi2_PSP = np.sum(((mu_obs - mu_PSP_final) / mu_err)**2)
chi2_LCDM = np.sum(((mu_obs - mu_LCDM_final) / mu_err)**2)

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ (ДИАГОНАЛЬНЫЕ ОШИБКИ)")
print("=" * 70)
print(f"chi2_PSP:   {chi2_PSP:.2f}")
print(f"chi2_LCDM:  {chi2_LCDM:.2f}")
print(f"Δchi2 = PSP - LCDM = {chi2_PSP - chi2_LCDM:.2f}")
print("=" * 70)