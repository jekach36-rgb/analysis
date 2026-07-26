import numpy as np
from scipy.integrate import quad

print("=" * 70)
print("ЧЕСТНОЕ СРАВНЕНИЕ PSP vs ΛCDM ПО КВАЗАРАМ (LUSSO+ 2020)")
print("=" * 70)

# --- 1. Модель PSP (расстояние через LX/LUV) ---
def R_shell(M, R0=1.0):
    return R0 * (1 + 0.5 * M**2)

def Phi_M(M):
    return 1.0 + 0.5 * M

def calc_distance(M, r=0.1):
    def integrand(m):
        R = R_shell(m)
        Phi = Phi_M(m)
        factor = 1 - (r**2) / (R**2)
        return R * Phi * factor
    result, _ = quad(integrand, 0, M)
    return result

def mu_PSP(LX, LUV, M_abs=-19.3, r=0.1):
    xi = LX / LUV
    M = 0.29 + 0.01 * (xi - 1)
    D = calc_distance(M, r)
    if D <= 0:
        return -999
    return 5 * np.log10(D) + 25 + M_abs

# --- 2. Модель ΛCDM (через красное смещение) ---
def H_LCDM(z, H0=67.36, Om=0.315):
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

def D_L_LCDM(z, H0=67.36, Om=0.315):
    def integrand(zp):
        return 1.0 / H_LCDM(zp, H0, Om)
    integral, _ = quad(integrand, 0, z)
    c = 299792.458
    return (c / 100.0) * (1 + z) * integral

def mu_LCDM(z, M_abs=-19.3, H0=67.36, Om=0.315):
    D = D_L_LCDM(z, H0, Om)
    return 5 * np.log10(D) + 25 + M_abs

# --- 3. Загрузка данных ---
def load_lusso():
    data = []
    with open('table3.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            try:
                z = float(parts[3])
                logFUV = float(parts[4])
                logFX = float(parts[6])
                DM = float(parts[10])
                e_DM = float(parts[11])
                if z > 0 and DM > 0 and e_DM > 0:
                    data.append([z, logFUV, logFX, DM, e_DM])
            except:
                continue
    return np.array(data)

data = load_lusso()
print(f"Загружено квазаров: {len(data)}")

# --- 4. Подгонка M_abs (для PSP и ΛCDM) ---
M_abs_values = np.linspace(-20, -18, 100)
chi2_psp_list = []
chi2_lcdm_list = []

for M_abs in M_abs_values:
    chi2_psp_temp = 0
    chi2_lcdm_temp = 0
    for i in range(len(data)):
        z, logFUV, logFX, DM_obs, e_DM = data[i]
        
        # PSP
        LUV = 10**logFUV
        LX = 10**logFX
        mu_pred_psp = mu_PSP(LX, LUV, M_abs)
        if mu_pred_psp > 0:
            chi2_psp_temp += ((DM_obs - mu_pred_psp) / e_DM)**2
        
        # ΛCDM
        mu_pred_lcdm = mu_LCDM(z, M_abs)
        chi2_lcdm_temp += ((DM_obs - mu_pred_lcdm) / e_DM)**2
    
    chi2_psp_list.append(chi2_psp_temp)
    chi2_lcdm_list.append(chi2_lcdm_temp)

best_idx_psp = np.argmin(chi2_psp_list)
best_idx_lcdm = np.argmin(chi2_lcdm_list)
M_abs_psp = M_abs_values[best_idx_psp]
M_abs_lcdm = M_abs_values[best_idx_lcdm]
chi2_psp = chi2_psp_list[best_idx_psp]
chi2_lcdm = chi2_lcdm_list[best_idx_lcdm]

print(f"\nЛучшие M_abs:")
print(f"  PSP:   {M_abs_psp:.3f}")
print(f"  ΛCDM:  {M_abs_lcdm:.3f}")

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ (ЧЕСТНОЕ СРАВНЕНИЕ)")
print("=" * 70)
print(f"χ²_PSP:   {chi2_psp:.2f}")
print(f"χ²_LCDM:  {chi2_lcdm:.2f}")
print(f"Δχ² = PSP - ΛCDM = {chi2_psp - chi2_lcdm:.2f}")
print("=" * 70)

if chi2_psp < chi2_lcdm:
    print("\n✅ PSP лучше описывает данные квазаров, чем ΛCDM.")
else:
    print("\n⚠️ ΛCDM лучше описывает данные квазаров, чем PSP.")
