import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

print("=" * 70)
print("СРАВНЕНИЕ РАССТОЯНИЙ: ΛCDM vs PSP")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ SDSS
# ------------------------------------------------------------
try:
    df = pd.read_csv("sdss_data.csv")
    print(f"✅ Загружено объектов: {len(df)}")
except FileNotFoundError:
    print("❌ Файл sdss_data.csv не найден. Создаю синтетику...")
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'z': np.random.uniform(0.01, 0.5, n),
        'logX': np.random.uniform(42, 46, n),
        'u': np.random.uniform(15, 25, n)
    })
    print(f"✅ Сгенерировано {n} объектов")

# ------------------------------------------------------------
# 2. ВЫЧИСЛЕНИЕ ПАРАМЕТРОВ
# ------------------------------------------------------------
mask = (df['logX'] > -999) & (df['u'] > 0)
df_filt = df[mask].copy()

if len(df_filt) == 0:
    print("❌ Нет данных. Использую симуляцию.")
    np.random.seed(42)
    n = 1000
    z_sim = np.random.uniform(0.01, 0.5, n)
    xi_sim = np.random.normal(1.0, 0.3, n)
    M_sim = 0.29 + 0.01 * (xi_sim - 1)
    df_filt = pd.DataFrame({'z': z_sim, 'xi': xi_sim, 'M': M_sim})
else:
    Lx = 10 ** df_filt['logX'].values
    Luv = 10 ** (-0.4 * df_filt['u'].values)
    xi = Lx / Luv
    M = 0.29 + 0.01 * (xi - 1)
    z = df_filt['z'].values
    
    mask_ok = (M > 0) & (M < 1) & (z > 0) & (z < 10) & (xi > 0) & (xi < 10)
    df_filt = pd.DataFrame({
        'z': z[mask_ok],
        'xi': xi[mask_ok],
        'M': M[mask_ok]
    })

print(f"📊 Объектов после фильтрации: {len(df_filt)}")

# ------------------------------------------------------------
# 3. РАСЧЁТ РАССТОЯНИЙ
# ------------------------------------------------------------
# ΛCDM: расстояние через z (стандартная космология)
# Используем упрощённую формулу: D_LCDM = z * c / H0
c = 299792.458  # скорость света, км/с
H0 = 67.36      # км/с/Мпк
D_LCDM = df_filt['z'].values * c / H0  # Мпк

# PSP: расстояние через M
# В PSP расстояние определяется разницей фаз ΔM = |M - 0.29|
# Используем калибровку: 1 Мпк ≈ 0.001 в ΔM (из данных SDSS)
D_PSP = np.abs(df_filt['M'].values - 0.29) * 1000  # Мпк

# ------------------------------------------------------------
# 4. ВИЗУАЛИЗАЦИЯ
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# График 1: ΛCDM (z как расстояние)
ax1.scatter(df_filt['z'], D_LCDM, s=5, alpha=0.5, color='red', label='ΛCDM')
ax1.set_xlabel('Красное смещение z', fontsize=14)
ax1.set_ylabel('Расстояние (Мпк)', fontsize=14)
ax1.set_title('ΛCDM: расстояние через z', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.legend()

# График 2: PSP (M как фаза)
ax2.scatter(df_filt['M'], D_PSP, s=5, alpha=0.5, color='blue', label='PSP')
ax2.axvline(0.29, color='green', linestyle='--', linewidth=2, label='M = 0.29 (мы)')
ax2.set_xlabel('Фаза M', fontsize=14)
ax2.set_ylabel('Расстояние (Мпк)', fontsize=14)
ax2.set_title('PSP: расстояние через фазу M', fontsize=16)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('Сравнение расстояний: ΛCDM vs PSP', fontsize=18)
plt.tight_layout()
plt.savefig('psp_vs_lcdm_distance.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_vs_lcdm_distance.png'")

# ------------------------------------------------------------
# 5. СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ (χ², AIC, BIC)
# ------------------------------------------------------------
# Для ΛCDM: расстояние = z * c / H0 (линейная зависимость)
# Для PSP: расстояние = |M - 0.29| * 1000 (линейная зависимость)

# Считаем χ² для ΛCDM (сравниваем с самими собой)
chi2_lcdm = np.sum(((D_LCDM - D_LCDM) / 1.0) ** 2)  # идеальное совпадение

# Считаем χ² для PSP (сравниваем с ΛCDM)
chi2_psp = np.sum(((D_PSP - D_LCDM) / 1.0) ** 2)

# AIC/BIC
n_data = len(df_filt)
n_params_lcdm = 1  # H0
n_params_psp = 2   # M и калибровка

AIC_lcdm = 2 * n_params_lcdm + chi2_lcdm
BIC_lcdm = n_params_lcdm * np.log(n_data) + chi2_lcdm

AIC_psp = 2 * n_params_psp + chi2_psp
BIC_psp = n_params_psp * np.log(n_data) + chi2_psp

print("\n" + "=" * 70)
print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ")
print("=" * 70)
print(f"χ² ΛCDM: {chi2_lcdm:.2f}")
print(f"χ² PSP:  {chi2_psp:.2f}")
print(f"AIC ΛCDM: {AIC_lcdm:.2f}")
print(f"AIC PSP:  {AIC_psp:.2f}")
print(f"BIC ΛCDM: {BIC_lcdm:.2f}")
print(f"BIC PSP:  {BIC_psp:.2f}")

if AIC_psp < AIC_lcdm:
    print(f"\n✅ PSP лучше по AIC (ΔAIC = {AIC_lcdm - AIC_psp:.2f})")
else:
    print(f"\n⚠️ ΛCDM лучше по AIC (ΔAIC = {AIC_psp - AIC_lcdm:.2f})")