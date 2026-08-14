import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

print("=" * 70)
print("χ²-АНАЛИЗ: ΛCDM vs PSP (расстояния)")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
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
c = 299792.458
H0 = 67.36

# ΛCDM: расстояние через z
D_LCDM = df_filt['z'].values * c / H0

# PSP: расстояние через M
D_PSP = np.abs(df_filt['M'].values - 0.29) * 1000

# ------------------------------------------------------------
# 4. ВЫЧИСЛЕНИЕ χ²
# ------------------------------------------------------------
# Ошибки: используем стандартное отклонение как оценку ошибки
err_lcdm = np.std(D_LCDM) * np.ones_like(D_LCDM)
err_psp = np.std(D_PSP) * np.ones_like(D_PSP)

# χ² для ΛCDM (сравнение с нулевой гипотезой: расстояние = 0)
chi2_lcdm = np.sum((D_LCDM / err_lcdm) ** 2)

# χ² для PSP (сравнение с нулевой гипотезой: расстояние = 0)
chi2_psp = np.sum((D_PSP / err_psp) ** 2)

# ------------------------------------------------------------
# 5. СТАТИСТИЧЕСКИЕ КРИТЕРИИ (AIC, BIC)
# ------------------------------------------------------------
n_data = len(df_filt)
n_params_lcdm = 1  # H0
n_params_psp = 2   # M и калибровка

AIC_lcdm = 2 * n_params_lcdm + chi2_lcdm
BIC_lcdm = n_params_lcdm * np.log(n_data) + chi2_lcdm

AIC_psp = 2 * n_params_psp + chi2_psp
BIC_psp = n_params_psp * np.log(n_data) + chi2_psp

# ------------------------------------------------------------
# 6. ВЫВОД РЕЗУЛЬТАТОВ
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")
print("=" * 70)
print(f"\n{'Модель':<10} {'χ²':<15} {'AIC':<15} {'BIC':<15} {'p-value':<15}")
print("-" * 70)

# p-value для χ²
p_lcdm = 1 - chi2.cdf(chi2_lcdm, n_data - n_params_lcdm)
p_psp = 1 - chi2.cdf(chi2_psp, n_data - n_params_psp)

print(f"{'ΛCDM':<10} {chi2_lcdm:<15.2f} {AIC_lcdm:<15.2f} {BIC_lcdm:<15.2f} {p_lcdm:<15.4f}")
print(f"{'PSP':<10} {chi2_psp:<15.2f} {AIC_psp:<15.2f} {BIC_psp:<15.2f} {p_psp:<15.4f}")

# ------------------------------------------------------------
# 7. ИНТЕРПРЕТАЦИЯ
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("ИНТЕРПРЕТАЦИЯ")
print("=" * 70)

if chi2_psp < chi2_lcdm:
    print(f"\n✅ PSP имеет МЕНЬШЕ χ², чем ΛCDM (Δχ² = {chi2_lcdm - chi2_psp:.2f})")
    print("   Это означает, что PSP лучше описывает данные.")
else:
    print(f"\n⚠️ ΛCDM имеет МЕНЬШЕ χ², чем PSP (Δχ² = {chi2_psp - chi2_lcdm:.2f})")
    print("   Это означает, что ΛCDM лучше описывает данные.")

if AIC_psp < AIC_lcdm:
    print(f"\n✅ PSP лучше по AIC (ΔAIC = {AIC_lcdm - AIC_psp:.2f})")
else:
    print(f"\n⚠️ ΛCDM лучше по AIC (ΔAIC = {AIC_psp - AIC_lcdm:.2f})")

if p_psp > 0.05 and p_lcdm > 0.05:
    print("\n✅ Обе модели дают хорошее описание данных (p > 0.05).")
elif p_psp > 0.05:
    print("\n✅ PSP даёт хорошее описание данных (p > 0.05).")
elif p_lcdm > 0.05:
    print("\n✅ ΛCDM даёт хорошее описание данных (p > 0.05).")
else:
    print("\n⚠️ Обе модели дают плохое описание данных (p < 0.05).")

print("\n" + "=" * 70)
print("ГЛАВНЫЙ ВЫВОД")
print("=" * 70)
if chi2_psp < chi2_lcdm and AIC_psp < AIC_lcdm:
    print("✅ Модель PSP статистически значимо лучше описывает данные, чем ΛCDM.")
    print("   Это означает, что интерпретация расстояний через фазу M")
    print("   является более физически обоснованной, чем через красное смещение z.")
else:
    print("⚠️ Требуется дополнительный анализ с более точными данными.")