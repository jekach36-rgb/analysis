#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PSP vs ΛCDM: ФИНАЛЬНОЕ СРАВНЕНИЕ
==================================
- Данные: Pantheon+ (1701 точка)
- Автоматический поиск файла Pantheon_SH0ES.dat
- Модели: PSP и ΛCDM
- Сравнение через χ², AIC, BIC
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PSP vs ΛCDM: ФИНАЛЬНОЕ СРАВНЕНИЕ")
print("="*70)

# ============================================================
# 1. ПОИСК ФАЙЛА
# ============================================================

def find_file(filename):
    """Ищет файл в разных местах."""
    paths = [
        filename,
        './' + filename,
        '../' + filename,
        '../../' + filename,
        'D:/' + filename,
        'D:\\' + filename,
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        'D:/ТЕОРИЯ ЦИКЛА ВСЕЛЕННОЙ ЧЕРНОКНИЖНОГО (ТЦВЧ)/' + filename,
        'D:/ТЕОРИЯ ЦИКЛА ВСЕЛЕННОЙ ЧЕРНОКНИЖНОГО (ТЦВЧ)/Скрипты/' + filename,
    ]
    
    for p in paths:
        if os.path.exists(p):
            print(f"✅ Файл найден: {p}")
            return p
    
    print(f"❌ Файл {filename} не найден!")
    print("   Поиск в:", os.getcwd())
    print("   Пожалуйста, укажите правильный путь.")
    return None

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("\n🔄 Загрузка данных Pantheon+...")

file_path = find_file("Pantheon_SH0ES.dat")
if file_path is None:
    print("   Создаю синтетические данные для демонстрации...")
    # Создаём синтетические данные
    np.random.seed(42)
    z_data = np.linspace(0.01, 2.2, 100)
    m_b = 42 + 5 * np.log10(z_data + 0.1) + np.random.normal(0, 0.15, len(z_data))
    m_b_err = np.ones_like(z_data) * 0.15
else:
    df = pd.read_csv(file_path, sep='\\s+', header=0)
    z_data = df['zCMB'].values
    m_b = df['m_b_corr'].values
    m_b_err = df['m_b_corr_err_DIAG'].values

# Очистка
mask = (z_data > 0) & (m_b > 0) & (m_b_err > 0)
z_data = z_data[mask]
m_b = m_b[mask]
m_b_err = m_b_err[mask]

print(f"✅ Загружено {len(z_data)} точек")
print(f"   z: {z_data.min():.4f} ... {z_data.max():.4f}")

# ============================================================
# 3. ИНТЕГРАЛ
# ============================================================

z_grid = np.logspace(-4, 1, 200)

def compute_mu_grid(H0, model_func):
    mu_grid = np.zeros_like(z_grid)
    for i, zi in enumerate(z_grid):
        if zi < 1e-5:
            mu_grid[i] = np.nan
            continue
        z_int = np.linspace(0, zi, 50)
        H_int = model_func(z_int)
        integral = np.trapezoid(1.0 / H_int, z_int)
        DL = (1 + zi) * integral * 2997.9
        mu_grid[i] = 25 + 5 * np.log10(DL)
    return interp1d(z_grid, mu_grid, kind='cubic', bounds_error=False, fill_value=np.nan)

# ============================================================
# 4. МОДЕЛИ
# ============================================================

def H_psp(z, H0, alpha, beta):
    return H0 * np.sqrt(1 + alpha * (z/2.5)**2 * np.exp(beta * z))

def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1+z)**3 + (1-Om))

# ============================================================
# 5. ФУНКЦИИ ПРАВДОПОДОБИЯ
# ============================================================

def chi2_psp(params):
    H0, alpha, beta, M_B = params
    def H_func(z):
        return H_psp(z, H0, alpha, beta)
    mu_interp = compute_mu_grid(H0, H_func)
    mu_model = mu_interp(z_data) + M_B
    mask_valid = ~np.isnan(mu_model)
    return np.sum(((m_b[mask_valid] - mu_model[mask_valid]) / m_b_err[mask_valid])**2)

def chi2_lcdm(params):
    H0, Om, M_B = params
    def H_func(z):
        return H_lcdm(z, H0, Om)
    mu_interp = compute_mu_grid(H0, H_func)
    mu_model = mu_interp(z_data) + M_B
    mask_valid = ~np.isnan(mu_model)
    return np.sum(((m_b[mask_valid] - mu_model[mask_valid]) / m_b_err[mask_valid])**2)

# ============================================================
# 6. ОПТИМИЗАЦИЯ
# ============================================================

print("\n🔧 Оптимизация параметров...")

print("   → PSP...")
res_psp = minimize(chi2_psp, [67.4, 0.125, 0.35, -19.3], method='Nelder-Mead')
H0_psp, alpha_psp, beta_psp, M_B_psp = res_psp.x
chi2_psp_val = res_psp.fun

print("   → ΛCDM...")
res_lcdm = minimize(chi2_lcdm, [67.4, 0.315, -19.3], method='Nelder-Mead')
H0_lcdm, Om_lcdm, M_B_lcdm = res_lcdm.x
chi2_lcdm_val = res_lcdm.fun

# ============================================================
# 7. AIC / BIC
# ============================================================

N = len(z_data)
n_psp = 4
n_lcdm = 3

aic_psp = chi2_psp_val + 2 * n_psp
aic_lcdm = chi2_lcdm_val + 2 * n_lcdm

bic_psp = chi2_psp_val + n_psp * np.log(N)
bic_lcdm = chi2_lcdm_val + n_lcdm * np.log(N)

# ============================================================
# 8. РЕЗУЛЬТАТЫ
# ============================================================

print("\n" + "="*70)
print("📊 РЕЗУЛЬТАТЫ")
print("="*70)

print(f"\nТочек данных: {N}")

print("\n1️⃣ PSP:")
print(f"   H₀ = {H0_psp:.2f} км/с/Мпк")
print(f"   α  = {alpha_psp:.4f}")
print(f"   β  = {beta_psp:.4f}")
print(f"   M_B = {M_B_psp:.3f}")
print(f"   χ² = {chi2_psp_val:.2f}")
print(f"   AIC = {aic_psp:.2f}, BIC = {bic_psp:.2f}")

print("\n2️⃣ ΛCDM:")
print(f"   H₀ = {H0_lcdm:.2f} км/с/Мпк")
print(f"   Ωm = {Om_lcdm:.3f}")
print(f"   M_B = {M_B_lcdm:.3f}")
print(f"   χ² = {chi2_lcdm_val:.2f}")
print(f"   AIC = {aic_lcdm:.2f}, BIC = {bic_lcdm:.2f}")

print("\n3️⃣ СРАВНЕНИЕ:")
delta_aic = aic_lcdm - aic_psp
delta_bic = bic_lcdm - bic_psp

print(f"   ΔAIC (ΛCDM - PSP) = {delta_aic:.2f}")
print(f"   ΔBIC (ΛCDM - PSP) = {delta_bic:.2f}")

if delta_aic > 0:
    print("   ✅ PSP ЛУЧШЕ ΛCDM по AIC")
else:
    print("   ❌ ΛCDM ЛУЧШЕ PSP по AIC")

# ============================================================
# 9. СОХРАНЕНИЕ
# ============================================================

print("\n📈 Построение графиков...")

# Вычисляем модели для графика
def H_func_psp(z):
    return H_psp(z, H0_psp, alpha_psp, beta_psp)

def H_func_lcdm(z):
    return H_lcdm(z, H0_lcdm, Om_lcdm)

mu_interp_psp = compute_mu_grid(H0_psp, H_func_psp)
mu_interp_lcdm = compute_mu_grid(H0_lcdm, H_func_lcdm)

mu_psp = mu_interp_psp(z_data) + M_B_psp
mu_lcdm = mu_interp_lcdm(z_data) + M_B_lcdm

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

idx = np.argsort(z_data)

ax1 = axes[0, 0]
ax1.errorbar(z_data[idx], m_b[idx], yerr=m_b_err[idx], fmt='ko', capsize=1, markersize=1, alpha=0.2)
ax1.plot(z_data[idx], mu_psp[idx], 'b-', linewidth=2, label='PSP')
ax1.plot(z_data[idx], mu_lcdm[idx], 'r--', linewidth=2, label='ΛCDM')
ax1.set_xlabel('z')
ax1.set_ylabel('m_b')
ax1.set_title('Pantheon+')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.errorbar(z_data[idx], m_b[idx] - mu_psp[idx], yerr=m_b_err[idx], fmt='bo', capsize=1, markersize=1, alpha=0.3)
ax2.set_xlabel('z')
ax2.set_ylabel('Остатки (PSP)')
ax2.set_title('Остатки PSP')
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.errorbar(z_data[idx], m_b[idx] - mu_lcdm[idx], yerr=m_b_err[idx], fmt='ro', capsize=1, markersize=1, alpha=0.3)
ax3.set_xlabel('z')
ax3.set_ylabel('Остатки (ΛCDM)')
ax3.set_title('Остатки ΛCDM')
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
z_plot = np.linspace(0.01, 3, 100)
ax4.plot(z_plot, H_psp(z_plot, H0_psp, alpha_psp, beta_psp), 'b-', linewidth=2, label='PSP')
ax4.plot(z_plot, H_lcdm(z_plot, H0_lcdm, Om_lcdm), 'r--', linewidth=2, label='ΛCDM')
ax4.set_xlabel('z')
ax4.set_ylabel('H(z) [км/с/Мпк]')
ax4.set_title('Сравнение H(z)')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('psp_vs_lcdm_final.png', dpi=150)
print("✅ График: psp_vs_lcdm_final.png")

with open('psp_vs_lcdm_final_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("PSP vs ΛCDM: ФИНАЛЬНОЕ СРАВНЕНИЕ\n")
    f.write("="*70 + "\n\n")
    f.write(f"Точек данных: {N}\n\n")
    f.write("PSP:\n")
    f.write(f"  H₀ = {H0_psp:.2f} км/с/Мпк\n")
    f.write(f"  α  = {alpha_psp:.4f}\n")
    f.write(f"  β  = {beta_psp:.4f}\n")
    f.write(f"  M_B = {M_B_psp:.3f}\n")
    f.write(f"  χ² = {chi2_psp_val:.2f}\n")
    f.write(f"  AIC = {aic_psp:.2f}\n")
    f.write(f"  BIC = {bic_psp:.2f}\n\n")
    f.write("ΛCDM:\n")
    f.write(f"  H₀ = {H0_lcdm:.2f} км/с/Мпк\n")
    f.write(f"  Ωm = {Om_lcdm:.3f}\n")
    f.write(f"  M_B = {M_B_lcdm:.3f}\n")
    f.write(f"  χ² = {chi2_lcdm_val:.2f}\n")
    f.write(f"  AIC = {aic_lcdm:.2f}\n")
    f.write(f"  BIC = {bic_lcdm:.2f}\n\n")
    f.write(f"ΔAIC = {delta_aic:.2f}\n")
    f.write(f"ΔBIC = {delta_bic:.2f}\n")

print("\n✅ Результаты: psp_vs_lcdm_final_results.txt")
print("="*70)
print("✅ ГОТОВО!")

