#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
БЫСТРЫЙ АНАЛИЗ PSP
==================
- Используем предварительно вычисленные интегралы (сетка)
- Минимизация без тормозов
- Свободный поиск параметров (только γ < 0)

Автор: Е.В. Чернокнижный
Версия: 13.1 (ИСПРАВЛЕННАЯ)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("🔄 Загрузка данных Pantheon+...")
df = pd.read_csv("D:\\Pantheon_SH0ES.dat", sep='\\s+', header=0)

z_data = df['zCMB'].values
mu_data = df['m_b_corr'].values
mu_err = df['m_b_corr_err_DIAG'].values

# Очистка
mask = (z_data > 0) & (mu_data > 0) & (mu_err > 0) & np.isfinite(z_data) & np.isfinite(mu_data) & np.isfinite(mu_err)
z_data = z_data[mask]
mu_data = mu_data[mask]
mu_err = mu_err[mask]

print(f"✅ Загружено {len(z_data)} точек Pantheon+")
print(f"   z: {z_data.min():.4f} ... {z_data.max():.4f}")
print()

# ============================================================
# 2. ПРЕДВАРИТЕЛЬНЫЙ РАСЧЁТ ИНТЕГРАЛОВ (БЫСТРО)
# ============================================================

# Создаём сетку по z для интерполяции
z_grid = np.logspace(-4, 1, 200)  # от 0.0001 до 10

def compute_mu_grid(H0, model_func):
    """
    Быстрое вычисление mu(z) на сетке
    """
    mu_grid = np.zeros_like(z_grid)
    for i, zi in enumerate(z_grid):
        if zi < 1e-5:
            mu_grid[i] = 42
            continue
        # Численное интегрирование
        z_int = np.linspace(0, zi, 50)
        H_int = model_func(z_int)
        # ИСПРАВЛЕНО: np.trapz -> np.trapezoid
        integral = np.trapezoid(1.0 / H_int, z_int)
        DL = (1 + zi) * integral * 2997.9
        mu_grid[i] = 25 + 5 * np.log10(DL)
    return interp1d(z_grid, mu_grid, kind='cubic', fill_value='extrapolate')

# ============================================================
# 3. МОДЕЛИ
# ============================================================

def H_psp(z, H0, alpha, beta):
    return H0 * np.sqrt(1 + alpha * (z/2.5)**2 * np.exp(beta * z))

def H_psp_vortex(z, H0, alpha, beta, gamma, z_peak, dz):
    return H_psp(z, H0, alpha, beta) + gamma * np.exp(-0.5 * ((z - z_peak) / dz)**2)

def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1+z)**3 + (1-Om))

# ============================================================
# 4. ФУНКЦИИ ПРАВДОПОДОБИЯ (БЫСТРЫЕ)
# ============================================================

def chi2_psp(params):
    H0, alpha, beta = params
    def H_func(z):
        return H_psp(z, H0, alpha, beta)
    mu_interp = compute_mu_grid(H0, H_func)
    mu_model = mu_interp(z_data)
    return np.sum(((mu_data - mu_model) / mu_err)**2)

def chi2_psp_vortex(params):
    H0, alpha, beta, gamma, z_peak, dz = params
    def H_func(z):
        return H_psp_vortex(z, H0, alpha, beta, gamma, z_peak, dz)
    mu_interp = compute_mu_grid(H0, H_func)
    mu_model = mu_interp(z_data)
    return np.sum(((mu_data - mu_model) / mu_err)**2)

def chi2_lcdm(params):
    H0, Om = params
    def H_func(z):
        return H_lcdm(z, H0, Om)
    mu_interp = compute_mu_grid(H0, H_func)
    mu_model = mu_interp(z_data)
    return np.sum(((mu_data - mu_model) / mu_err)**2)

# ============================================================
# 5. ОПТИМИЗАЦИЯ
# ============================================================

print("🔧 Поиск наилучших параметров...")
print("   Свободный поиск (только γ < 0)")
print()

# PSP (базовая)
print("   → Базовая PSP...")
res_psp = minimize(chi2_psp, [67.4, 0.125, 0.35], method='Nelder-Mead')
H0_psp, alpha_psp, beta_psp = res_psp.x
chi2_psp_val = res_psp.fun

# PSP + вихри (только γ < 0)
print("   → PSP + вихри...")
bounds_vortex = Bounds(
    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    [np.inf, np.inf, np.inf, -1e-6, np.inf, np.inf]
)
res_vortex = minimize(chi2_psp_vortex, [67.4, 0.125, 0.35, -50, 0.24, 0.05],
                      method='L-BFGS-B', bounds=bounds_vortex)
H0_v, alpha_v, beta_v, gamma_v, z_peak_v, dz_v = res_vortex.x
chi2_vortex_val = res_vortex.fun

# ΛCDM
print("   → ΛCDM...")
res_lcdm = minimize(chi2_lcdm, [67.4, 0.315], method='Nelder-Mead')
H0_lcdm, Om_lcdm = res_lcdm.x
chi2_lcdm_val = res_lcdm.fun

# ============================================================
# 6. РЕЗУЛЬТАТЫ
# ============================================================

print("\n" + "="*70)
print("📊 РЕЗУЛЬТАТЫ СВОБОДНОГО ПОИСКА")
print("="*70)

print(f"\nКоличество точек: {len(z_data)}")
print(f"Диапазон z (CMB): {z_data.min():.4f} ... {z_data.max():.4f}")

print("\n1️⃣ PSP (базовая):")
print(f"   H₀  = {H0_psp:.2f} км/с/Мпк")
print(f"   α   = {alpha_psp:.4f}")
print(f"   β   = {beta_psp:.4f}")
print(f"   χ²  = {chi2_psp_val:.2f}")

print("\n2️⃣ PSP + ВИХРИ (свободный поиск):")
print(f"   H₀      = {H0_v:.2f} км/с/Мпк")
print(f"   α       = {alpha_v:.4f}")
print(f"   β       = {beta_v:.4f}")
print(f"   γ       = {gamma_v:.3f} (амплитуда вихрей)")
print(f"   z_peak  = {z_peak_v:.3f}")
print(f"   dz      = {dz_v:.3f}")
print(f"   χ²      = {chi2_vortex_val:.2f}")

print("\n3️⃣ ΛCDM:")
print(f"   H₀  = {H0_lcdm:.2f} км/с/Мпк")
print(f"   Ωm  = {Om_lcdm:.3f}")
print(f"   χ²  = {chi2_lcdm_val:.2f}")

# Сравнение
print("\n4️⃣ СРАВНЕНИЕ:")
if chi2_vortex_val < chi2_psp_val:
    imp = chi2_psp_val - chi2_vortex_val
    print(f"   ✅ ВИХРИ УЛУЧШАЮТ PSP на {imp:.2f} пунктов χ²")
else:
    print(f"   ❌ Вихри НЕ улучшают PSP")
    
if chi2_vortex_val < chi2_lcdm_val:
    print(f"   ✅ PSP+вихри лучше ΛCDM на {chi2_lcdm_val - chi2_vortex_val:.2f} χ²")
else:
    print(f"   ❌ ΛCDM лучше PSP+вихри на {chi2_vortex_val - chi2_lcdm_val:.2f} χ²")

print("\n5️⃣ ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
print(f"   γ = {gamma_v:.3f}")
if gamma_v < 0:
    print("   ✅ γ < 0 → B_tor > B_BH (поле тора доминирует)")
else:
    print("   ❌ γ > 0 → B_tor < B_BH (поле квазара доминирует)")

# ============================================================
# 7. ГРАФИК
# ============================================================

print("\n📈 Построение графиков...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

def H_func_psp(z):
    return H_psp(z, H0_psp, alpha_psp, beta_psp)

def H_func_vortex(z):
    return H_psp_vortex(z, H0_v, alpha_v, beta_v, gamma_v, z_peak_v, dz_v)

# Модельные mu
mu_interp_psp = compute_mu_grid(H0_psp, H_func_psp)
mu_interp_vortex = compute_mu_grid(H0_v, H_func_vortex)

mu_psp = mu_interp_psp(z_data)
mu_vortex = mu_interp_vortex(z_data)

# График 1: SN Ia
ax1 = axes[0]
idx = np.argsort(z_data)
ax1.errorbar(z_data[idx], mu_data[idx], yerr=mu_err[idx], fmt='ko', capsize=1, markersize=2, alpha=0.2)
ax1.plot(z_data[idx], mu_psp[idx], 'b-', linewidth=2, label='PSP')
ax1.plot(z_data[idx], mu_vortex[idx], 'r-', linewidth=2, label='PSP+вихри')
ax1.set_xlabel('z (CMB)')
ax1.set_ylabel('m_b_corr')
ax1.set_title(f'Pantheon+ ({len(z_data)} точек)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Остатки
ax2 = axes[1]
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.errorbar(z_data[idx], mu_data[idx] - mu_psp[idx], yerr=mu_err[idx], fmt='bo', capsize=1, markersize=2, alpha=0.4, label='PSP')
ax2.errorbar(z_data[idx], mu_data[idx] - mu_vortex[idx], yerr=mu_err[idx], fmt='rx', capsize=1, markersize=2, alpha=0.4, label='PSP+вихри')
ax2.set_xlabel('z (CMB)')
ax2.set_ylabel('Остатки')
ax2.set_title('Остатки моделей')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psp_analysis_full.png', dpi=150)
print("✅ График сохранён как 'psp_analysis_full.png'")

# ============================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

with open('psp_results_full.txt', 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТЫ СВОБОДНОГО ПОИСКА PSP\n")
    f.write("="*60 + "\n\n")
    f.write(f"Количество точек Pantheon+: {len(z_data)}\n")
    f.write(f"Диапазон z (CMB): {z_data.min():.4f} ... {z_data.max():.4f}\n\n")
    f.write(f"PSP: H0={H0_psp:.2f}, alpha={alpha_psp:.4f}, beta={beta_psp:.4f}, chi2={chi2_psp_val:.2f}\n")
    f.write(f"PSP+вихри: H0={H0_v:.2f}, alpha={alpha_v:.4f}, beta={beta_v:.4f}, gamma={gamma_v:.3f}, z_peak={z_peak_v:.3f}, dz={dz_v:.3f}, chi2={chi2_vortex_val:.2f}\n")
    f.write(f"ΛCDM: H0={H0_lcdm:.2f}, Omega_m={Om_lcdm:.3f}, chi2={chi2_lcdm_val:.2f}\n")

print("\n✅ Результаты сохранены в 'psp_results_full.txt'")
print("\n" + "="*70)
print("🎯 СВОБОДНЫЙ ПОИСК ЗАВЕРШЁН!")
print("="*70)