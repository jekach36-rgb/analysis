#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
СРАВНЕНИЕ УСТОЙЧИВОСТИ PSP И ΛCDM
====================================
- Тест на зашумленных данных
- Оценка дисперсии и смещения
- Вывод: какая модель устойчивее

Автор: Е.В. Чернокнижный
Версия: 1.1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ============================================================
# 1. ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ
# ============================================================

def H_psp(z, H0, alpha, beta):
    return H0 * np.sqrt(1 + alpha * (z/2.5)**2 * np.exp(beta * z))

def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1+z)**3 + (1-Om))

def mu_model(z, H0, model_func):
    """Модуль расстояния через интеграл"""
    mu_arr = []
    for zi in z:
        if zi < 0.0001:
            mu_arr.append(42)
            continue
        try:
            z_int = np.linspace(0, zi, 50)
            H_int = np.array([model_func(zp) for zp in z_int])
            integral = np.trapz(1.0 / H_int, z_int)
            DL = (1 + zi) * integral * 2997.9
            mu_arr.append(25 + 5 * np.log10(DL))
        except:
            mu_arr.append(42)
    return np.array(mu_arr)

def chi2_psp(params, z, mu, mu_err):
    H0, alpha, beta = params
    def H_func(zp):
        return H_psp(zp, H0, alpha, beta)
    mu_model_arr = mu_model(z, H0, H_func)
    return np.sum(((mu - mu_model_arr) / mu_err)**2)

def chi2_lcdm(params, z, mu, mu_err):
    H0, Om = params
    def H_func(zp):
        return H_lcdm(zp, H0, Om)
    mu_model_arr = mu_model(z, H0, H_func)
    return np.sum(((mu - mu_model_arr) / mu_err)**2)

# ============================================================
# 2. СИНТЕТИЧЕСКИЕ ДАННЫЕ С ШУМОМ
# ============================================================

np.random.seed(42)
z_data = np.linspace(0.01, 0.5, 50)
mu_data = 42 + 5 * np.log10(z_data * (1 + z_data/2))
mu_err = np.ones_like(z_data) * 0.15

def add_noise(mu, z, amplitude=0.3):
    """Добавляем шум: сверхновые + гамма-всплески + прецессия"""
    mu_noisy = mu.copy()
    
    # 1. Сверхновые — резкие пики
    sn_peaks = np.random.choice(len(z), size=5, replace=False)
    mu_noisy[sn_peaks] += np.random.normal(0, 0.5, size=5)
    
    # 2. Гамма-всплески — случайные смещения
    grb = np.random.normal(0, amplitude, size=len(z))
    mu_noisy += grb
    
    # 3. Прецессия — синусоидальный дрейф
    mu_noisy += 0.1 * np.sin(2 * np.pi * z / 0.05)
    
    return mu_noisy

mu_noisy = add_noise(mu_data, z_data)  # <-- ИСПРАВЛЕНО!

# ============================================================
# 3. ПРОГОН ОБЕИХ МОДЕЛЕЙ
# ============================================================

print("="*60)
print("ТЕСТ НА УСТОЙЧИВОСТЬ PSP vs ΛCDM")
print("="*60)

# Без шума
res_psp_clean = minimize(chi2_psp, [67.4, 0.125, 0.35], args=(z_data, mu_data, mu_err), method='Nelder-Mead')
res_lcdm_clean = minimize(chi2_lcdm, [67.4, 0.315], args=(z_data, mu_data, mu_err), method='Nelder-Mead')

# С шумом
res_psp_noisy = minimize(chi2_psp, [67.4, 0.125, 0.35], args=(z_data, mu_noisy, mu_err), method='Nelder-Mead')
res_lcdm_noisy = minimize(chi2_lcdm, [67.4, 0.315], args=(z_data, mu_noisy, mu_err), method='Nelder-Mead')

# ============================================================
# 4. ОЦЕНКА СМЕЩЕНИЯ И ДИСПЕРСИИ
# ============================================================

print("\n1️⃣ ПАРАМЕТРЫ PSP:")
print(f"   Без шума: H0 = {res_psp_clean.x[0]:.2f}, α = {res_psp_clean.x[1]:.4f}, β = {res_psp_clean.x[2]:.4f}")
print(f"   С шумом:  H0 = {res_psp_noisy.x[0]:.2f}, α = {res_psp_noisy.x[1]:.4f}, β = {res_psp_noisy.x[2]:.4f}")
print(f"   Смещение: ΔH0 = {abs(res_psp_noisy.x[0] - res_psp_clean.x[0]):.2f}")

print("\n2️⃣ ПАРАМЕТРЫ ΛCDM:")
print(f"   Без шума: H0 = {res_lcdm_clean.x[0]:.2f}, Ωm = {res_lcdm_clean.x[1]:.3f}")
print(f"   С шумом:  H0 = {res_lcdm_noisy.x[0]:.2f}, Ωm = {res_lcdm_noisy.x[1]:.3f}")
print(f"   Смещение: ΔH0 = {abs(res_lcdm_noisy.x[0] - res_lcdm_clean.x[0]):.2f}")

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

print("\n📈 Построение графиков...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(z_data, mu_data, 'bo-', label='Без шума')
ax1.plot(z_data, mu_noisy, 'rx-', label='С шумом')
ax1.set_xlabel('z')
ax1.set_ylabel('μ')
ax1.set_title('Данные без шума и с шумом')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.bar(['PSP', 'ΛCDM'], [abs(res_psp_noisy.x[0] - res_psp_clean.x[0]), abs(res_lcdm_noisy.x[0] - res_lcdm_clean.x[0])], color=['blue', 'orange'])
ax2.set_ylabel('Смещение H0 (км/с/Мпк)')
ax2.set_title('Устойчивость H0 к шуму')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.hist([res_psp_clean.x[0], res_psp_noisy.x[0]], bins=10, alpha=0.5, label=['PSP без шума', 'PSP с шумом'])
ax3.hist([res_lcdm_clean.x[0], res_lcdm_noisy.x[0]], bins=10, alpha=0.5, label=['ΛCDM без шума', 'ΛCDM с шумом'])
ax3.set_xlabel('H0')
ax3.set_ylabel('Частота')
ax3.set_title('Распределение H0')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.bar(['PSP', 'ΛCDM'], [res_psp_noisy.fun, res_lcdm_noisy.fun], color=['blue', 'orange'])
ax4.set_ylabel('χ² с шумом')
ax4.set_title('Качество подгонки')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stability_test.png', dpi=150)
print("✅ График сохранён как 'stability_test.png'")

# ============================================================
# 6. ВЫВОДЫ
# ============================================================

print("\n" + "="*60)
print("📊 ВЫВОДЫ ПО УСТОЙЧИВОСТИ")
print("="*60)

psp_stability = abs(res_psp_noisy.x[0] - res_psp_clean.x[0])
lcdm_stability = abs(res_lcdm_noisy.x[0] - res_lcdm_clean.x[0])

if psp_stability < lcdm_stability:
    print("""
    ✅ PSP УСТОЙЧИВЕЕ ΛCDM:
       - Смещение H0 у PSP меньше, чем у ΛCDM
       - PSP лучше держит параметры при шуме
       - PSP предсказуема даже в нестабильной среде
    """)
else:
    print("""
    ❌ ΛCDM УСТОЙЧИВЕЕ PSP:
       - ΛCDM меньше реагирует на шум
       - PSP требует больше данных для стабильности
    """)

print("="*60)
print("🎯 ТЕСТ ЗАВЕРШЁН!")
print("="*60)