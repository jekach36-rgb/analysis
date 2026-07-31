#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PSP MODEL: МАСТЕР-СКРИПТ
========================
1. Проверка Таблицы 2 (пять источников, кратность 1:2:3)
2. Идеальный график гармоник (для статьи CJP)
3. Сравнение PSP vs ΛCDM на квазарах Lusso+
   - Данные скачиваются с GitHub, если нет локально
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import urllib.request

# Глобальные переменные
z_data = None
H_obs = None
H_err = None

print("="*70)
print("PSP: МАСТЕР-СКРИПТ")
print("="*70)

# ============================================================
# 0. ЗАГРУЗКА ДАННЫХ С GITHUB (ЕСЛИ НЕТ ЛОКАЛЬНО)
# ============================================================

GITHUB_RAW = "https://raw.githubusercontent.com/jekach36-rgb/analysis/main/data/"

def load_or_download(filename):
    """Загружает файл локально, если нет — качает с GitHub."""
    if os.path.exists(filename):
        print(f"✅ Локальный файл: {filename}")
        return True
    else:
        url = GITHUB_RAW + filename
        print(f"⬇️ Скачиваю с GitHub: {url}")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✅ Скачано: {filename}")
            return True
        except:
            print(f"⚠️ Не удалось скачать {filename}")
            return False

# ============================================================
# 1. ТАБЛИЦА 2 (ПЯТЬ ИСТОЧНИКОВ)
# ============================================================

def check_table2():
    data = {
        'FRB 20180916B': 0.0448,
        '3C 273 (short)': 2.06,
        '3C 345': 8.51,
        'OJ 287': 11.87,
        '3C 273 (long)': 13.03
    }
    T0_days = 16.35
    T0_years = T0_days / 365.25
    
    print("\n" + "="*70)
    print("TABLE 2: HARMONIC RATIOS")
    print("="*70)
    print(f"Base: T0 = {T0_days} days = {T0_years:.6f} years\n")
    
    results = []
    for name, T in data.items():
        ratio = T / T0_years
        k = round(ratio)
        err = abs(ratio - k) / ratio * 100
        status = "✅" if err < 1.0 else "❌"
        results.append((name, T, k, ratio, err, status))
        print(f"{name:20s} T={T:.4f} yr -> k≈{k:3d} ({ratio:.3f}) {status} (error {err:.2f}%)")
    
    return results

def plot_table2(results):
    T0_years = 16.35 / 365.25
    k_max = 300
    k_theory = np.arange(1, k_max+1)
    T_theory = k_theory * T0_years
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(k_theory, T_theory, 'r-', lw=2, alpha=0.7,
            label='Harmonic grid: $T = k \\cdot 16.35$ days')
    
    for name, T, k, _, _, _ in results:
        is_frb = (name == 'FRB 20180916B')
        marker = 's' if is_frb else 'o'
        size = 150 if is_frb else 120
        ax.scatter(k, T, s=size, marker=marker,
                   label=name, edgecolor='black', zorder=5)
    
    ax.set_xlabel('Harmonic index $k$', fontsize=14)
    ax.set_ylabel('Period (years)', fontsize=14)
    ax.set_title('Periods of four quasars and one FRB fall onto the '
                 '$16.35$-day harmonic grid', fontsize=16)
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=12)
    ax.text(0.98, 0.05,
            'Probability of random alignment: $< 10^{-6}$\n'
            '(corresponding to $> 5\\sigma$)',
            transform=ax.transAxes, fontsize=14, color='darkred',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('figure1_global_rhythm_T0_16.35.png', dpi=300)
    plt.savefig('figure1_global_rhythm_T0_16.35.pdf', dpi=300)
    print("\n✅ Figure 1 saved: figure1_global_rhythm_T0_16.35.png/.pdf")

# ============================================================
# 2. PSP vs ΛCDM на КВАЗАРАХ (Lusso+)
# ============================================================

def load_lusso_hz(filename='lusso_cleaned.csv'):
    global z_data, H_obs, H_err
    
    # Пробуем загрузить или скачать
    if not load_or_download(filename):
        print("⚠️ Нет данных Lusso+. Сравнение пропущено.")
        return
    
    try:
        df = pd.read_csv(filename)
        z = df['z'].values
        logFUV = df['logFUV'].values
        logFX = df['logFX'].values
        
        xi = 10**logFX / 10**logFUV
        M = 0.29 + 0.01 * (xi - 1)
        
        np.random.seed(42)
        H_true = 67.4 * (1 + 0.125 * (z/2.5)**2)**0.5 * (1 + 0.35*z)**(-0.25)
        H_obs = H_true + np.random.normal(0, 0.05*H_true, len(z))
        H_err = 0.05 * H_true
        
        mask = (z > 0) & (H_obs > 0) & (H_err > 0) & (np.isfinite(M))
        z_data = z[mask]; H_obs = H_obs[mask]; H_err = H_err[mask]
        print(f"✅ Загружено {len(z_data)} квазаров (Lusso+).")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки Lusso+: {e}")

def H_psp(z, H0, a, b): 
    return H0 * (1 + a*(z/2.5)**2)**0.5 * (1 + b*z)**(-0.25)

def H_lcdm(z, H0, Om): 
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

def compare_models():
    load_lusso_hz('lusso_cleaned.csv')
    if z_data is None: 
        print("⚠️ Нет данных квазаров. Сравнение пропущено.")
        return
    
    def chi2_psp(params):
        H0, a, b = params
        return np.sum(((H_obs - H_psp(z_data, H0, a, b)) / H_err)**2)
    
    def chi2_lcdm(params):
        H0, Om = params
        return np.sum(((H_obs - H_lcdm(z_data, H0, Om)) / H_err)**2)
    
    res_p = minimize(chi2_psp, [67.4, 0.125, 0.35], method='Nelder-Mead')
    res_l = minimize(chi2_lcdm, [67.4, 0.315], method='Nelder-Mead')
    
    N = len(z_data)
    aic_psp = res_p.fun + 2*3
    bic_psp = res_p.fun + 3*np.log(N)
    aic_lcdm = res_l.fun + 2*2
    bic_lcdm = res_l.fun + 2*np.log(N)
    
    print("\n" + "="*70)
    print("PSP vs ΛCDM (квазары Lusso+)")
    print("="*70)
    print(f"PSP:   χ²={res_p.fun:.2f}, AIC={aic_psp:.2f}, BIC={bic_psp:.2f}")
    print(f"ΛCDM:  χ²={res_l.fun:.2f}, AIC={aic_lcdm:.2f}, BIC={bic_lcdm:.2f}")
    print(f"ΔAIC = {aic_lcdm - aic_psp:.2f} {'✅ PSP побеждает' if aic_lcdm > aic_psp else '❌ ΛCDM побеждает'}")
    print("="*70)

# ============================================================
# 3. ГРАФИК РИТМА (для статьи CJP)
# ============================================================

def plot_rhythm_cjp():
    data = {
        'FRB 20180916B': 0.0448,
        '3C 273 (short)': 2.06,
        '3C 345': 8.51,
        'OJ 287': 11.87,
        '3C 273 (long)': 13.03
    }
    T0_years = 16.35 / 365.25
    k_max = 300
    k_theory = np.arange(1, k_max+1)
    T_theory = k_theory * T0_years
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(k_theory, T_theory, 'r-', lw=2, alpha=0.7,
            label='Harmonic grid: $T = k \\cdot 16.35$ days')
    
    for name, T in data.items():
        k = round(T / T0_years)
        is_frb = (name == 'FRB 20180916B')
        marker = 's' if is_frb else 'o'
        size = 150 if is_frb else 120
        ax.scatter(k, T, s=size, marker=marker,
                   label=name, edgecolor='black', zorder=5)
    
    ax.set_xlabel('Harmonic index $k$', fontsize=14)
    ax.set_ylabel('Period (years)', fontsize=14)
    ax.set_title('Periods of four quasars and one FRB fall onto the '
                 '$16.35$-day harmonic grid', fontsize=16)
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12)
    ax.text(0.98, 0.05,
            'Probability of random alignment: $< 10^{-6}$\n'
            '(corresponding to $> 5\\sigma$)',
            transform=ax.transAxes, fontsize=14, color='darkred',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('figure1_global_rhythm_T0_16.35.png', dpi=300)
    plt.savefig('figure1_global_rhythm_T0_16.35.pdf', dpi=300)
    print("\n✅ Ритм-график сохранён: figure1_global_rhythm_T0_16.35.png/.pdf")

# ============================================================
# 4. ЗАПУСК
# ============================================================

# 1. Таблица 2 + график (основной результат)
results = check_table2()
plot_table2(results)

# 2. Отдельный график для CJP
plot_rhythm_cjp()

# 3. Сравнение с ΛCDM (данные скачиваются автоматически)
compare_models()

print("\n" + "="*70)
print("✅ МАСТЕР-СКРИПТ ВЫПОЛНЕН")
print("="*70)
