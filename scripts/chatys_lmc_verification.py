#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА ЗАКОНА ФУЖЕРА: ВЫБОРКА LMC (Chatys+ 2019)
Автор: Чернокнижный Евгений Валерьевич
Дата: 8 августа 2026
Источник: J/MNRAS/487/4832 (VizieR)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

print("="*70)
print("ПРОВЕРКА ЗАКОНА ФУЖЕРА: ВЫБОРКА LMC (206 объектов)")
print("="*70)

T0 = 16.35

# ============================================================
# ДАННЫЕ ИЗ КАТАЛОГА CHATYS+ 2019 (137 объектов с периодами)
# ============================================================
data = [
    {"id": "001", "periods": [610, 1950], "kmag": 7.76},
    {"id": "003", "periods": [2290], "kmag": 8.36},
    {"id": "004", "periods": [490, 674, 2681], "kmag": 8.24},
    {"id": "005", "periods": [365], "kmag": 8.74},
    {"id": "007", "periods": [5000], "kmag": 8.36},
    {"id": "009", "periods": [4340], "kmag": 7.61},
    {"id": "010", "periods": [755, 2350], "kmag": 7.20},
    {"id": "011", "periods": [3970], "kmag": 8.66},
    {"id": "012", "periods": [1560], "kmag": 8.43},
    {"id": "014", "periods": [435, 1500], "kmag": 7.37},
    {"id": "016", "periods": [565], "kmag": 7.66},
    {"id": "017", "periods": [715, 3070], "kmag": 7.66},
    {"id": "019", "periods": [610], "kmag": 7.70},
    {"id": "020", "periods": [335], "kmag": 7.97},
    {"id": "021", "periods": [320, 2510], "kmag": 8.45},
    {"id": "023", "periods": [610], "kmag": 7.97},
    {"id": "024", "periods": [950], "kmag": 7.32},
    {"id": "027", "periods": [2375], "kmag": 8.72},
    {"id": "028", "periods": [4100], "kmag": 8.11},
    {"id": "029", "periods": [3413, 322], "kmag": 8.39},
    {"id": "030", "periods": [990], "kmag": 6.78},
    {"id": "031", "periods": [2950], "kmag": 8.03},
    {"id": "032", "periods": [556], "kmag": 8.40},
    {"id": "033", "periods": [405], "kmag": 8.38},
    {"id": "034", "periods": [695], "kmag": 7.64},
    {"id": "035", "periods": [2760], "kmag": 8.11},
    {"id": "037", "periods": [3675], "kmag": 8.32},
    {"id": "038", "periods": [2667], "kmag": 8.75},
    {"id": "039", "periods": [640], "kmag": 7.04},
    {"id": "040", "periods": [520], "kmag": 8.02},
    {"id": "042", "periods": [615], "kmag": 7.69},
    {"id": "043", "periods": [765], "kmag": 7.59},
    {"id": "044", "periods": [805], "kmag": 7.42},
    {"id": "045", "periods": [575, 265], "kmag": 7.82},
    {"id": "046", "periods": [835], "kmag": 8.57},
    {"id": "047", "periods": [3650, 375], "kmag": 7.60},
    {"id": "048", "periods": [1750], "kmag": 8.22},
    {"id": "049", "periods": [520], "kmag": 7.98},
    {"id": "050", "periods": [1635], "kmag": 8.14},
    {"id": "052", "periods": [285], "kmag": 8.51},
    {"id": "053", "periods": [1250, 380], "kmag": 8.90},
    {"id": "054", "periods": [675], "kmag": 7.74},
    {"id": "056", "periods": [4900], "kmag": 6.81},
    {"id": "057", "periods": [510, 2470], "kmag": 7.99},
    {"id": "060", "periods": [510], "kmag": 8.03},
    {"id": "061", "periods": [655, 3510], "kmag": 7.70},
    {"id": "062", "periods": [405, 2650], "kmag": 8.48},
    {"id": "063", "periods": [3210], "kmag": 7.26},
    {"id": "065", "periods": [435, 2250], "kmag": 8.55},
    {"id": "067", "periods": [3700], "kmag": 8.78},
    {"id": "070", "periods": [3670], "kmag": 8.32},
    {"id": "071", "periods": [1230], "kmag": 7.97},
    {"id": "072", "periods": [650], "kmag": 7.84},
    {"id": "074", "periods": [3735], "kmag": 7.60},
    {"id": "076", "periods": [1250], "kmag": 8.29},
    {"id": "077", "periods": [1270], "kmag": 8.15},
    {"id": "079", "periods": [510, 4100], "kmag": 8.09},
    {"id": "081", "periods": [1665], "kmag": 8.31},
    {"id": "082", "periods": [3735, 370], "kmag": 8.38},
    {"id": "083", "periods": [770], "kmag": 7.48},
    {"id": "085", "periods": [480], "kmag": 8.05},
    {"id": "087", "periods": [2570], "kmag": 8.58},
    {"id": "092", "periods": [640], "kmag": 7.90},
    {"id": "093", "periods": [2275], "kmag": 8.57},
    {"id": "094", "periods": [3390, 370], "kmag": 8.35},
    {"id": "097", "periods": [765], "kmag": 7.30},
    {"id": "098", "periods": [500], "kmag": 8.73},
    {"id": "099", "periods": [845], "kmag": 6.89},
    {"id": "100", "periods": [520, 3775], "kmag": 7.88},
    {"id": "101", "periods": [295], "kmag": 8.41},
    {"id": "102", "periods": [750, 365], "kmag": 7.79},
    {"id": "103", "periods": [575], "kmag": 7.97},
    {"id": "104", "periods": [3445], "kmag": 8.77},
    {"id": "105", "periods": [810, 1900], "kmag": 8.81},
    {"id": "106", "periods": [3935], "kmag": 8.64},
    {"id": "107", "periods": [625, 2665], "kmag": 7.45},
    {"id": "108", "periods": [3570], "kmag": 8.60},
    {"id": "109", "periods": [2235], "kmag": 8.48},
    {"id": "111", "periods": [365, 4330, 670], "kmag": 7.55},
    {"id": "112", "periods": [1160], "kmag": 8.85},
    {"id": "113", "periods": [4265], "kmag": 7.59},
    {"id": "114", "periods": [2155], "kmag": 8.76},
    {"id": "115", "periods": [190, 2770], "kmag": 8.43},
    {"id": "116", "periods": [200, 2610], "kmag": 8.83},
    {"id": "118", "periods": [365], "kmag": 8.33},
    {"id": "120", "periods": [652], "kmag": 7.63},
    {"id": "121", "periods": [565, 3175], "kmag": 7.63},
    {"id": "122", "periods": [3255], "kmag": 8.22},
    {"id": "124", "periods": [725], "kmag": 7.37},
    {"id": "128", "periods": [465], "kmag": 7.96},
    {"id": "129", "periods": [965, 365], "kmag": 8.63},
    {"id": "131", "periods": [440], "kmag": 8.05},
    {"id": "132", "periods": [485, 2350], "kmag": 8.61},
    {"id": "134", "periods": [350, 1669], "kmag": 7.82},
    {"id": "136", "periods": [380, 2350], "kmag": 8.49},
    {"id": "137", "periods": [645], "kmag": 7.90},
    {"id": "140", "periods": [295, 1550], "kmag": 8.89},
    {"id": "141", "periods": [2665], "kmag": 8.28},
    {"id": "143", "periods": [710], "kmag": 8.02},
    {"id": "144", "periods": [415, 575, 2550], "kmag": 8.34},
    {"id": "145", "periods": [310, 850, 1950], "kmag": 8.23},
    {"id": "146", "periods": [755], "kmag": 7.26},
    {"id": "148", "periods": [2620], "kmag": 8.04},
    {"id": "149", "periods": [1700], "kmag": 8.45},
    {"id": "154", "periods": [750], "kmag": 7.50},
    {"id": "155", "periods": [465], "kmag": 7.81},
    {"id": "158", "periods": [1800], "kmag": 8.23},
    {"id": "159", "periods": [3560], "kmag": 8.72},
    {"id": "166", "periods": [1965], "kmag": 8.30},
    {"id": "167", "periods": [4560], "kmag": 8.51},
    {"id": "170", "periods": [2200], "kmag": 8.29},
    {"id": "173", "periods": [3215], "kmag": 8.78},
    {"id": "175", "periods": [1300], "kmag": 7.44},
    {"id": "177", "periods": [2570], "kmag": 7.54},
    {"id": "178", "periods": [715, 2515], "kmag": 7.49},
    {"id": "179", "periods": [3445], "kmag": 7.98},
    {"id": "180", "periods": [2315], "kmag": 7.77},
    {"id": "182", "periods": [435, 2455], "kmag": 7.82},
    {"id": "183", "periods": [2100], "kmag": 8.45},
    {"id": "184", "periods": [2778], "kmag": 8.41},
    {"id": "185", "periods": [3215], "kmag": 8.40},
    {"id": "186", "periods": [265, 1530], "kmag": 8.56},
    {"id": "187", "periods": [4100, 1055], "kmag": 8.68},
    {"id": "188", "periods": [2245], "kmag": 8.82},
    {"id": "189", "periods": [445], "kmag": 8.37},
    {"id": "195", "periods": [2550], "kmag": 8.22},
    {"id": "199", "periods": [3355], "kmag": 8.05},
    {"id": "200", "periods": [415], "kmag": 8.34},
    {"id": "208", "periods": [2665], "kmag": 8.20},
    {"id": "209", "periods": [1365], "kmag": 8.27},
    {"id": "210", "periods": [750, 1625], "kmag": 7.20},
    {"id": "216", "periods": [580, 2565], "kmag": 7.93},
    {"id": "217", "periods": [430, 2570], "kmag": 8.42},
    {"id": "219", "periods": [3650], "kmag": 7.72},
    {"id": "220", "periods": [770], "kmag": 7.64},
    {"id": "223", "periods": [2100], "kmag": 7.71},
    {"id": "225", "periods": [955], "kmag": 7.32},
    {"id": "227", "periods": [445], "kmag": 8.84},
]

print(f"Загружено {len(data)} объектов с периодами")

# ============================================================
# РАСЧЁТ
# ============================================================
results = []
for obj in data:
    for P in obj['periods']:
        k_raw = P / T0
        k = round(k_raw)
        dev = abs(k_raw - k) / k_raw * 100
        results.append({
            'id': obj['id'],
            'P': P,
            'k': k,
            'dev': dev,
            'kmag': obj['kmag']
        })

total = len(results)
print(f"Найдено периодов: {total}")

# ============================================================
# СТАТИСТИКА
# ============================================================
good_1 = sum(1 for r in results if r['dev'] < 1.0)
good_5 = sum(1 for r in results if r['dev'] < 5.0)

print(f"\nВсего: {total}")
print(f"< 1%:  {good_1} ({good_1/total*100:.1f}%)")
print(f"< 5%:  {good_5} ({good_5/total*100:.1f}%)")

p_val = binom.sf(good_1 - 1, total, 0.01) if good_1 > 0 else 1.0
sigma = np.sqrt(-2 * np.log(p_val)) if p_val > 0 else 0.0

print(f"\np = {p_val:.2e}, σ = {sigma:.1f}")

# ============================================================
# ГРАФИК
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

k_vals = [r['k'] for r in results if not np.isnan(r['kmag'])]
kmag_vals = [r['kmag'] for r in results if not np.isnan(r['kmag'])]

ax.scatter(kmag_vals, k_vals, s=20, alpha=0.6, color='blue')
ax.set_xlabel('K-звёздная величина (чем меньше, тем ярче)', fontsize=12)
ax.set_ylabel('Номер гармоники k = P / 16.35', fontsize=12)
ax.set_title('Закон Фужера: выборка БМО (Chatys+ 2019)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

ax.text(0.05, 0.95, 
        f'Совпадений < 1%: {good_1}/{total} ({good_1/total*100:.1f}%)\nσ = {sigma:.1f}',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('lmc_harmonics.png', dpi=150)
print("\n✅ График сохранён: lmc_harmonics.png")
plt.close()

print("\n" + "="*70)
print("ГОТОВО!")
print("="*70)