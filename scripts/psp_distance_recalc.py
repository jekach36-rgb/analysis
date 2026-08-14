import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("ПЕРЕСЧЁТ РАССТОЯНИЙ В МОДЕЛИ PSP")
print("=" * 70)

# ------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------
# Загрузи свой реальный файл с данными (например, SDSS или Pantheon+)
# Пример: df = pd.read_csv('your_data.csv')

# Пока используем синтетические данные для демонстрации
np.random.seed(42)
n = 1000
z_data = np.random.uniform(0.01, 2.0, n)
Lx_data = 10 ** np.random.uniform(42, 46, n)
Luv_data = 10 ** np.random.uniform(42, 46, n)

# ------------------------------------------------------------
# 2. РАСЧЁТ ξ И M
# ------------------------------------------------------------
xi = Lx_data / Luv_data
M = 0.29 + 0.01 * (xi - 1)

# ------------------------------------------------------------
# 3. ВЫВОД СТАТИСТИКИ
# ------------------------------------------------------------
print(f"\nСреднее M: {np.mean(M):.4f}")
print(f"Стандартное отклонение M: {np.std(M):.4f}")
print(f"Доля объектов с M ≈ 0.29: {np.mean(np.abs(M - 0.29) < 0.01) * 100:.1f}%")

# ------------------------------------------------------------
# 4. ГРАФИК: Z vs M
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(z_data, M, s=5, alpha=0.5, color='blue')
ax.axhline(0.29, color='red', linestyle='--', linewidth=2, label='M = 0.29 (наша фаза)')
ax.set_xlabel('Красное смещение z (ΛCDM)', fontsize=14)
ax.set_ylabel('Фаза M (PSP)', fontsize=14)
ax.set_title('Пересчёт расстояний: ΛCDM vs PSP', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.savefig('psp_distance_recalc.png', dpi=300)
plt.show()

print("\n✅ График сохранён как 'psp_distance_recalc.png'")