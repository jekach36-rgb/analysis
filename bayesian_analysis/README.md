## 🔬 Байесовский тест: PSP vs ΛCDM на 423 галактиках

Проведено байесовское сравнение моделей PSP и ΛCDM на основе объединенного каталога кривых вращения галактик (SPARC + THINGS + LITTLE THINGS + WALLABY DR2).

**Результаты:**

| Модель | Параметров | Средний BIC |
|--------|------------|-------------|
| **PSP** | 1 | **51,476,751.19** |
| ΛCDM (NFW) | 2 | 51,485,591.32 |

- **Разница BIC (PSP - ΛCDM): -8,840.13**
- **Байесовский фактор: K = ∞ (решающее доказательство)**

**Вывод:** PSP статистически превосходит ΛCDM с решающим Байесовским фактором, объясняя кривые вращения галактик **БЕЗ** темной материи и с **МЕНЬШИМ** числом параметров.

📂 **[Код и данные в папке bayesian_analysis/](https://github.com/jekach36-rgb/analysis/tree/main/bayesian_analysis)**
## Визуализация результатов

### 1. Сравнение BIC (PSP vs ΛCDM)
![BIC Comparison](figures/bic_comparison.png)

### 2. Распределение разницы BIC
![Delta BIC Distribution](figures/delta_bic_distribution.png)

### 3. Топ-10 галактик, где PSP лучше
![Top 10 Galaxies](figures/top10_galaxies.png)

### 4. Сравнение для всех галактик
![All Galaxies](figures/bic_all_galaxies.png)
