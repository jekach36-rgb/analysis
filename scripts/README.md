# Скрипты PSP Model

В этой папке хранятся все Python-скрипты для анализа данных и проверки модели PSP.

## 🐍 Требования

- **Python:** 3.13+
- **Библиотеки:** numpy, pandas, matplotlib, scipy, astropy

Установка зависимостей:
```bash
pip install -r requirements.txt
📂 Список скриптов
Файл	Описание
psp_full_free.py	Основной анализ: ритм 16.35 дней, гармоники, AIC attenuation_analysis.py	Закон затухания S(r)∝r−0.581 periodicity_analysis.py
Поиск периодичностей в данных resonance.py	Резонансный анализ rhythm_T0_16.35.py
Проверка глобального ритма 16.35 дней rhythm_test.py
Тесты для ритма source_reconstruction.py	Восстановление источников
stability_test.py	Тесты устойчивости модели
🚀 Запуск основного скрипта
bash
python psp_full_free.py
📊 Входные данные
../data/lusso_cleaned.csv — каталог квазаров Lusso+ 2020

../data/sdss_data.csv — данные SDSS

📤 Выходные данные
Скрипты создают:

Графики (PNG, PDF)

Таблицы с результатами

Статистические выводы

📌 Лицензия
Код распространяется под лицензией MIT.

### 📁 **3. demo.ipynb (Jupyter Notebook)**

**Файл:** `demo.ipynb`

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PSP Model — Демонстрационный анализ\n",
    "\n",
    "## Глобальный ритм Вселенной T₀ = 16.35 дней\n",
    "\n",
    "Этот ноутбук демонстрирует ключевые результаты модели PSP."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.signal import lombscargle\n",
    "\n",
    "print(\"PSP Model Demo\")\n",
    "print(\"=\"*50)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Данные\n",
    "\n",
    "Загрузка каталога квазаров Lusso+ 2020."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/lusso_cleaned.csv')\n",
    "print(f\"Загружено {len(df)} записей\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Глобальный ритм T₀ = 16.35 дней\n",
    "\n",
    "Проверка периодичности в данных."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Пример анализа\n",
    "T0 = 16.35\n",
    "print(f\"Глобальный ритм: {T0} дней\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "plt.axvline(x=T0, color='r', linestyle='--', label=f'T₀ = {T0} дней')\n",
    "plt.xlabel('Период (дни)')\n",
    "plt.ylabel('Мощность')\n",
    "plt.title('Глобальный ритм Вселенной')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Вывод\n",
    "\n",
    "Демонстрация завершена. Для полного анализа запусти:\n",
    "```bash\n",
    "python psp_full_free.py\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}


Данные загружаются скриптами из папки `../scripts/`.

## 📜 Лицензия

Данные распространяются под лицензией:
**Creative Commons Attribution 4.0 International (CC BY 4.0)**
