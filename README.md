# PSP Cosmological Model — Analysis Codes

This repository contains the analysis codes for the Phase-State Parameter (PSP) cosmological model.

## Preprints

- **Version 11.3 (Final)**:  
  DOI: [10.5281/zenodo.21652276](https://doi.org/10.5281/zenodo.21652276)  
  - Russian and English versions available.

- **Version 1 (Original)**:  
  DOI: [10.24108/preprints-3115804](https://doi.org/10.24108/preprints-3115804)

## Contents

- MCMC analysis for Pantheon+ (1701 SN Ia)
- BAO analysis for DESI DR2
- Quasar analysis for Lusso+ 2020
- Retroprediction test for 2010–2024 data
- Rhythm detection: T₀ = 16.35 days

## Author

Evgeny Chernoknizhny  
ORCID: [0009-0007-2558-9172](https://orcid.org/0009-0007-2558-9172)

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)
## 🧪 Код и воспроизводимость

Этот репозиторий содержит полный набор скриптов для воспроизведения результатов статьи.

### Основной скрипт (эмпирическое подтверждение)
- `psp_full_free.py` — **главный скрипт статьи**.  
  Загружает данные SDSS DR5, выполняет анализ методом Ломба–Скаргла и строит спектр мощности с пиками 1λξ, 2λξ, 3λξ (уровень значимости 5.9σ).  
  **Для проверки результата рецензенту достаточно запустить именно этот файл.**

### Физическая интерпретация
- `resonance.py` — **резонансный анализ**.  
  Показывает, что периоды наблюдаемых квазаров (OJ 287, 3C 273) и пики B-мод являются гармониками собственных частот оболочки Вселенной.  
  Этот скрипт выводит радиус оболочки \(R_{\text{shell}}\), совпадающий с калибровкой модели PSP.

### Дополнительные проверки
- `attenuation_analysis.py` — проверка закона затухания \(S(r) \propto r^{-0.581}\) на данных Lusso+ 2020.
- `periodicity_analysis.py` — альтернативный анализ периодичности на SDSS.
- `stability_test.py` — тесты устойчивости модели.
- `source_reconstruction.py` — реконструкция источника сигнала.
