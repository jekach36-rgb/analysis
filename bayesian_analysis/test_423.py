import numpy as np
import json
from scipy.optimize import minimize
import time

print("="*70)
print("PSP vs LCDM: ALL 423 REAL GALAXIES")
print("="*70)

with open('rotation_curve_corpus_v7.json', 'r') as f:
    data = json.load(f)

galaxies = data['galaxies']
valid = [g for g in galaxies if g.get('data') or g.get('rotation_curve')]
print(f"Загружено: {len(valid)} галактик\n")

G = 4.302e-3
Rshell = 8.36e26 / 3.086e21
Rtor = 0.1 * Rshell

def psp_model(r, m, e0):
    return np.sqrt(G*m/r + e0 * (r/Rshell)**(-0.581) * (r/Rtor))

def lcdm_model(r, m, md, rs):
    return np.sqrt(G*m/r + G*md*r/(r+rs)**2)

def fit_galaxy(g, model_type):
    points = g.get('data') or g.get('rotation_curve', [])
    if not points:
        return None
    
    if 'Rad' in points[0]:
        r = np.array([p['Rad'] for p in points])
        v = np.array([p.get('Vobs', p.get('Vrot', 0)) for p in points])
        err = np.array([p.get('errV', p.get('e_Vrot', 5)) for p in points])
        m = np.array([1e10] * len(r))
    else:
        r = np.array([p['rad_kpc'] for p in points])
        v = np.array([p['vrot_kms'] for p in points])
        err = np.array([5] * len(r))
        m = np.array([1e10] * len(r))
    
    err = np.maximum(err, 0.1*np.mean(v))
    n = len(r)
    
    if model_type == 'psp':
        obj = lambda p: np.sum(((v - psp_model(r, m, p[0]))/err)**2)
        bounds = [(1e-15, 1e-5)]
        x0 = [1e-10]
        n_params = 1
    else:
        obj = lambda p: np.sum(((v - lcdm_model(r, m, p[0], p[1]))/err)**2)
        bounds = [(1e8, 1e15), (1e-2, 1e3)]
        x0 = [1e12, 10]
        n_params = 2
    
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B')
    return res.fun + n_params * np.log(n)

# ВСЕ 423 ГАЛАКТИКИ
test_galaxies = valid
psp_bics = []
lcdm_bics = []
names = []
errors = []

start = time.time()

for i, g in enumerate(test_galaxies):
    name = g.get('galaxy', f'galaxy_{i}')
    print(f"{i+1}/{len(test_galaxies)} {name}")
    try:
        p = fit_galaxy(g, 'psp')
        l = fit_galaxy(g, 'lcdm')
        if p is not None and l is not None:
            psp_bics.append(p)
            lcdm_bics.append(l)
            names.append(name)
            print(f"  PSP: {p:.2f}, LCDM: {l:.2f}, DIFF: {p-l:.2f}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  ERROR: {e}")

elapsed = time.time() - start

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ")
print("="*70)

print(f"\nОбработано: {len(psp_bics)} галактик")
print(f"Ошибок: {len(errors)}")
print(f"Время: {elapsed:.1f} сек")

if psp_bics:
    psp_mean = np.mean(psp_bics)
    lcdm_mean = np.mean(lcdm_bics)
    diff = psp_mean - lcdm_mean
    
    print(f"\nСредний BIC (PSP):   {psp_mean:.2f}")
    print(f"Средний BIC (LCDM):  {lcdm_mean:.2f}")
    print(f"РАЗНИЦА (PSP - LCDM): {diff:.2f}")
    
    if diff < 0:
        print("\n" + "="*70)
        print(">>> РЕЗУЛЬТАТ: PSP ЛУЧШЕ")
        try:
            K = np.exp(-diff/2)
            if np.isinf(K):
                print(">>> Байесовский фактор: K = inf (РЕШАЮЩЕЕ ДОКАЗАТЕЛЬСТВО)")
            else:
                print(f">>> Байесовский фактор: K = {K:.2e}")
        except:
            print(">>> Байесовский фактор: K > 10^100 (РЕШАЮЩЕЕ ДОКАЗАТЕЛЬСТВО)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(">>> РЕЗУЛЬТАТ: LCDM ЛУЧШЕ")
        print("="*70)
else:
    print("Нет результатов")

# Сохраняем результаты
if psp_bics:
    import csv
    with open('results_423.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['galaxy', 'psp_bic', 'lcdm_bic', 'diff'])
        for n, p, l in zip(names, psp_bics, lcdm_bics):
            writer.writerow([n, p, l, p-l])
    print(f"\nРезультаты сохранены в results_423.csv")