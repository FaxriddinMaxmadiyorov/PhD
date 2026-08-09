"""
Cluster-bootstrap ishonch intervali (CI).

Muhim metodologik eslatma: 1386 ta namuna to'liq mustaqil EMAS - ular 42 ta
manba tasvirdan olingan (har biridan 11 sigma x 3 realizatsiya). Shuning
uchun oddiy (namuna darajasidagi) bootstrap noto'g'ri bo'lardi - u
kuzatuvlar orasidagi bog'liqlikni e'tiborsiz qoldiradi va CI ni sun'iy
ravishda torайтiradi.

Bu yerda CLUSTER bootstrap qo'llaniladi: har bir bootstrap iteratsiyasida
42 ta manba tasvir ID'si o'rnini bosuvchi (with replacement) tanlanadi,
tanlangan tasvirlarga tegishli BARCHA namunalar (33 tadan) birgalikda
olinadi. Bu klasterlar ichidagi bog'liqlikni to'g'ri hisobga oladi.
"""
import numpy as np
from scipy import stats
import json

rng = np.random.default_rng(123)
data = np.load('data.npz')
M1, M2, M3, PSNR = data['M1'], data['M2'], data['M3'], data['PSNR']
image_id = data['image_id']
n_images = int(data['n_images'])

N_BOOT = 2000

def minmax(v):
    return (v - v.min()) / (v.max() - v.min())

def pearson_r(a, b):
    return stats.pearsonr(a, b).statistic

boot_r_raw = np.zeros(N_BOOT)
boot_r_norm = np.zeros(N_BOOT)

image_indices = [np.where(image_id == i)[0] for i in range(n_images)]

for b in range(N_BOOT):
    sampled_images = rng.integers(0, n_images, size=n_images)  # with replacement
    idx = np.concatenate([image_indices[i] for i in sampled_images])

    m1, m2, m3, p = M1[idx], M2[idx], M3[idx], PSNR[idx]
    comp_raw = (m1 + m2 + m3) / 3.0
    comp_norm = (minmax(m1) + minmax(m2) + minmax(m3)) / 3.0

    boot_r_raw[b] = pearson_r(comp_raw, p)
    boot_r_norm[b] = pearson_r(comp_norm, p)

def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

ci_raw = ci95(boot_r_raw)
ci_norm = ci95(boot_r_norm)

# Bootstrap-based test: farq ishonchli ijobiymi? (norm - raw > 0 bo'lgan ulush)
diff = boot_r_norm - boot_r_raw  # ikkalasi ham manfiy r, |r| solishtirish uchun ishora almashtiramiz
prob_norm_better = float(np.mean(np.abs(boot_r_norm) > np.abs(boot_r_raw)))

result = dict(
    n_bootstrap=N_BOOT,
    r_raw_point=round(float(pearson_r((M1+M2+M3)/3.0, PSNR)), 4),
    r_raw_ci95=[round(ci_raw[0], 4), round(ci_raw[1], 4)],
    r_norm_point=round(float(pearson_r((minmax(M1)+minmax(M2)+minmax(M3))/3.0, PSNR)), 4),
    r_norm_ci95=[round(ci_norm[0], 4), round(ci_norm[1], 4)],
    prob_normalized_composite_stronger=round(prob_norm_better, 4),
)
print(json.dumps(result, indent=2, ensure_ascii=False))

with open('bootstrap_ci.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)