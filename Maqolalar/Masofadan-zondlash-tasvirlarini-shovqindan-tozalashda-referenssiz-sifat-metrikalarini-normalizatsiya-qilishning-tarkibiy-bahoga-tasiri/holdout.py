"""
1) Guruhlangan (tasvir darajasidagi) hold-out cross-validation.

   Reviewer eslatmasi: min-max normalizatsiya parametrlari (min, max) BUTUN
   datasetdan hisoblangani sababli, ilgari keltirilgan r_norm = 0,893
   IN-SAMPLE (o'rgatilgan namuna ichidagi) natija hisoblanadi - bu
   normalizatsiyaning YANGI, ko'rilmagan datasetda qanday ishlashi haqida
   hech narsa demaydi.

   Bu yerda 42 ta manba tasvir 5 ta guruhga (fold) bo'linadi. Har bir
   fold uchun: min/max parametrlari FAQAT qolgan 4 ta fold tasvirlaridan
   (train) hisoblanadi, so'ngra shu parametrlar hold-out fold'ning
   namunalariga qo'llaniladi (extrapolation ruxsat etiladi - qiymatlar
   [0;1] dan tashqariga chiqishi mumkin). Barcha 5 ta fold bo'yicha
   hold-out bashoratlar yig'ilib, YAGONA pooled out-of-fold Pirson
   korrelyatsiyasi hisoblanadi - bu haqiqiy out-of-sample baho.

2) Juftlashgan (paired) klaster-bootstrap: Delta|r| = |r_norm| - |r_raw|
   ishonch intervali - normalizatsiyaning yutug'i tasodifiy emasligini
   to'g'ridan-to'g'ri tekshirish uchun.
"""
import numpy as np
from scipy import stats
import json

rng = np.random.default_rng(7)
data = np.load('data.npz')
M1, M2, M3, PSNR = data['M1'], data['M2'], data['M3'], data['PSNR']
image_id = data['image_id']
n_images = int(data['n_images'])

def pearson_r(a, b):
    return stats.pearsonr(a, b).statistic

# ============================================================
# 1) GURUHLANGAN HOLD-OUT CROSS-VALIDATION (5-fold, tasvir darajasida)
# ============================================================
N_FOLDS = 5
img_order = rng.permutation(n_images)
folds = np.array_split(img_order, N_FOLDS)

oof_composite_norm = np.zeros_like(PSNR)  # pooled out-of-fold predictions
fold_r = []

for k in range(N_FOLDS):
    holdout_imgs = set(folds[k].tolist())
    train_mask = np.array([iid not in holdout_imgs for iid in image_id])
    hold_mask = ~train_mask

    # Train fold'dan min/max hisoblanadi
    mins = dict(M1=M1[train_mask].min(), M2=M2[train_mask].min(), M3=M3[train_mask].min())
    maxs = dict(M1=M1[train_mask].max(), M2=M2[train_mask].max(), M3=M3[train_mask].max())

    def norm_with_train_params(v, name):
        return (v - mins[name]) / (maxs[name] - mins[name])

    m1n = norm_with_train_params(M1[hold_mask], 'M1')
    m2n = norm_with_train_params(M2[hold_mask], 'M2')
    m3n = norm_with_train_params(M3[hold_mask], 'M3')
    comp = (m1n + m2n + m3n) / 3.0

    oof_composite_norm[hold_mask] = comp
    fold_r.append(round(float(pearson_r(comp, PSNR[hold_mask])), 4))

r_oof_pooled = float(pearson_r(oof_composite_norm, PSNR))  # yagona pooled out-of-fold r

# Solishtirish uchun: to'liq in-sample normalizatsiya (butun dataset min/max)
def minmax_full(v):
    return (v - v.min()) / (v.max() - v.min())
composite_norm_insample = (minmax_full(M1) + minmax_full(M2) + minmax_full(M3)) / 3.0
r_insample = float(pearson_r(composite_norm_insample, PSNR))

composite_raw = (M1 + M2 + M3) / 3.0
r_raw = float(pearson_r(composite_raw, PSNR))

print("=== Guruhlangan hold-out CV (5-fold) ===")
print("Har bir foldning hold-out r:", fold_r)
print(f"In-sample r (to'liq dataset min/max): {r_insample:.4f}")
print(f"Pooled out-of-fold r (train-fold min/max): {r_oof_pooled:.4f}")
print(f"Xom tarkibiy baho r (normalizatsiyasiz, referens): {r_raw:.4f}")

# ============================================================
# 2) JUFTLASHGAN KLASTER-BOOTSTRAP: Delta|r| = |r_norm| - |r_raw|
# ============================================================
N_BOOT = 2000
image_indices = [np.where(image_id == i)[0] for i in range(n_images)]
boot_diff = np.zeros(N_BOOT)
boot_r_raw = np.zeros(N_BOOT)
boot_r_norm = np.zeros(N_BOOT)

for b in range(N_BOOT):
    sampled_images = rng.integers(0, n_images, size=n_images)
    idx = np.concatenate([image_indices[i] for i in sampled_images])
    m1, m2, m3, p = M1[idx], M2[idx], M3[idx], PSNR[idx]

    comp_raw = (m1 + m2 + m3) / 3.0
    comp_norm = (minmax_full(m1) + minmax_full(m2) + minmax_full(m3)) / 3.0
    r_r = pearson_r(comp_raw, p)
    r_n = pearson_r(comp_norm, p)
    boot_r_raw[b] = r_r
    boot_r_norm[b] = r_n
    boot_diff[b] = abs(r_n) - abs(r_r)

diff_ci = (float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5)))
diff_mean = float(np.mean(boot_diff))
p_one_sided = float(np.mean(boot_diff <= 0))  # H0: normalizatsiya yutug'i yo'q yoki manfiy

print("\n=== Juftlashgan bootstrap: Delta|r| = |r_norm| - |r_raw| ===")
print(f"O'rtacha Delta|r| = {diff_mean:.4f}, 95% CI = [{diff_ci[0]:.4f}; {diff_ci[1]:.4f}]")
print(f"Bir tomonlama bootstrap p-qiymat (Delta|r| <= 0): {p_one_sided:.4f}")

result = dict(
    holdout_cv=dict(n_folds=N_FOLDS, fold_r=fold_r, r_insample=round(r_insample, 4),
                     r_oof_pooled=round(r_oof_pooled, 4), r_raw_reference=round(r_raw, 4)),
    paired_bootstrap=dict(n_bootstrap=N_BOOT, delta_abs_r_mean=round(diff_mean, 4),
                           delta_abs_r_ci95=[round(diff_ci[0], 4), round(diff_ci[1], 4)],
                           p_one_sided=round(p_one_sided, 4)),
)
with open('holdout_and_diff.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\nSaqlandi: holdout_and_diff.json")