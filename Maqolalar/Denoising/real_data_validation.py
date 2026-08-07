"""
Haqiqiy Sentinel-2 (EuroSAT allBands) ma'lumotlarida tasdiqlash
=================================================================

Haqiqiy tasvirda shovqinning haqiqiy turi va darajasi noma'lum, shuning
uchun baholovchining xatoligini bevosita o'lchash mumkin emas. Buning
o'rniga uchta tekshiriladigan bashorat sinovdan o'tkaziladi.

Bashorat 1 (fazoviy ruxsat). EuroSAT mahsulotida 13 kanal 10 m tarmoqqa
keltirilgan, ammo ularning nativ ruxsati turlicha: B2, B3, B4, B8 nativ
10 m; B5, B6, B7, B8A, B11, B12 nativ 20 m; B1, B9, B10 nativ 60 m.
Qayta namunalash qo'shni piksellarni aralashtirib shovqinni silliqlaydi,
shuning uchun nativ ruxsati past kanallarda baholangan shovqin darajasi
ham past chiqishi kerak.

Bashorat 2 (izchillik). Bir xil yer qoplami sinfiga tegishli turli
patchlarda baholar bir-biriga yaqin bo'lishi kerak, chunki ular bir xil
sensor va shunga o'xshash yoritilganlik sharoitida olingan.

Bashorat 3 (fizik rejim). Yorqinligi yuqori kanallarda foton oqimi katta
bo'lgani uchun Puasson tashkil etuvchisining ulushi ham yuqori bo'lishi
kerak, ya'ni log(P/G) ko'rsatkichi yorqinlik bilan musbat bog'liqlikda
bo'lishi lozim.

MUHIM CHEKLOV: EuroSAT L1C mahsuloti radiometrik kalibrlash va geometrik
qayta namunalashdan o'tgan, ya'ni xom sensor qiymatlari emas. Qayta
namunalash shovqinni fazoviy korrelyatsiyalangan holga keltiradi,
holbuki MAD baholovchisi ham, NLF fiti ham piksellararo mustaqil
shovqinga tayanadi. Shu sababli olinadigan a va b qiymatlari sensor
parametrlari emas, mahsulotdagi qoldiq shovqin xarakteristikalari
sifatida talqin qilinishi kerak.
"""

import argparse
import glob
import os

import numpy as np
import tifffile
from scipy import stats

from noise_characterization import characterize_noise_per_band
from noise_type_classification import expand_features

# EuroSAT allBands kanallari va ularning nativ fazoviy ruxsati (metr)
BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
              "B8", "B8A", "B9", "B10", "B11", "B12"]
NATIVE_GSD = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]

REFLECTANCE_SCALE = 10000.0   # L1C mahsulotida aks ettirish koeffitsiyenti


def load_cube(path: str) -> np.ndarray:
    """uint16 L1C qiymatlarini [0, 1] oralig'idagi aks ettirishga keltiradi."""
    cube = tifffile.imread(path).astype(np.float64) / REFLECTANCE_SCALE
    return np.clip(cube, 0.0, 1.0)


def analyze_file(path: str) -> dict:
    cube = load_cube(path)
    char = characterize_noise_per_band(cube)
    mean_levels = cube.reshape(-1, cube.shape[2]).mean(axis=0)
    feats = expand_features(char["features"], mean_levels)

    return {
        "path": path,
        "class": os.path.basename(os.path.dirname(path)),
        "a": feats[:, 0],
        "b": feats[:, 1],
        "impulse": feats[:, 2],
        "kurtosis": feats[:, 3],
        "log_ratio": feats[:, 4],
        "mean_level": feats[:, 5],
        "sigma": np.sqrt(np.clip(feats[:, 0] * mean_levels + feats[:, 1], 0, None)),
    }


def collect_files(inputs) -> list:
    """Fayl, shablon yoki papka ro'yxatidan .tif fayllarni yig'adi."""
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += glob.glob(os.path.join(item, "**", "*.tif"), recursive=True)
            files += glob.glob(os.path.join(item, "**", "*.tiff"), recursive=True)
        else:
            files += glob.glob(item)
    return sorted({f for f in files if f.lower().endswith((".tif", ".tiff"))})


def main(inputs=("dataset",)):
    files = collect_files(list(inputs))
    if not files:
        raise SystemExit(
            f"Tif fayl topilmadi: {list(inputs)}\n"
            "Papka yo'lini argument sifatida bering, masalan:\n"
            "    python3 real_data_validation.py dataset/")

    results = [analyze_file(f) for f in files]
    classes = sorted({r["class"] for r in results})
    print(f"Tahlil qilingan patchlar: {len(results)},  sinflar: {len(classes)}")
    print(f"Sinflar: {', '.join(classes)}\n")

    sigma = np.array([r["sigma"] for r in results])          # (N, 13)
    mean_lv = np.array([r["mean_level"] for r in results])
    log_r = np.array([r["log_ratio"] for r in results])
    impulse = np.array([r["impulse"] for r in results])

    # ---- Bashorat 1: fazoviy ruxsat ----
    print("=" * 74)
    print("BASHORAT 1: nativ fazoviy ruxsat va baholangan shovqin darajasi")
    print("=" * 74)
    print(f"{'kanal':>7}{'nativ GSD':>11}{'yorqinlik':>12}"
          f"{'sigma* (ort)':>14}{'sigma* (std)':>14}{'log(P/G)':>11}")
    for i, name in enumerate(BAND_NAMES):
        print(f"{name:>7}{NATIVE_GSD[i]:>9} m{mean_lv[:, i].mean():>12.4f}"
              f"{sigma[:, i].mean():>14.5f}{sigma[:, i].std():>14.5f}"
              f"{log_r[:, i].mean():>11.2f}")

    gsd = np.array(NATIVE_GSD, dtype=float)
    band_sigma = sigma.mean(axis=0)
    rho, pval = stats.spearmanr(gsd, band_sigma)
    print(f"\nSpirmen korrelyatsiyasi (GSD va sigma*): rho = {rho:+.3f}, p = {pval:.4f}")

    for g in [10, 20, 60]:
        sel = gsd == g
        print(f"  nativ {g:>2} m kanallar (n={int(sel.sum())}): "
              f"sigma* o'rtacha = {band_sigma[sel].mean():.5f}")

    # ---- Bashorat 2: sinf ichidagi izchillik ----
    print("\n" + "=" * 74)
    print("BASHORAT 2: bir xil sinf ichidagi baholarning izchilligi")
    print("=" * 74)
    within, between = [], []
    for i in range(len(BAND_NAMES)):
        col = sigma[:, i]
        overall = col.std()
        cls_stds = []
        for cname in classes:
            vals = np.array([r["sigma"][i] for r in results if r["class"] == cname])
            if vals.size >= 2:
                cls_stds.append(vals.std())
        if cls_stds and overall > 0:
            within.append(float(np.mean(cls_stds)))
            between.append(float(overall))

    within, between = np.array(within), np.array(between)
    print(f"Sinf ichidagi o'rtacha standart og'ish : {within.mean():.5f}")
    print(f"Umumiy standart og'ish                 : {between.mean():.5f}")
    print(f"Nisbat (ichki / umumiy)                : {(within / between).mean():.3f}")
    print("Nisbat birdan sezilarli kichik bo'lsa, baholar sinf ichida izchil.")

    # ---- Bashorat 3: yorqinlik va Puasson ulushi ----
    print("\n" + "=" * 74)
    print("BASHORAT 3: yorqinlik va Puasson tashkil etuvchisining ulushi")
    print("=" * 74)
    flat_mean = mean_lv.ravel()
    flat_lr = log_r.ravel()
    valid = np.isfinite(flat_mean) & np.isfinite(flat_lr)
    rho3, p3 = stats.spearmanr(flat_mean[valid], flat_lr[valid])
    print(f"Kanal-patch juftliklari soni: {int(valid.sum())}")
    print(f"Spirmen korrelyatsiyasi (yorqinlik va log(P/G)): rho = {rho3:+.3f}, p = {p3:.2e}")

    # ---- Qo'shimcha: impulsivlik ----
    print("\n" + "=" * 74)
    print("QO'SHIMCHA: impulsivlik ulushi")
    print("=" * 74)
    print(f"Barcha kanal-patch juftliklari bo'yicha o'rtacha impuls ulushi: "
          f"{impulse.mean():.4f}")
    print(f"Eng yuqori impuls ulushiga ega kanal: "
          f"{BAND_NAMES[int(np.argmax(impulse.mean(axis=0)))]} "
          f"({impulse.mean(axis=0).max():.4f})")
    print("Kutilgani: L1C mahsulotida impulsiv buzilish deyarli uchramaydi,")
    print("chunki u qayta ishlash bosqichida bartaraf etilgan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Haqiqiy Sentinel-2 ma'lumotlarida tasdiqlash")
    parser.add_argument("inputs", nargs="*", default=["dataset"],
                        help="tif fayl(lar), shablon yoki papka (standart: dataset)")
    args = parser.parse_args()
    main(args.inputs)