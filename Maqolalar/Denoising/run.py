"""
Buyruq satridan ishga tushirish
=================================

Ishlatish:

    # bitta fayl
    python3 run.py rasmlar/River_1.tif

    # bir nechta fayl
    python3 run.py rasmlar/*.tif

    # butun papka (ichki papkalar bilan birga)
    python3 run.py rasmlar/

    # chiqish papkasini belgilash
    python3 run.py rasmlar/ -o natijalar/

    # strategiyani tanlash
    python3 run.py rasmlar/River_1.tif --strategy hybrid

Har bir kirish fayli uchun uchta natija hosil bo'ladi:

    natijalar/River_1_denoised.tif      tozalangan tenzor (uint16, QGIS ochadi)
    natijalar/River_1_comparison.png    original / tozalangan / farq
    natijalar/River_1_report.txt        kanal bo'yicha hisobot

Birinchi ishga tushirishda klassifikator o'rgatiladi va clf.pkl
sifatida saqlanadi; keyingi safar u qayta ishlatiladi.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import tifffile

from full_pipeline import run_pipeline
from save_output import save_cube_tif, save_comparison_figure

REFLECTANCE_SCALE = 10000.0
CLASSIFIER_PATH = "clf.pkl"

# Sentinel-2 kanallari (13 kanalli mahsulot uchun). Kanallar soni boshqacha
# bo'lsa, hisobotda oddiy raqamli nomlar ishlatiladi.
SENTINEL2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8", "B8A", "B9", "B10", "B11", "B12"]


def get_classifier(path: str = CLASSIFIER_PATH):
    """Saqlangan klassifikatorni yuklaydi, bo'lmasa o'rgatib saqlaydi."""
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    print("Klassifikator topilmadi, o'rgatilmoqda (bir necha daqiqa)...")
    from noise_type_classification import NoiseTypeClassifier, build_calibration_set

    features, labels = build_calibration_set(n_scenes=120, size=96, n_bands=6)
    classifier = NoiseTypeClassifier(k=21).fit(features, labels)
    with open(path, "wb") as fh:
        pickle.dump(classifier, fh)
    print(f"Saqlandi: {path}\n")
    return classifier


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


def load_cube(path: str, scale: float) -> np.ndarray:
    """uint16 aks ettirish qiymatlarini [0, 1] oralig'iga keltiradi."""
    cube = tifffile.imread(path).astype(np.float64) / scale
    return np.clip(cube, 0.0, 1.0)


def write_report(path: str, src: str, result: dict, band_names=None):
    n_bands = len(result["band_sigmas"])
    names = band_names or [f"kanal_{i}" for i in range(n_bands)]
    aqi = np.atleast_1d(result["aqi"]) if result["aqi"] is not None else [None] * n_bands

    lines = [
        f"Fayl      : {src}",
        f"Strategiya: {result['strategy']}",
        f"Kanallar  : {n_bands}",
        "",
        f"{'kanal':>8}{'turi':>14}{'sigma*':>12}{'usul':>15}{'AQI':>9}",
    ]
    for i in range(n_bands):
        aqi_txt = f"{aqi[i]:.3f}" if aqi[i] is not None else "-"
        lines.append(f"{names[i]:>8}{result['band_types'][i]:>14}"
                     f"{result['band_sigmas'][i]:>12.6f}"
                     f"{result['band_denoisers'][i]:>15}{aqi_txt:>9}")

    if result["low_quality_bands"]:
        marked = ", ".join(names[i] for i in result["low_quality_bands"])
        lines += ["", f"Sifati past deb belgilangan kanallar: {marked}"]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multispektrli tasvirlarni kanal darajasida shovqinsizlantirish")
    parser.add_argument("inputs", nargs="+",
                        help="tif fayl(lar), shablon yoki papka")
    parser.add_argument("-o", "--output", default="natijalar",
                        help="chiqish papkasi (standart: natijalar)")
    parser.add_argument("--strategy", default="bm3d",
                        choices=["bm3d", "hybrid", "routing"],
                        help="haqiqiy L1C uchun bm3d tavsiya etiladi")
    parser.add_argument("--scale", type=float, default=REFLECTANCE_SCALE,
                        help="uint16 -> aks ettirish bo'luvchisi (standart 10000)")
    parser.add_argument("--no-figure", action="store_true",
                        help="solishtirish rasmini yaratmaslik")
    args = parser.parse_args()

    files = collect_files(args.inputs)
    if not files:
        sys.exit("Tif fayl topilmadi. Yo'lni tekshiring.")

    os.makedirs(args.output, exist_ok=True)

    # Klassifikator faqat tasniflashga tayanadigan strategiyalarda kerak.
    # strategy="bm3d" tasniflashni butunlay chetlab o'tadi, shuning uchun
    # uni o'rgatish (bir necha daqiqa) ortiqcha bo'lar edi.
    classifier = None if args.strategy == "bm3d" else get_classifier()

    print(f"Topildi: {len(files)} ta fayl,  strategiya: {args.strategy}")
    if classifier is None:
        print("Tasniflash ishlatilmaydi (strategy=bm3d), klassifikator kerak emas.")
    print(f"Chiqish papkasi: {args.output}\n")

    for n, src in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(src))[0]
        cube = load_cube(src, args.scale)

        if cube.ndim == 2:
            cube = cube[:, :, None]

        band_names = SENTINEL2_BANDS if cube.shape[2] == 13 else None
        result = run_pipeline(cube, classifier, strategy=args.strategy)
        denoised = result["denoised"]
        if denoised.ndim == 2:
            denoised = denoised[:, :, None]

        save_cube_tif(denoised, os.path.join(args.output, f"{stem}_denoised.tif"))
        write_report(os.path.join(args.output, f"{stem}_report.txt"), src, result,
                     band_names=band_names)

        if not args.no_figure and cube.shape[2] >= 4:
            save_comparison_figure(cube, denoised,
                                   os.path.join(args.output, f"{stem}_comparison.png"),
                                   title=f"{stem}  (strategy={args.strategy})")

        removed = float(np.abs(cube - denoised).mean())
        print(f"[{n}/{len(files)}] {stem:<28} "
              f"kanallar={cube.shape[2]:>2}  "
              f"sigma* ort={result['band_sigmas'].mean():.6f}  "
              f"olib tashlandi={removed:.6f}")

    print(f"\nTayyor. Natijalar: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()