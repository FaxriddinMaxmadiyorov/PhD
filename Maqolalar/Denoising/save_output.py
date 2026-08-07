"""
Tozalangan tenzorni faylga saqlash va vizual solishtirish
=======================================================

run_pipeline() natijani massiv sifatida qaytaradi. Ushbu modul uni
diskka yozish va ko'rish uchun yordamchi funksiyalarni beradi.
"""

import numpy as np
import tifffile

REFLECTANCE_SCALE = 10000.0


def save_cube_tif(cube: np.ndarray, path: str, as_uint16: bool = True):
    """
    Tozalangan tenzorni GeoTIFF sifatida saqlaydi.

    as_uint16=True bo'lsa, kirish mahsuloti bilan bir xil formatda
    ([0,1] -> uint16, 10000 ga ko'paytirilgan holda) yoziladi. Bu
    natijani standart dasturlarda (QGIS, SNAP) ochish uchun qulay.
    """
    if as_uint16:
        data = np.clip(cube * REFLECTANCE_SCALE, 0, 65535).astype(np.uint16)
    else:
        data = cube.astype(np.float32)
    tifffile.imwrite(path, data)
    return path


def rgb_composite(cube: np.ndarray, bands=(3, 2, 1),
                   lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """
    Ko'rish uchun RGB kompozit. Sentinel-2 da B4, B3, B2 (indekslar
    3, 2, 1) tabiiy ranglarni beradi. Kontrast protsentil bo'yicha
    cho'ziladi, chunki xom aks ettirish qiymatlari juda qorong'i ko'rinadi.
    """
    im = cube[:, :, list(bands)]
    lo, hi = np.percentile(im, lo_pct), np.percentile(im, hi_pct)
    return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)


def save_comparison_figure(original: np.ndarray, denoised: np.ndarray,
                            path: str, title: str = ""):
    """Original, tozalangan va olib tashlangan shovqin xaritasini yonma-yon chizadi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = np.abs(original - denoised).mean(axis=2)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(rgb_composite(original)); ax[0].set_title("Original", fontsize=10)
    ax[1].imshow(rgb_composite(denoised)); ax[1].set_title("Tozalangan", fontsize=10)
    ax[2].imshow(diff, cmap="magma"); ax[2].set_title("Olib tashlangan shovqin", fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import pickle
    from full_pipeline import run_pipeline

    cube = np.clip(tifffile.imread("euro/maq/River/River_1.tif").astype(np.float64)
                   / REFLECTANCE_SCALE, 0, 1)
    classifier = pickle.load(open("clf.pkl", "rb"))

    result = run_pipeline(cube, classifier, strategy="bm3d")

    save_cube_tif(result["denoised"], "River_1_denoised.tif")
    save_comparison_figure(cube, result["denoised"], "River_1_comparison.png",
                            title="River_1, strategy=bm3d")

    print("Saqlandi: River_1_denoised.tif, River_1_comparison.png")
    print(f"Kirish  : {cube.shape}, {cube.dtype}")
    print(f"Chiqish : {result['denoised'].shape}, {result['denoised'].dtype}")