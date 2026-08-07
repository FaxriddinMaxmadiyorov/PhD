"""
Yarim sun'iy baholash tajribasi
=================================

Muammo: haqiqiy sun'iy yo'ldosh tasvirining shovqinsiz nusxasi mavjud
emas, shuning uchun u yerda PSNR hisoblab bo'lmaydi. Sof sintetik
sahnalar esa haqiqiy yer usti teksturasini to'liq aks ettirmaydi.

Yechim: haqiqiy EuroSAT patchini olib, unga MA'LUM buzilish qo'shamiz.
Asl patch etalon vazifasini bajaradi. U mutlaq shovqinsiz emas (unda
qoldiq shovqin bor), ammo bu taqqoslashga xalal bermaydi: solishtirilayotgan
barcha strategiyalar bir xil boshlang'ich sharoitda bo'ladi. Adabiyotda
bunday etalon pseudo ground truth deb ataladi.

Uchta holat sinaladi:

  1. IMPULS qo'shilgan
     hybrid strategiyaning asosiy da'vosini tekshiradi.
     Kutilgan natija: hybrid ustun.

  2. GAUSS qo'shilgan
     Tur klassifikatsiyasi Gauss-Puasson spektri ichida foyda beradimi.
     Kutilgan natija: ikkalasi teng yoki bm3d bir oz ustun.

  3. HECH NARSA qo'shilmagan  <-- eng muhim holat
     Asl patch o'zgarishsiz beriladi. Ideal algoritm uni deyarli
     o'zgartirmasligi kerak. Agar impuls detektori xato ishlab,
     kanallarni impulsiv deb belgilasa, hybrid keraksiz median tuzatish
     qo'llaydi va asl patchdan uzoqlashadi. Bu detektor xatosining
     to'g'ridan-to'g'ri sonli o'lchovi.

Ishlatish:

    python3 semi_eval.py dataset/
    python3 semi_eval.py dataset/ --n-files 20 --impulse 0.02
"""

import argparse
import glob
import os

import numpy as np
import tifffile
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from full_pipeline import run_pipeline

REFLECTANCE_SCALE = 10000.0
ALL_STRATEGIES = ["bm3d", "hybrid", "routing"]


def collect_files(inputs) -> list:
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += glob.glob(os.path.join(item, "**", "*.tif"), recursive=True)
            files += glob.glob(os.path.join(item, "**", "*.tiff"), recursive=True)
        else:
            files += glob.glob(item)
    return sorted({f for f in files if f.lower().endswith((".tif", ".tiff"))})


def load_cube(path: str) -> np.ndarray:
    cube = tifffile.imread(path).astype(np.float64) / REFLECTANCE_SCALE
    return np.clip(cube, 0.0, 1.0)


def add_impulse(cube: np.ndarray, prob: float, rng, band_fraction: float = 0.4):
    """
    Tasodifiy tanlangan kanallarga tuz-qalampir buzilishini qo'shadi.

    band_fraction - kanallarning qaysi ulushi buziladi. Barcha kanalni
    buzish real emas: amalda buzilish alohida o'qish zanjiriga tegadi.
    """
    out = cube.copy()
    n_bands = cube.shape[2]
    n_corrupt = max(1, int(round(band_fraction * n_bands)))
    corrupted = rng.choice(n_bands, size=n_corrupt, replace=False)

    for i in corrupted:
        mask = rng.random(cube.shape[:2]) < prob
        out[:, :, i][mask] = rng.choice([0.0, 1.0], size=int(mask.sum()))
    return out, sorted(corrupted.tolist())


def add_gaussian(cube: np.ndarray, sigma: float, rng):
    """Barcha kanalga bir xil kuchdagi additiv Gauss shovqini."""
    return np.clip(cube + rng.normal(0, sigma, cube.shape), 0, 1)


def ssim_multiband(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([ssim(a[:, :, i], b[:, :, i], data_range=1.0)
                          for i in range(a.shape[2])]))


def evaluate(files, classifier, impulse_prob: float, gauss_sigma: float,
              seed: int = 0, isolated: bool = False, strategies=None):
    strategies = strategies or ALL_STRATEGIES
    rng = np.random.default_rng(seed)

    conditions = {
        "impuls qo'shilgan": lambda c: add_impulse(c, impulse_prob, rng)[0],
        "Gauss qo'shilgan": lambda c: add_gaussian(c, gauss_sigma, rng),
        "hech narsa qo'shilmagan": lambda c: c.copy(),
    }

    results = {cond: {s: {"psnr": [], "ssim": []} for s in strategies}
               for cond in conditions}
    baseline = {cond: [] for cond in conditions}
    names = []

    for n, path in enumerate(files, 1):
        original = load_cube(path)
        if original.ndim != 3:
            continue
        names.append(os.path.splitext(os.path.basename(path))[0])

        for cond, corrupt in conditions.items():
            damaged = corrupt(original)
            # Buzilish qo'shilmagan holatda tasvirlar aynan teng bo'ladi va
            # PSNR cheksizlikka aylanadi. Bo'lishni bajarmasdan oldin
            # tekshiramiz, aks holda NumPy ogohlantirish beradi.
            same = not np.any(original != damaged)
            baseline[cond].append(np.inf if same
                                  else psnr(original, damaged, data_range=1.0))

            for strat in strategies:
                res = run_pipeline(damaged, classifier, strategy=strat,
                                    use_isolated_impulse=isolated)
                out = res["denoised"]
                results[cond][strat]["psnr"].append(
                    psnr(original, out, data_range=1.0))
                results[cond][strat]["ssim"].append(ssim_multiband(original, out))

        print(f"  [{n}/{len(files)}] {os.path.basename(path)}")

    return results, baseline, names


def print_table(results, baseline):
    for cond in results:
        print()
        print("=" * 76)
        print(f"HOLAT: {cond}")
        print("=" * 76)
        base = np.mean(baseline[cond])
        if np.isfinite(base):
            print(f"Kirish tasviri (asl patchga nisbatan): {base:.2f} dB")
        else:
            print("Kirish tasviri asl patchga aynan teng (PSNR cheksiz)")
        print()
        print(f"{'strategiya':<12}{'PSNR, dB':>12}{'std':>9}{'SSIM':>10}"
              f"{'bm3d ga nisbatan':>20}")

        ref = np.mean(results[cond]["bm3d"]["psnr"])
        for strat in results[cond]:
            p = np.array(results[cond][strat]["psnr"])
            s = np.mean(results[cond][strat]["ssim"])
            delta = "" if strat == "bm3d" else f"{p.mean() - ref:+.2f} dB"
            print(f"{strat:<12}{p.mean():>12.2f}{p.std():>9.2f}{s:>10.4f}{delta:>20}")


def plot_results(results, names, out_path: str, baseline_strategy: str = "bm3d"):
    """
    Har bir holat uchun ikkita panel chizadi.

    Chap panel  - strategiyalar bo'yicha PSNR taqsimoti (box plot ustiga
                  har bir patchning o'z nuqtasi).
    O'ng panel  - har bir patch uchun bazaviy strategiyaga nisbatan farq.
                  Nol chizig'idan yuqorida - ustunlik, pastda - zarar.

    Ikkinchi panel muhimroq: o'rtacha qiymat bir nechta patchning katta
    yutug'i hisobiga shakllanishi mumkin, individual nuqtalar esa buni
    darrov ko'rsatadi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    conditions = list(results)
    fig, axes = plt.subplots(len(conditions), 2, figsize=(12, 3.4 * len(conditions)))
    if len(conditions) == 1:
        axes = axes[None, :]

    for r, cond in enumerate(conditions):
        strategies = list(results[cond])
        data = [np.array(results[cond][s]["psnr"]) for s in strategies]

        ax = axes[r, 0]
        try:
            ax.boxplot(data, tick_labels=strategies, widths=0.5,
                       medianprops=dict(color="black"))
        except TypeError:   # matplotlib < 3.9
            ax.boxplot(data, labels=strategies, widths=0.5,
                       medianprops=dict(color="black"))
        for i, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(i).normal(0, 0.045, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=0.6, zorder=3)
        ax.set_ylabel("PSNR, dB")
        ax.set_title(cond, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[r, 1]
        base = np.array(results[cond][baseline_strategy]["psnr"])
        others = [s for s in strategies if s != baseline_strategy]
        x = np.arange(len(base))
        for s in others:
            diff = np.array(results[cond][s]["psnr"]) - base
            ax.scatter(x, diff, s=34, alpha=0.8, label=s)
            # nolga teng emasligini tekshiruvchi test
            if len(diff) >= 5:
                try:
                    _, pval = sps.wilcoxon(diff)
                    ax.plot([], [], " ",
                            label=f"  o'rtacha {diff.mean():+.2f} dB, p={pval:.3f}")
                except ValueError:
                    pass
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel(f"{baseline_strategy} ga nisbatan farq, dB")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Yarim sun'iy baholash tajribasi")
    parser.add_argument("inputs", nargs="*", default=["dataset"],
                        help="tif fayl(lar), shablon yoki papka")
    parser.add_argument("--n-files", type=int, default=12,
                        help="nechta fayl ishlatilsin (standart 12)")
    parser.add_argument("--impulse", type=float, default=0.02,
                        help="impuls ulushi, standart 0.02 (2%%)")
    parser.add_argument("--gauss", type=float, default=0.01,
                        help="Gauss shovqinining sigma qiymati, standart 0.01")
    parser.add_argument("--isolated-impulse", action="store_true",
                        help="impulsivlikni yakkalik sharti bilan aniqlash")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES, help="qaysi strategiyalar solishtirilsin")
    parser.add_argument("--no-plot", action="store_true", help="grafik chizmaslik")
    parser.add_argument("--plot-path", default="semi_eval_natija.png",
                        help="grafik fayli nomi")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = collect_files(args.inputs)
    if not files:
        raise SystemExit(f"Tif fayl topilmadi: {args.inputs}")

    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_files:
        idx = rng.choice(len(files), size=args.n_files, replace=False)
        files = [files[i] for i in sorted(idx)]

    from run import get_classifier
    classifier = get_classifier()

    print(f"Fayllar: {len(files)},  impuls ulushi: {args.impulse}, "
          f"Gauss sigma: {args.gauss}")
    print(f"Impuls detektori: {'yakkalik sharti bilan' if args.isolated_impulse else 'oddiy'}\n")

    results, baseline, names = evaluate(files, classifier, args.impulse, args.gauss,
                                  seed=args.seed, isolated=args.isolated_impulse,
                                  strategies=args.strategies)
    print_table(results, baseline)
    if not args.no_plot and len(results[list(results)[0]]) >= 2:
        plot_results(results, names, args.plot_path)
        print(f"\nGrafik saqlandi: {args.plot_path}")


if __name__ == "__main__":
    main()