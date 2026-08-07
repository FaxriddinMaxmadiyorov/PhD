"""
Baholash protokoli va ablatsiya tajribalari
=============================================

Maqoladagi sonli natijalarni ishlab chiqaradi:

  A) shovqin turini kanal darajasida klassifikatsiyalash aniqligi
  B) kanal bo'yicha sigma* baholash aniqligi
  C) ASOSIY ABLATSIYA: kanal bo'yicha ishlov vs kanallar bo'yicha
     o'rtachalash - yakuniy PSNR/SSIM bo'yicha taqqoslash
  D) ierarxik regulyarizatsiyaning hissasi
  E) yopiq tsiklli qayta sozlashning hissasi
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from multispectral_data import make_multispectral_scene, apply_multispectral_noise
from noise_characterization import characterize_noise_per_band
from noise_type_classification import (
    NoiseTypeClassifier, build_calibration_set, features_from_image,
    expand_features, NOISE_TYPES,
)
from noise_level_estimation import estimate_noise_level_per_band
from adaptive_denoising import adaptive_denoise_per_band
from quality_feedback import build_natural_reference
from full_pipeline import run_pipeline

SIZE = 128
N_BANDS = 6


def _ssim_multiband(a, b):
    return float(np.mean([
        ssim(a[:, :, i], b[:, :, i], data_range=1.0) for i in range(a.shape[2])
    ]))


def part_a_classification():
    print("=" * 78)
    print("A) SHOVQIN TURINI KANAL DARAJASIDA KLASSIFIKATSIYALASH")
    print("=" * 78)

    x, y = build_calibration_set(n_scenes=120, size=96, n_bands=N_BANDS, seed=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y,
                                               random_state=0)
    clf = NoiseTypeClassifier(k=21).fit(x_tr, y_tr)
    y_pred = clf.model.predict(clf.scaler.transform(x_te))

    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=NOISE_TYPES)

    print(f"Namunalar (kanal darajasida): {len(x)},  test ulushi 30%")
    print(f"Umumiy aniqlik: {acc:.3f}\n")
    print(f"{'haqiqiy \\ bashorat':<20}" + "".join(f"{t:>14}" for t in NOISE_TYPES))
    for i, t in enumerate(NOISE_TYPES):
        print(f"{t:<20}" + "".join(f"{v:>14}" for v in cm[i]))
    print()
    return clf


def part_b_sigma(n_scenes: int = 25):
    print("=" * 78)
    print("B) KANAL BO'YICHA SIGMA* BAHOLASH ANIQLIGI")
    print("=" * 78)

    rng = np.random.default_rng(5)
    by_type = {t: [] for t in NOISE_TYPES}

    for s in range(n_scenes):
        clean = make_multispectral_scene(SIZE, N_BANDS, seed=3000 + s)
        noisy, truth = apply_multispectral_noise(
            clean, rng, level=float(rng.uniform(0.2, 1.0)), impulse_bands=0.2
        )
        types = [t["type_true"] for t in truth]
        sigmas = estimate_noise_level_per_band(noisy, types)
        for i, t in enumerate(truth):
            rel = abs(sigmas[i] - t["sigma_true"]) / max(t["sigma_true"], 1e-9)
            by_type[t["type_true"]].append(rel)

    print(f"{'Tur':<14}{'Kanallar soni':>16}{'Nisbiy xato (ort)':>22}{'Median':>12}")
    rows = []
    for t in NOISE_TYPES:
        vals = np.array(by_type[t])
        if vals.size == 0:
            continue
        print(f"{t:<14}{vals.size:>16}{100 * vals.mean():>21.1f}%"
              f"{100 * np.median(vals):>11.1f}%")
        rows.append((t, vals.size, 100 * vals.mean(), 100 * np.median(vals)))
    print()
    return rows


def part_c_ablation(clf, n_scenes: int = 12):
    print("=" * 78)
    print("C) ASOSIY ABLATSIYA: KANAL BO'YICHA vs KANALLAR BO'YICHA O'RTACHALASH")
    print("=" * 78)

    rng = np.random.default_rng(77)
    results = {}

    for level in [0.3, 0.6, 0.9]:
        acc_per, acc_avg = [], []
        rows = {"noisy": [], "perband": [], "averaged": []}
        ssims = {"perband": [], "averaged": []}

        for s in range(n_scenes):
            clean = make_multispectral_scene(SIZE, N_BANDS, seed=4000 + s)
            noisy, truth = apply_multispectral_noise(clean, rng, level=level,
                                                      impulse_bands=0.2)
            true_types = [t["type_true"] for t in truth]

            # --- Kanal bo'yicha (taklif etilgan) ---
            feats = features_from_image(noisy)
            types_pb = list(clf.predict_bands(feats))
            sig_pb = estimate_noise_level_per_band(noisy, types_pb)
            out_pb = adaptive_denoise_per_band(noisy, types_pb, sig_pb)

            # --- Kanallar bo'yicha o'rtachalash (bazaviy) ---
            char = characterize_noise_per_band(noisy)
            mean_raw = char["features"].mean(axis=0)
            mean_level = float(noisy.mean())
            feat_avg = expand_features(mean_raw.reshape(1, -1), [mean_level])
            type_avg = str(clf.predict_bands(feat_avg)[0])
            sig_avg = float(np.mean(estimate_noise_level_per_band(noisy, type_avg)))
            out_avg = adaptive_denoise_per_band(noisy, type_avg, sig_avg)

            rows["noisy"].append(psnr(clean, noisy, data_range=1.0))
            rows["perband"].append(psnr(clean, out_pb, data_range=1.0))
            rows["averaged"].append(psnr(clean, out_avg, data_range=1.0))
            ssims["perband"].append(_ssim_multiband(clean, out_pb))
            ssims["averaged"].append(_ssim_multiband(clean, out_avg))

            acc_per.append(np.mean([a == b for a, b in zip(types_pb, true_types)]))
            acc_avg.append(np.mean([type_avg == b for b in true_types]))

        results[level] = {
            "psnr_noisy": float(np.mean(rows["noisy"])),
            "psnr_perband": float(np.mean(rows["perband"])),
            "psnr_averaged": float(np.mean(rows["averaged"])),
            "ssim_perband": float(np.mean(ssims["perband"])),
            "ssim_averaged": float(np.mean(ssims["averaged"])),
            "acc_perband": 100 * float(np.mean(acc_per)),
            "acc_averaged": 100 * float(np.mean(acc_avg)),
        }

    print(f"{'Daraja':>7}{'PSNR shovq.':>13}{'PSNR ortach':>13}{'PSNR kanal':>12}"
          f"{'Farq':>8}{'SSIM ortach':>13}{'SSIM kanal':>12}"
          f"{'Aniq.ort':>10}{'Aniq.kanal':>12}")
    for level, r in results.items():
        diff = r["psnr_perband"] - r["psnr_averaged"]
        print(f"{level:>7.1f}{r['psnr_noisy']:>13.2f}{r['psnr_averaged']:>13.2f}"
              f"{r['psnr_perband']:>12.2f}{diff:>+8.2f}"
              f"{r['ssim_averaged']:>13.3f}{r['ssim_perband']:>12.3f}"
              f"{r['acc_averaged']:>9.0f}%{r['acc_perband']:>11.0f}%")
    print()
    return results


def part_d_shrinkage(clf, n_scenes: int = 12):
    print("=" * 78)
    print("D) IERARXIK REGULYARIZATSIYANING HISSASI")
    print("=" * 78)

    out = {}
    for shrink in [0.0, 0.3, 1.0]:
        rng = np.random.default_rng(91)
        accs = []
        for s in range(n_scenes):
            clean = make_multispectral_scene(SIZE, N_BANDS, seed=5000 + s)
            noisy, truth = apply_multispectral_noise(
                clean, rng, level=float(rng.uniform(0.2, 1.0)), impulse_bands=0.2
            )
            true_types = [t["type_true"] for t in truth]
            feats = features_from_image(noisy, shrinkage=shrink)
            pred = clf.predict_bands(feats)
            accs.append(np.mean([a == b for a, b in zip(pred, true_types)]))
        out[shrink] = 100 * float(np.mean(accs))
        print(f"shrinkage={shrink:.1f}:  tur aniqligi = {out[shrink]:.1f}%")
    print()
    return out


def part_e_feedback(clf, reference, n_scenes: int = 8):
    print("=" * 78)
    print("E) YOPIQ TSIKLLI QAYTA SOZLASHNING HISSASI")
    print("=" * 78)

    out = {}
    for thr in [0.70, 0.85, 0.95]:
        rng = np.random.default_rng(33)
        gains, iters, conv = [], [], []
        for s in range(n_scenes):
            clean = make_multispectral_scene(SIZE, N_BANDS, seed=6000 + s)
            noisy, _ = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)
            res = run_pipeline(noisy, clf, reference, aqi_threshold=thr)
            gains.append(psnr(clean, res["final_image"], data_range=1.0)
                         - psnr(clean, noisy, data_range=1.0))
            iters.append(res["iterations_used"])
            conv.append(res["converged_bands"].mean())
        out[thr] = (float(np.mean(iters)), 100 * float(np.mean(conv)), float(np.mean(gains)))
        print(f"AQI chegara={thr:.2f}:  ort.iteratsiya={np.mean(iters):.2f}  "
              f"yaqinlashgan kanallar={100 * np.mean(conv):.0f}%  "
              f"ort.PSNR yutuq={np.mean(gains):+.2f} dB")
    print()
    return out


if __name__ == "__main__":
    clf = part_a_classification()
    part_b_sigma()
    part_c_ablation(clf)
    part_d_shrinkage(clf)

    print("Tabiiylik referensi hisoblanmoqda...")
    reference = build_natural_reference(make_multispectral_scene, n_scenes=15,
                                         size=96, n_bands=N_BANDS)
    print()
    part_e_feedback(clf, reference)