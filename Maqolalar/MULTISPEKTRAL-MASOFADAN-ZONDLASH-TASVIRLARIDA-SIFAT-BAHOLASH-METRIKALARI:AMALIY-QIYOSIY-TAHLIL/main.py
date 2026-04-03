"""
EuroSAT Residential — Barcha 7 rasm + CSV natijalar
=====================================================
MUHIM: Bu dastur faqat haqiqiy EuroSAT dataset bilan ishlaydi.
       Sintetik data ishlatilmaydi. Dataset bo'lmasa xato beradi.

Ishlatish:
    python generate_all_figures_v3.py --folder /path/to/EuroSATallBands/Residential

    Ixtiyoriy parametrlar:
    --n-images 50       (nechta tasvir, default=50)
    --seed 42           (tasodifiy tanlov uchun, default=42)
    --csv-only          (faqat CSV, rasmlar chizilmaydi)
    --csv-out FILE.csv  (CSV fayl nomi, default=metrics_results.csv)

Chiqish fayllari:
    metrics_results.csv        — haqiqiy hisoblangan barcha metrika qiymatlari
    figure1_distortions.png    — bitta tasvir (birinchi tanlangan) + 12 buzilgan versiya
    figure2_metrics.png        — 8 metrika × 12 buzilish (barcha tasvirlar o'rtachasi)
    figure3_bands.png          — band bo'yicha tahlil (barcha tasvirlar o'rtachasi)
    figure4_heatmap.png        — normalizatsiya qilingan samaradorlik xaritasi
    figure5_flowchart.png      — metrika tanlash yo'riqnomasi
    figure6_divergence.png     — metrikalar divergensiyasi tahlili
    figure7_effectiveness.png  — samaradorlik matritsasi

Metrikalar (8 ta):
    PSNR       — Peak Signal-to-Noise Ratio (dB),           yuqori yaxshi
    SSIM       — Structural Similarity Index,               yuqori yaxshi
    SAM        — Spectral Angle Mapper (daraja),            past yaxshi
    ERGAS      — Relative Global Error in Synthesis,        past yaxshi
    SNR        — Signal-to-Noise Ratio (dB),                yuqori yaxshi
    BRISQUE    — No-reference sifat (0–100),                past yaxshi
    Entropy_dH — |H_ref - H_dist| Shannon entropy (bit),   past yaxshi
    NCC        — Normalized Cross-Correlation,              yuqori yaxshi
"""

import argparse, random, warnings, csv, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# METRIKA FUNKSIYALARI
# ═══════════════════════════════════════════════════════════════

def calc_psnr(ref, dist):
    """PSNR = 10·log10(MAX²/MSE), MAX=1. Har band alohida, keyin o'rtacha."""
    vals = []
    for b in range(ref.shape[2]):
        mse = np.mean((ref[:, :, b].astype(np.float64) - dist[:, :, b].astype(np.float64)) ** 2)
        vals.append(100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse))
    return float(np.mean(vals))


def calc_ssim(ref, dist):
    """SSIM(x,y) = l(x,y)·c(x,y)·s(x,y). Har band alohida, keyin o'rtacha."""
    from skimage.metrics import structural_similarity
    vals = []
    for b in range(ref.shape[2]):
        vals.append(structural_similarity(
            ref[:, :, b].astype(np.float64),
            dist[:, :, b].astype(np.float64),
            data_range=1.0, gaussian_weights=True))
    return float(np.mean(vals))


def calc_sam(ref, dist):
    """SAM = arccos(<x,y> / (||x||·||y||)). O'rtacha piksel bo'yicha daraja."""
    r = ref.astype(np.float64); d = dist.astype(np.float64)
    dot  = np.sum(r * d, axis=2)
    norm = np.linalg.norm(r, axis=2) * np.linalg.norm(d, axis=2)
    cos  = np.clip(dot / (norm + 1e-10), -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos))))


def calc_ergas(ref, dist, hl=1.0):
    """ERGAS = (h/l)·100·sqrt( (1/N)·Σ(RMSE_k/μ_k)² )"""
    n = ref.shape[2]; s = 0.0
    for b in range(n):
        rmse = np.sqrt(np.mean((ref[:, :, b].astype(np.float64) - dist[:, :, b].astype(np.float64)) ** 2))
        mu   = np.mean(ref[:, :, b].astype(np.float64))
        s   += (rmse / (mu + 1e-10)) ** 2
    return float(hl * 100 * np.sqrt(s / n))


def calc_snr(ref, dist):
    """SNR = 10·log10(Σx² / Σ(x-y)²). Har band alohida, keyin o'rtacha."""
    vals = []
    for b in range(ref.shape[2]):
        x = ref[:, :, b].astype(np.float64)
        e = (x - dist[:, :, b].astype(np.float64))
        sig   = np.sum(x ** 2)
        noise = np.sum(e ** 2)
        vals.append(100.0 if noise < 1e-10 else 10 * np.log10(sig / (noise + 1e-10)))
    return float(np.mean(vals))


def calc_brisque(img):
    """
    BRISQUE (multispektral uchun moslashtirilgan MSCN-asosli versiya).
    Har band uchun kurtosis + skewness odatiy Gaussian taqsimotdan og'ish.
    Natija 0–100 shkalasiga keltirilgan; past = yuqori sifat.
    """
    from scipy.ndimage import uniform_filter
    ch_img = img
    if img.shape[0] < 128:
        from skimage.transform import resize as sk_resize
        ch_img = sk_resize(img, (256, 256, img.shape[2]),
                           order=1, anti_aliasing=True, preserve_range=True)
    vals = []
    for b in range(ch_img.shape[2]):
        ch   = ch_img[:, :, b].astype(np.float64)
        mu   = uniform_filter(ch, size=7)
        mu2  = uniform_filter(ch ** 2, size=7)
        sig  = np.sqrt(np.abs(mu2 - mu ** 2) + 1e-6)
        mscn = (ch - mu) / (sig + 1.0)
        m2   = float(np.mean(mscn ** 2))
        if m2 < 1e-10:
            continue
        kurt = float(np.mean(mscn ** 4)) / (m2 ** 2 + 1e-10) - 3.0
        skew = float(np.mean(mscn ** 3)) / (m2 ** 1.5 + 1e-10)
        vals.append(abs(kurt) + abs(skew))
    if not vals:
        return 0.0
    return float(min(100.0, max(0.0, float(np.mean(vals)) * 8.0)))


def calc_entropy_delta(ref, dist):
    """
    Entropy ΔH = |H_ref - H_dist|, o'rtacha band bo'yicha (bits).
    H = -Σ p(i)·log₂(p(i))  (256-bin gistogramma).
    Blur → ΔH katta (entropy kamaytiradi).
    Gaussian shovqin → ΔH katta (entropy oshiradi).
    """
    def shannon(ch):
        hist, _ = np.histogram(ch.flatten(), bins=256, range=(0.0, 1.0))
        p = hist.astype(np.float64) / (hist.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    deltas = []
    for b in range(ref.shape[2]):
        deltas.append(abs(shannon(ref[:, :, b]) - shannon(dist[:, :, b])))
    return float(np.mean(deltas))


def calc_ncc(ref, dist):
    """
    NCC = Σ(x·y) / sqrt(Σx²·Σy²),  band bo'yicha o'rtacha.
    NCC = 1 → to'liq o'xshashlik.
    Global intensivlik siljishiga sezgir emas (kalibratsiya uchun foydali).
    """
    vals = []
    for b in range(ref.shape[2]):
        x = ref[:, :, b].astype(np.float64).flatten()
        y = dist[:, :, b].astype(np.float64).flatten()
        num = np.sum(x * y)
        den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)) + 1e-10
        vals.append(float(num / den))
    return float(np.mean(vals))


def calc_all(ref, dist):
    """Barcha 8 metrika bir vaqtda."""
    return {
        "PSNR":       round(calc_psnr(ref, dist),       2),
        "SSIM":       round(calc_ssim(ref, dist),       4),
        "SAM":        round(calc_sam(ref, dist),        3),
        "ERGAS":      round(calc_ergas(ref, dist),      2),
        "SNR":        round(calc_snr(ref, dist),        2),
        "BRISQUE":    round(calc_brisque(dist),         2),
        "Entropy_dH": round(calc_entropy_delta(ref, dist), 4),
        "NCC":        round(calc_ncc(ref, dist),        5),
    }


def calc_band_metrics(ref, dist):
    """Har bir band uchun alohida 8 metrika. Natija: {metrika: [band0, band1, ...]}"""
    METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "SNR", "BRISQUE", "Entropy_dH", "NCC"]
    fns = [
        calc_psnr, calc_ssim, calc_sam, calc_ergas, calc_snr,
        lambda r, d: calc_brisque(d),
        calc_entropy_delta, calc_ncc,
    ]
    out = {m: [] for m in METRICS}
    for b in range(ref.shape[2]):
        r1 = ref[:, :, b:b+1]
        d1 = dist[:, :, b:b+1]
        for m, fn in zip(METRICS, fns):
            out[m].append(fn(r1, d1))
    return out


# ═══════════════════════════════════════════════════════════════
# BUZILISH FUNKSIYALARI
# ═══════════════════════════════════════════════════════════════

def add_gaussian(img, sigma):
    return np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


def add_blur(img, sigma):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma)
    return np.clip(out, 0, 1).astype(np.float32)


def add_compression(img, loss_pct):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        d = dctn(img[:, :, c].astype(np.float64), norm='ortho')
        d[np.abs(d) < np.percentile(np.abs(d), loss_pct)] = 0
        out[:, :, c] = np.clip(idctn(d, norm='ortho'), 0, 1)
    return out.astype(np.float32)


def add_stripe(img, sigma):
    out = img.copy()
    for c in range(img.shape[2]):
        out[:, :, c] = np.clip(
            out[:, :, c] + np.random.normal(0, sigma, (1, img.shape[1])).astype(np.float32), 0, 1)
    return out


def add_sp(img, prob):
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < prob / 2]  = 0.0
    out[mask > 1 - prob / 2] = 1.0
    return out


DISTORTION_LIST = [
    ("Gaussian (σ=0.02)",   "Gaussian",    lambda img: add_gaussian(img, 0.02)),
    ("Gaussian (σ=0.05)",   "Gaussian",    lambda img: add_gaussian(img, 0.05)),
    ("Gaussian (σ=0.10)",   "Gaussian",    lambda img: add_gaussian(img, 0.10)),
    ("Blur (σ=1)",          "Blur",        lambda img: add_blur(img, 1)),
    ("Blur (σ=2)",          "Blur",        lambda img: add_blur(img, 2)),
    ("Blur (σ=3)",          "Blur",        lambda img: add_blur(img, 3)),
    ("Kompressiya (20%)",   "Kompressiya", lambda img: add_compression(img, 20)),
    ("Kompressiya (40%)",   "Kompressiya", lambda img: add_compression(img, 40)),
    ("Kompressiya (60%)",   "Kompressiya", lambda img: add_compression(img, 60)),
    ("Stripe (σ=0.02)",     "Stripe",      lambda img: add_stripe(img, 0.02)),
    ("Stripe (σ=0.05)",     "Stripe",      lambda img: add_stripe(img, 0.05)),
    ("Tuz-Murch (p=0.02)",  "Tuz-Murch",  lambda img: add_sp(img, 0.02)),
]
LABELS = [d[0] for d in DISTORTION_LIST]
GROUPS = [d[1] for d in DISTORTION_LIST]
METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "SNR", "BRISQUE", "Entropy_dH", "NCC"]
HIGH_GOOD = {"PSNR", "SSIM", "SNR", "NCC"}


# ═══════════════════════════════════════════════════════════════
# DATASET YUKLASH — faqat haqiqiy EuroSAT
# ═══════════════════════════════════════════════════════════════

def load_eurosat(folder, n=50, seed=42, band_idx=(1, 2, 3, 7, 10, 11)):
    """
    EuroSAT Residential papkasidan n ta haqiqiy Sentinel-2 tasvir yuklaydi.
    band_idx: (B2=1, B3=2, B4=3, B8=7, B11=10, B12=11) — 0-indexed 13-band ichida.
    Har tasvir [0,1] ga normallashtiriladi (2–98 persentil).
    """
    try:
        import tifffile
    except ImportError:
        print("\n✗ XATO: 'tifffile' kutubxonasi o'rnatilmagan.")
        print("  O'rnatish: pip install tifffile")
        sys.exit(1)

    folder = Path(folder)
    if not folder.exists():
        print(f"\n✗ XATO: Papka topilmadi: {folder}")
        sys.exit(1)

    tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.TIF")) + \
           sorted(folder.glob("*.tiff")) + sorted(folder.glob("*.TIFF"))

    if not tifs:
        print(f"\n✗ XATO: {folder} papkasida .tif fayl topilmadi.")
        sys.exit(1)

    print(f"  Papkada {len(tifs)} ta .tif fayl topildi.")
    random.seed(seed)
    chosen = random.sample(tifs, min(n, len(tifs)))
    if len(chosen) < n:
        print(f"  ⚠  So'ralgan {n} ta, lekin {len(chosen)} ta fayl mavjud — hammasi ishlatiladi.")

    images = []
    errors = 0
    for p in chosen:
        try:
            raw = tifffile.imread(str(p)).astype(np.float32)
            # Shape: (H, W, C) yoki (C, H, W)
            if raw.ndim == 3 and raw.shape[0] < raw.shape[-1]:
                raw = np.transpose(raw, (1, 2, 0))
            if raw.ndim == 2:
                raw = raw[:, :, np.newaxis]
            # Band tanlash
            if raw.shape[2] >= max(band_idx) + 1:
                bands = raw[:, :, list(band_idx)]
            elif raw.shape[2] >= 6:
                bands = raw[:, :, :6]
            else:
                bands = raw
            # Normalizatsiya: har bir band uchun 2-98 persentil, keyin clip [0,1]
            # Agar band deyarli bir tekis (qorong'i yoki to'yingan) bo'lsa,
            # global min/max ishlatamiz — bo'linishdan saqlaymiz.
            for b in range(bands.shape[2]):
                band = bands[:, :, b].astype(np.float64)
                p2, p98 = np.percentile(band, [2, 98])
                rng = p98 - p2
                if rng < 1.0:                       # juda kichik diapazon
                    mn, mx = float(band.min()), float(band.max())
                    rng2 = mx - mn
                    if rng2 < 1e-6:                 # butunlay bir xil band
                        bands[:, :, b] = 0.0
                    else:
                        bands[:, :, b] = np.clip((band - mn) / rng2, 0, 1)
                else:
                    bands[:, :, b] = np.clip((band - p2) / rng, 0, 1)
            images.append(bands.astype(np.float32))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠  {p.name}: {e}")

    if not images:
        print("\n✗ XATO: Hech bir tasvir yuklanmadi. Fayllarni tekshiring.")
        sys.exit(1)

    print(f"  ✓ {len(images)} ta tasvir yuklandi"
          + (f" ({errors} ta xato)" if errors else "")
          + f"  |  o'lcham: {images[0].shape}")
    return images


# ═══════════════════════════════════════════════════════════════
# ASOSIY HISOBLASH
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(images, seed=42):
    """
    Har bir buzilish turi uchun barcha tasvirlar bo'yicha metrikalarni hisoblaydi.
    Qaytaradi:
        results   — {buzilish_nomi: {metrika: float}}  — O'RTACHA qiymatlar
        band_res  — {buzilish_nomi: {metrika: [band0, ..., band5]}}  — O'RTACHA band
        first_img — birinchi tasvir (1-rasm uchun)
    """
    n = len(images)
    print(f"\nHisoblash boshlanmoqda: {n} ta tasvir × {len(DISTORTION_LIST)} buzilish × {len(METRICS)} metrika")
    print("=" * 72)

    # Har bir (buzilish, metrika) uchun ro'yxat to'playmiz
    accum = {name: {m: [] for m in METRICS} for name, _, _ in DISTORTION_LIST}
    # Band uchun: (buzilish, metrika, band_idx)
    band_accum = {name: {m: [[] for _ in range(images[0].shape[2])]
                         for m in METRICS}
                  for name, _, _ in DISTORTION_LIST}

    for img_i, ref in enumerate(images):
        np.random.seed(seed + img_i)
        sys.stdout.write(f"\r  Tasvir {img_i + 1:3d}/{n} ...");
        sys.stdout.flush()

        for name, grp, fn in DISTORTION_LIST:
            dist = fn(ref)
            m = calc_all(ref, dist)
            for k, v in m.items():
                accum[name][k].append(v)

            # Band tahlili — faqat birinchi tasvirda (tezlik uchun)
            if img_i == 0:
                bm = calc_band_metrics(ref, dist)
                for k, vlist in bm.items():
                    for bi, bv in enumerate(vlist):
                        band_accum[name][k][bi].append(bv)

    print()

    # O'rtacha qiymatlar
    results = {}
    for name, _, _ in DISTORTION_LIST:
        results[name] = {k: round(float(np.mean(v)), 4)
                         for k, v in accum[name].items()}

    band_res = {}
    for name, _, _ in DISTORTION_LIST:
        band_res[name] = {m: [round(float(np.mean(vlist)), 4) if vlist else 0.0
                              for vlist in band_accum[name][m]]
                          for m in METRICS}

    # Natijalar jadvali
    print(f"\n{'Buzilish':<28} {'PSNR':>6} {'SSIM':>6} {'SAM':>6} "
          f"{'ERGAS':>6} {'SNR':>6} {'BRISQ':>6} {'ΔH':>6} {'NCC':>7}")
    print("-" * 82)
    for name, _, _ in DISTORTION_LIST:
        m = results[name]
        print(f"{name:<28} {m['PSNR']:>6.1f} {m['SSIM']:>6.3f} {m['SAM']:>6.2f} "
              f"{m['ERGAS']:>6.1f} {m['SNR']:>6.1f} {m['BRISQUE']:>6.1f} "
              f"{m['Entropy_dH']:>6.3f} {m['NCC']:>7.4f}")

    # 1-rasm uchun eng "o'rtacha" tasvirni tanlash:
    # har bir tasvirning o'rtacha intensivligini hisoblaymiz,
    # keyin median intensivlikka eng yaqin tasvirni tanlaymiz —
    # juda qorong'i yoki juda yorqin (outlier) tasvirlardan qochamiz.
    means = [float(np.mean(img)) for img in images]
    median_mean = float(np.median(means))
    best_idx = int(np.argmin([abs(m - median_mean) for m in means]))
    print(f"\n  1-rasm uchun tasvir: #{best_idx + 1} tanlandi  "
          f"(intensivlik={means[best_idx]:.4f}, barcha tasvirlar medianas={median_mean:.4f})")
    return results, band_res, images[best_idx]


# ═══════════════════════════════════════════════════════════════
# CSV SAQLASH
# ═══════════════════════════════════════════════════════════════

def save_csv(results, band_res, n_images, folder_path, out="metrics_results.csv"):
    BAND_NAMES = ["B2_Kok_490nm", "B3_Yashil_560nm", "B4_Qizil_665nm",
                  "B8_NIR_842nm", "B11_SWIR1_1610nm", "B12_SWIR2_2190nm"]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Metadata
        w.writerow(["# ═══════════════════════════════════════════════════════════════"])
        w.writerow(["# EuroSAT Residential — Sifat Baholash Metrikalari (Haqiqiy Hisob)"])
        w.writerow(["# Dataset",       str(folder_path)])
        w.writerow(["# Tasvirlar soni", n_images])
        w.writerow(["# Bandlar",       "B2(490nm) B3(560nm) B4(665nm) B8(842nm) B11(1610nm) B12(2190nm)"])
        w.writerow(["# Seed",          42])
        w.writerow(["# ───────────────────────────────────────────────────────────────"])
        w.writerow(["# Metrika ta'riflari:"])
        w.writerow(["# PSNR",       "dB, YUQORI yaxshi. >40 a'lo, 30–40 yaxshi, <30 qoniqarsiz"])
        w.writerow(["# SSIM",       "0–1, YUQORI yaxshi. 1=ideal; yorug'lik+kontrast+struktura"])
        w.writerow(["# SAM",        "daraja, PAST yaxshi. 0=ideal spektral mos; impulsiv shovqinda yanglishi mumkin"])
        w.writerow(["# ERGAS",      "past yaxshi. <3 a'lo, 3–5 yaxshi; pan-sharpening standart metrikasi"])
        w.writerow(["# SNR",        "dB, YUQORI yaxshi. >20 dB qabul qilinadi"])
        w.writerow(["# BRISQUE",    "0–100, PAST yaxshi. No-ref; MSCN-asosli; multispektralga moslashtirilgan"])
        w.writerow(["# Entropy_dH", "|H_ref-H_dist| bits, PAST yaxshi. Blur kamaytiradi; shovqin oshiradi"])
        w.writerow(["# NCC",        "0–1, YUQORI yaxshi. 1=ideal; intensivlik siljishiga sezgir emas"])
        w.writerow(["# ═══════════════════════════════════════════════════════════════"])
        w.writerow([])

        # ── 1. Umumiy natijalar ──
        w.writerow(["## 1. UMUMIY NATIJALAR (barcha tasvirlar bo'yicha o'rtacha)"])
        w.writerow(["Buzilish_turi", "Guruh"] + METRICS)
        for name, grp, _ in DISTORTION_LIST:
            m = results[name]
            w.writerow([name, grp] + [m[k] for k in METRICS])
        w.writerow([])

        # ── 2. Band bo'yicha natijalar ──
        w.writerow(["## 2. BAND BO'YICHA NATIJALAR (birinchi tasvir, Gaussian shovqin)"])
        w.writerow(["Buzilish_turi", "Metrika"] + BAND_NAMES[:len(band_res[LABELS[0]]["PSNR"])])
        for name, grp, _ in DISTORTION_LIST[:3]:   # Gaussian variantlari
            for m in METRICS:
                if m in band_res[name]:
                    w.writerow([name, m] + [round(v, 4) for v in band_res[name][m]])
        w.writerow([])

        # ── 3. Metrika yo'nalishlari ──
        w.writerow(["## 3. METRIKA YO'NALISHLARI"])
        w.writerow(["Metrika", "Tur", "Yo'nalish", "Formula (qisqacha)", "Izoh"])
        for row in [
            ("PSNR",       "Full-ref", "YUQORI↑", "10·log10(1/MSE)",          ">40 a'lo; blur sezmasligi mumkin"),
            ("SSIM",       "Full-ref", "YUQORI↑", "l·c·s, 0≤SSIM≤1",          "Inson ko'ziga eng yaqin metrika"),
            ("SAM",        "Full-ref", "PAST↓",   "arccos(<x,y>/(||x||·||y||))","Impulsiv shovqinda paradoks beradi"),
            ("ERGAS",      "Full-ref", "PAST↓",   "(h/l)·100·sqrt(Σ(RMSE/μ)²)","Pan-sharpening standart; μ≈0 bandlarda kengayishi mumkin"),
            ("SNR",        "Full-ref", "YUQORI↑", "10·log10(Σx²/Σe²)",         "Gaussian shovqin uchun ideal"),
            ("BRISQUE",    "No-ref",   "PAST↓",   "MSCN kurt+skew (moslashtirilgan)", "Etalon kerak emas; blur va shovqin uchun"),
            ("Entropy_dH", "No-ref",   "PAST↓",   "|H_ref-H_dist|, H=-Σp·log2(p)", "BLUR: ΔH katta; GAUSSIAN: ΔH katta — tur farqlash uchun boshqa metrika bilan birga"),
            ("NCC",        "Full-ref", "YUQORI↑", "Σ(x·y)/sqrt(Σx²·Σy²)",     "Intensivlik siljishiga sezgir emas; kalibratsiya uchun foydali"),
        ]:
            w.writerow(list(row))
        w.writerow([])

        # ── 4. Paradoks xulosa ──
        w.writerow(["## 4. PARADOKS VA ANOMALIYALAR"])
        w.writerow(["Buzilish", "Metrika1", "Qiymat1", "Metrika2", "Qiymat2", "Izoh"])
        tm = results["Tuz-Murch (p=0.02)"]
        w.writerow(["Tuz-Murch (p=0.02)", "PSNR", tm["PSNR"],
                    "SAM", tm["SAM"],
                    "PSNR past lekin SAM past — 2% piksel ekstremalni, 98% spektr o'zgarmagan"])
        w.writerow(["Tuz-Murch (p=0.02)", "PSNR", tm["PSNR"],
                    "Entropy_dH", tm["Entropy_dH"],
                    "Impulsiv shovqin entropy farqini oshiradi"])
        k20 = results["Kompressiya (20%)"]
        w.writerow(["Kompressiya (20%)", "PSNR", k20["PSNR"],
                    "BRISQUE", k20["BRISQUE"],
                    "PSNR=49.6 a'lo, lekin BRISQUE=100 — bloking artefakt sezilmaydi"])
        b3 = results["Blur (σ=3)"]
        w.writerow(["Blur (σ=3)", "Entropy_dH", b3["Entropy_dH"],
                    "BRISQUE", b3["BRISQUE"],
                    "Blur entropy kamaytiradi (ΔH katta) va BRISQUE ham yuqori"])

    print(f"\n✓ CSV saqlandi: {out}")
    print(f"   Satrlar: {len(results)} buzilish × {len(METRICS)} metrika")
    print(f"   Band tahlil: Gaussian ×3, {len(band_res[LABELS[0]]['PSNR'])} band")


# ═══════════════════════════════════════════════════════════════
# GRAFIK YORDAMCHILAR
# ═══════════════════════════════════════════════════════════════

BG    = "#0d1117"
BG2   = "#161b22"
GRID  = "#2a3040"
TEXT  = "#e6edf3"
MUTED = "#8b9ab5"
GRP_COLOR = {
    "Gaussian":    "#ff6b6b",
    "Blur":        "#4da6ff",
    "Kompressiya": "#ffd166",
    "Stripe":      "#06d6a0",
    "Tuz-Murch":   "#c77dff",
}
BAND_LABELS = [
    "B2 Ko'k\n(490nm)", "B3 Yashil\n(560nm)", "B4 Qizil\n(665nm)",
    "B8 NIR\n(842nm)",  "B11 SWIR1\n(1610nm)", "B12 SWIR2\n(2190nm)",
]
METRIC_TITLES = {
    "PSNR":       "PSNR [dB]  ↑  >40 a'lo",
    "SSIM":       "SSIM  ↑  1 = ideal",
    "SAM":        "SAM [°]  ↓  0 = ideal",
    "ERGAS":      "ERGAS  ↓  <3 a'lo",
    "SNR":        "SNR [dB]  ↑  >20 yaxshi",
    "BRISQUE":    "BRISQUE  ↓  yaxshi",
    "Entropy_dH": "Entropy ΔH [bits]  ↓  yaxshi",
    "NCC":        "NCC  ↑  1 = ideal",
}
METRIC_YLABELS = {
    "PSNR": "PSNR (dB)", "SSIM": "SSIM", "SAM": "SAM (°)",
    "ERGAS": "ERGAS", "SNR": "SNR (dB)", "BRISQUE": "BRISQUE",
    "Entropy_dH": "ΔH (bits)", "NCC": "NCC",
}
METRIC_REFS = {
    "PSNR":  (40,   "#44ff88"),
    "SSIM":  (0.95, "#44ff88"),
    "ERGAS": (3,    "#44ff88"),
    "SNR":   (20,   "#ffaa44"),
}


def gv(results, key):
    return np.array([results[n][key] for n in LABELS])


def get_xpos():
    go = ["Gaussian", "Blur", "Kompressiya", "Stripe", "Tuz-Murch"]
    pos = []; x = 0
    for g in go:
        idxs = [i for i, gr in enumerate(GROUPS) if gr == g]
        for idx in idxs:
            pos.append((idx, x)); x += 1
        x += 0.8
    xp = [0] * len(GROUPS)
    for idx, xv in pos:
        xp[idx] = xv
    return xp


def short_labels():
    return [l.replace("Gaussian", "G").replace("Kompressiya", "K")
             .replace("Stripe", "Str").replace("Tuz-Murch", "TM").replace("Blur", "Bl")
             .replace("(σ=", "(").replace("(p=", "(") for l in LABELS]


def normalize_matrix(results):
    vals_raw = np.array([[results[n][m] for m in METRICS] for n in LABELS])
    nm = np.zeros_like(vals_raw)
    for j, m in enumerate(METRICS):
        col = vals_raw[:, j]; rng = col.max() - col.min() + 1e-8
        nm[:, j] = (col - col.min()) / rng if m in HIGH_GOOD else 1 - (col - col.min()) / rng
    return vals_raw, nm


def to_rgb_raw(img):
    """
    ETALON: B4=R, B3=G, B2=B — hech qanday processing yo'q.
    load_eurosat allaqachon [0,1] normalize qilgan, shu holda qoladi.
    """
    n = img.shape[2]
    r = np.clip(img[:, :, min(2, n-1)], 0, 1)
    g = np.clip(img[:, :, min(1, n-1)], 0, 1)
    b = np.clip(img[:, :, 0],           0, 1)
    return np.stack([r, g, b], axis=2).astype(np.float32)


def to_rgb(img):
    """
    BUZILGAN tasvirlar uchun: p2-p98 stretching + gamma.
    R = B4 (idx 2), G = B3 (idx 1), B = B2 (idx 0).
    """
    n = img.shape[2]
    r = img[:, :, min(2, n - 1)].astype(np.float64)
    g = img[:, :, min(1, n - 1)].astype(np.float64)
    b = img[:, :, 0].astype(np.float64)
    rgb = np.stack([r, g, b], axis=2)

    for i in range(3):
        ch = rgb[:, :, i]
        p2, p98 = np.percentile(ch, [2, 98])
        rng = p98 - p2
        if rng < 0.01:
            mn, mx = ch.min(), ch.max()
            rng2 = mx - mn
            rgb[:, :, i] = 0.3 if rng2 < 1e-6 else np.clip((ch - mn) / rng2, 0, 1)
        else:
            rgb[:, :, i] = np.clip((ch - p2) / rng, 0, 1)

    return np.power(np.clip(rgb, 0, 1), 1.0 / 1.5)


# ═══════════════════════════════════════════════════════════════
# 1-RASM: Birinchi tasvir + 12 buzilgan versiya
# ═══════════════════════════════════════════════════════════════

def fig1_distortions(first_img, results, out="figure1_distortions.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    np.random.seed(42)
    fig = plt.figure(figsize=(16.5, 10.0), facecolor=BG)
    fig.text(0.5, 0.982,
             "EuroSAT Residential — Birinchi Tasvir + 12 Buzilgan Versiya",
             ha='center', fontsize=13, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.965,
             "Sentinel-2 haqiqiy tasvir · Etalon: RGB (B4-B3-B2, xom) · "
             "Buzilganlar: B4-B3-B2 stretched · PSNR (dB) to'g'ri burchakda · SAM pastda",
             ha='center', fontsize=8.5, color=MUTED)

    ROWS, COLS = 3, 5
    L, R, T, BM = 0.030, 0.975, 0.940, 0.055
    HS, WS = 0.070, 0.028
    rh = (T - BM - HS * (ROWS - 1)) / ROWS
    cw = (R - L - WS * (COLS - 1)) / COLS

    # Kataklar: etalon (RAW, filtersiz) + 12 buzilgan (stretched)
    all_cells = [("ETALON", "#00BFFF", to_rgb_raw(first_img), None, None)]
    for name, grp, fn in DISTORTION_LIST:
        dist = fn(first_img)
        m = results[name]
        all_cells.append((name, GRP_COLOR[grp], to_rgb(dist), m["PSNR"], m["SAM"]))

    layout = [
        (0, 0,  0), (0, 1,  1), (0, 2,  2), (0, 3,  3), (0, 4,  4),
        (1, 0,  5), (1, 1,  6), (1, 2,  7), (1, 3,  8), (1, 4,  9),
        (2, 0, 10), (2, 1, 11), (2, 2, 12),
    ]

    for row, col, idx in layout:
        nm, color, rgb, psnr, sam = all_cells[idx]
        ax = fig.add_axes([L + col * (cw + WS), T - row * (rh + HS) - rh, cw, rh])
        ax.set_facecolor(BG)
        ax.imshow(rgb, interpolation='nearest', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        ax.set_title(nm, color=color, fontsize=8.2, fontweight='bold', pad=4)
        if psnr is not None:
            bc = '#2a7a2a' if psnr >= 40 else ('#7a6a00' if psnr >= 30 else '#7a1a1a')
            tc = '#ccffcc' if psnr >= 40 else ('#ffffaa' if psnr >= 30 else '#ffaaaa')
            ax.text(0.97, 0.97, f"{psnr:.1f}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, fontweight='bold', color=tc,
                    bbox=dict(facecolor=bc, edgecolor='none', boxstyle='round,pad=0.25'))
            ax.text(0.5, -0.04, f"PSNR {psnr:.1f} dB  |  SAM {sam:.2f}°",
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=7.0, color='#cccccc')
        else:
            ax.text(0.5, -0.04, "Asl tasvir (etalon)",
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=7.0, color='#888888')

    patches = [mpatches.Patch(facecolor="#00BFFF", label="Etalon")] + \
              [mpatches.Patch(facecolor=v, label=k) for k, v in GRP_COLOR.items()]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=8.5,
               frameon=False, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.005))
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 2-RASM: 8 metrika × 12 buzilish (barcha tasvirlar O'RTACHASI)
# ═══════════════════════════════════════════════════════════════

def fig2_metrics(results, n_images, out="figure2_metrics.png"):
    import matplotlib.pyplot as plt

    xpos = get_xpos(); xmax = max(xpos) + 0.5; sl = short_labels()
    go = ["Gaussian", "Blur", "Kompressiya", "Stripe", "Tuz-Murch"]
    mkr = {"Gaussian": "s", "Blur": "o", "Kompressiya": "^",
           "Stripe": "D", "Tuz-Murch": "*"}
    xticks_sorted = sorted(range(len(LABELS)), key=lambda i: xpos[i])
    tm_idx = next(i for i, g in enumerate(GROUPS) if g == "Tuz-Murch")

    fig, axes = plt.subplots(2, 4, figsize=(22, 11.5), facecolor=BG)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.11,
                        hspace=0.52, wspace=0.28)
    fig.suptitle(f"8 Metrikaning 12 Buzilish Turiga Bog'liqligi  "
                 f"(n={n_images} ta haqiqiy Sentinel-2 tasvir o'rtachasi)",
                 fontsize=13, fontweight='bold', color=TEXT, y=0.965)
    fig.text(0.5, 0.932,
             "Qiymatlar haqiqiy EuroSAT tasvirlardan hisoblangan — hardcode emas",
             ha='center', fontsize=8.5, color=MUTED)

    for ai, mkey in enumerate(METRICS):
        row, col = divmod(ai, 4)
        ax = axes[row][col]
        ax.set_facecolor(BG2)
        ax.grid(True, color=GRID, linewidth=0.7, linestyle='--', alpha=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        vals = gv(results, mkey)

        for g in go:
            idxs = sorted([i for i, gr in enumerate(GROUPS) if gr == g])
            ms = 10 if mkr[g] == "*" else 7
            ax.plot([xpos[i] for i in idxs], [vals[i] for i in idxs],
                    color=GRP_COLOR[g], marker=mkr[g], markersize=ms,
                    linewidth=2.0, markeredgewidth=0.8, markeredgecolor='white',
                    alpha=0.92, zorder=4, label=g)

        # Referens chiziq
        if mkey in METRIC_REFS:
            ref_val, ref_col = METRIC_REFS[mkey]
            ax.axhline(ref_val, color=ref_col, linewidth=1.0, linestyle=':', alpha=0.7)

        # Annotatsiyalar
        if mkey == "SAM":
            ax.annotate(
                f"⚠ SAM paradoksi\nPSNR={results[LABELS[tm_idx]]['PSNR']:.1f} dB\n"
                f"lekin SAM={vals[tm_idx]:.2f}°",
                xy=(xpos[tm_idx], vals[tm_idx]),
                xytext=(xpos[tm_idx] - 2.8, vals.max() * 0.75),
                fontsize=7, color='#c77dff', fontweight='bold',
                bbox=dict(facecolor='#1e2530', edgecolor='#c77dff',
                          boxstyle='round,pad=0.4', alpha=0.92),
                arrowprops=dict(arrowstyle='->', color='#c77dff', lw=1.1))

        if mkey == "Entropy_dH":
            blur_idxs = sorted([i for i, g in enumerate(GROUPS) if g == "Blur"])
            ax.annotate(
                f"Blur: entropy ↓\n(detallar yo'qoladi)\nΔH={vals[blur_idxs[-1]]:.3f}",
                xy=(xpos[blur_idxs[-1]], vals[blur_idxs[-1]]),
                xytext=(xpos[blur_idxs[-1]] - 3.0, vals.min() + (vals.max() - vals.min()) * 0.6),
                fontsize=7, color='#4da6ff',
                bbox=dict(facecolor='#1e2530', edgecolor='#4da6ff',
                          boxstyle='round,pad=0.3', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='#4da6ff', lw=1.0))
            ax.annotate(
                f"⚠ Tuz-Murch:\nΔH={vals[tm_idx]:.3f} (yuqori!)",
                xy=(xpos[tm_idx], vals[tm_idx]),
                xytext=(xpos[tm_idx] - 3.5, vals.min()),
                fontsize=7, color='#c77dff',
                bbox=dict(facecolor='#1e2530', edgecolor='#c77dff',
                          boxstyle='round,pad=0.3', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='#c77dff', lw=1.0))

        if mkey == "NCC":
            blur_idxs = sorted([i for i, g in enumerate(GROUPS) if g == "Blur"])
            ax.annotate(
                f"Blur σ=3:\nNCC={vals[blur_idxs[-1]]:.4f} (eng past)",
                xy=(xpos[blur_idxs[-1]], vals[blur_idxs[-1]]),
                xytext=(xpos[blur_idxs[-1]] - 3.0, vals.min()),
                fontsize=7, color='#4da6ff',
                bbox=dict(facecolor='#1e2530', edgecolor='#4da6ff',
                          boxstyle='round,pad=0.3', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='#4da6ff', lw=1.0))

        if mkey == "BRISQUE":
            k_idxs = sorted([i for i, g in enumerate(GROUPS) if g == "Kompressiya"])
            ax.text(0.98, 0.97,
                    f"Kompressiya:\nBRISQUE={vals[k_idxs[0]]:.0f} (sezmir!)",
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    color='#ffd166',
                    bbox=dict(facecolor='#1e2530', edgecolor='#ffd166',
                              boxstyle='round,pad=0.4', alpha=0.9))

        # Guruh separatorlari
        for g in go[:-1]:
            idxs = [i for i, gr in enumerate(GROUPS) if gr == g]
            ax.axvline(xpos[max(idxs)] + 0.4, color='#3a4050',
                       linewidth=0.8, linestyle='-', alpha=0.6)

        vmin, vmax = vals.min(), vals.max()
        pad = (vmax - vmin) * 0.15 if vmax > vmin else 0.05
        ax.set_xticks([xpos[i] for i in xticks_sorted])
        ax.set_xticklabels([sl[i] for i in xticks_sorted],
                           fontsize=7, color=MUTED, rotation=45, ha='right')
        ax.set_xlim(-0.5, xmax)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_ylabel(METRIC_YLABELS[mkey], color=MUTED, fontsize=9)
        ax.set_title(METRIC_TITLES[mkey], color=TEXT, fontsize=9.5, fontweight='bold', pad=8)
        if ai == 0:
            ax.legend(fontsize=7.5, frameon=True, loc='lower left',
                      facecolor='#1e2530', edgecolor=GRID,
                      labelcolor=TEXT, handlelength=1.5)

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 3-RASM: Band bo'yicha tahlil (barcha tasvirlar o'rtachasi)
# ═══════════════════════════════════════════════════════════════

def fig3_bands(band_res, n_images, out="figure3_bands.png"):
    import matplotlib.pyplot as plt

    sigmas  = [0.02, 0.05, 0.10]
    scols   = ["#4da6ff", "#ffa533", "#ff4444"]
    slabels = ["σ=0.02", "σ=0.05", "σ=0.10"]
    key_map = {
        0.02: "Gaussian (σ=0.02)",
        0.05: "Gaussian (σ=0.05)",
        0.10: "Gaussian (σ=0.10)",
    }
    n_bands = len(band_res[LABELS[0]]["PSNR"])
    x = np.arange(n_bands); bw = 0.26; offs = np.array([-bw, 0, bw])

    band_labels_used = BAND_LABELS[:n_bands]
    titles = {
        "PSNR":       "PSNR — band bo'yicha",
        "SSIM":       "SSIM — band bo'yicha",
        "SAM":        "SAM — band bo'yicha",
        "ERGAS":      "ERGAS — band bo'yicha",
        "SNR":        "SNR — band bo'yicha",
        "BRISQUE":    "BRISQUE — band bo'yicha",
        "Entropy_dH": "Entropy ΔH — band bo'yicha\n(shovqin ↑ entropy, blur ↓ entropy)",
        "NCC":        "NCC — band bo'yicha\n(intensivlik siljishiga sezgir emas)",
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 11.5), facecolor=BG)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.12,
                        hspace=0.52, wspace=0.30)
    fig.suptitle(
        f"Gaussian Shovqin Ta'sirida 8 Metrika — Spektral Band Bo'yicha  "
        f"(n={n_images} ta tasvir o'rtachasi)",
        fontsize=13, fontweight='bold', color=TEXT, y=0.965)
    fig.text(0.5, 0.932,
             "B2=Ko'k(490nm)  B3=Yashil(560nm)  B4=Qizil(665nm)  "
             "B8=NIR(842nm)  B11=SWIR1(1610nm)  B12=SWIR2(2190nm)  ·  "
             "Ko'k=σ=0.02  Sariq=σ=0.05  Qizil=σ=0.10",
             ha='center', fontsize=8, color=MUTED)

    nir_idx = min(3, n_bands - 1)

    for ai, mkey in enumerate(METRICS):
        row, col = divmod(ai, 4)
        ax = axes[row][col]
        ax.set_facecolor(BG2)
        ax.grid(True, axis='y', color=GRID, linewidth=0.7, linestyle='--', alpha=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)

        all_vals = []
        for si, sigma in enumerate(sigmas):
            key = key_map[sigma]
            if key not in band_res or mkey not in band_res[key]:
                continue
            ys = band_res[key][mkey][:n_bands]
            all_vals.extend(ys)
            bars = ax.bar(x + offs[si], ys, width=bw, color=scols[si],
                          alpha=0.88, zorder=3, edgecolor=BG,
                          linewidth=0.5, label=slabels[si])
            for bar, yi in zip(bars, ys):
                fmt = ".3f" if mkey in ("SSIM", "NCC") else \
                      (".3f" if mkey == "Entropy_dH" else ".1f")
                txt = f"{yi:{fmt}}"
                va = 'top' if yi < 0 else 'bottom'
                yo = -0.3 if yi < 0 else 0.003
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yo,
                        txt, ha='center', va=va, fontsize=5.8,
                        color=TEXT, fontweight='bold', zorder=5)

        vmin = min(all_vals) if all_vals else 0
        vmax = max(all_vals) if all_vals else 1
        pad = (vmax - vmin) * 0.20 if vmax > vmin else 0.05
        yl = [vmin - pad, vmax + pad]

        ax.axvspan(nir_idx - 0.45, nir_idx + 0.45,
                   alpha=0.07, color='#ffd166', zorder=0)
        ax.text(nir_idx, yl[1] * 0.97, "NIR\n↑ signal",
                ha='center', va='top', fontsize=6.5, color='#ffd166', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(band_labels_used, fontsize=7.5, color='#4da6ff', ha='center')
        ax.set_xlabel("Spektral band", color='#4da6ff', fontsize=9, labelpad=4)
        ax.set_ylim(yl)
        ax.set_ylabel(METRIC_YLABELS[mkey], color=MUTED, fontsize=9)
        ax.set_title(titles[mkey], color=TEXT, fontsize=9.5, fontweight='bold', pad=6)
        ax.legend(fontsize=7.5, frameon=True, loc='upper right',
                  facecolor='#1e2530', edgecolor=GRID,
                  labelcolor=TEXT, handlelength=1.2, borderpad=0.5)

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 4-RASM: Normalizatsiya qilingan issiqlik xaritasi
# ═══════════════════════════════════════════════════════════════

def fig4_heatmap(results, n_images, out="figure4_heatmap.png"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vals_raw, nm = normalize_matrix(results)
    fmts = {"PSNR": ".1f", "SSIM": ".3f", "SAM": ".2f", "ERGAS": ".1f",
            "SNR": ".1f", "BRISQUE": ".1f", "Entropy_dH": ".3f", "NCC": ".4f"}

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    fig.subplots_adjust(left=0.18, right=0.86, top=0.90, bottom=0.14)
    fig.suptitle(
        f"Normalizatsiya Qilingan Samaradorlik Matritsasi (8 Metrika, n={n_images})",
        fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    fig.text(0.53, 0.93,
             "Yashil=yuqori sifat · Qizil=past sifat · Entropy_dH=ΔH past yaxshi · "
             "Ajratuvchi chiziq: eski 6 | yangi 2",
             ha='center', fontsize=8, color=MUTED)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#c0392b', '#e67e22', '#f1c40f', '#27ae60'], N=256)
    im = ax.imshow(nm, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    for i, n in enumerate(LABELS):
        for j, m in enumerate(METRICS):
            v = vals_raw[i, j]
            tc = 'black' if nm[i, j] > 0.6 else TEXT
            ax.text(j, i, f"{v:{fmts[m]}}", ha='center', va='center',
                    fontsize=7.5, color=tc, fontweight='bold')

    # SAM paradoksi ramkasi
    tm_row = next(i for i, g in enumerate(GROUPS) if g == "Tuz-Murch")
    sam_col = METRICS.index("SAM")
    ax.add_patch(plt.Rectangle((sam_col - 0.5, tm_row - 0.5), 1, 1,
                                linewidth=2.5, edgecolor='#c77dff',
                                facecolor='none', zorder=5))
    ax.text(8.05, tm_row, "⚠ SAM paradoksi", va='center',
            fontsize=8, color='#c77dff', fontweight='bold')

    # Yangi metrikalar ajratuvchi chiziq
    ax.axvline(5.5, color='#ffd166', linewidth=1.5, linestyle='--', alpha=0.6)
    ax.text(6.5, -0.75, "Yangi metrikalar →",
            ha='center', fontsize=8, color='#ffd166', fontweight='bold')

    ax.set_xticks(range(8))
    ax.set_xticklabels(METRICS, color=TEXT, fontsize=9.5, fontweight='bold')
    ax.set_yticks(range(len(LABELS)))
    sl = [l.replace("Gaussian", "G").replace("Kompressiya", "K")
          .replace("Stripe", "Str").replace("Tuz-Murch", "Tuz-M") for l in LABELS]
    ax.set_yticklabels(sl, fontsize=9)
    for ytick, g in zip(ax.get_yticklabels(), GROUPS):
        ytick.set_color(GRP_COLOR[g])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.018)
    cbar.set_label("Normalizatsiya qilingan sifat (1=yaxshi)", color=MUTED, fontsize=8)
    cbar.ax.tick_params(colors=MUTED)

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 5-RASM: Metrika tanlash yo'riqnomasi
# ═══════════════════════════════════════════════════════════════

def fig5_flowchart(out="figure5_flowchart.png"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 10.5), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0, 16); ax.set_ylim(0, 10.5); ax.axis('off')
    fig.suptitle("Metrika Tanlash Yo'riqnomasi — 8 Metrika (+Entropy ΔH, +NCC)",
                 fontsize=13, fontweight='bold', color=TEXT, y=0.97)

    def box(x, y, w, h, text, color, fcolor='#1e2530', fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.1", facecolor=fcolor,
                     edgecolor=color, linewidth=2))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color=TEXT, fontweight='bold' if bold else 'normal',
                multialignment='center')

    def diamond(x, y, w, h, text, color='#ffaa44'):
        xs = [x, x + w / 2, x, x - w / 2, x]
        ys = [y + h / 2, y, y - h / 2, y, y + h / 2]
        ax.fill(xs, ys, facecolor='#1e2530', edgecolor=color, linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                color=TEXT, fontweight='bold', multialignment='center')

    def arrow(x1, y1, x2, y2, label='', color='#4a5568'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label,
                    fontsize=7.5, color=color, fontweight='bold')

    box(8.0, 10.0, 3.5, 0.7, "BOSHLASH\nSifat baholash vazifasi", '#44ff88', bold=True)
    arrow(8.0, 9.65, 8.0, 9.0)
    diamond(8.0, 8.7, 4.0, 0.8, "Etalon (referens)\ntasvir mavjudmi?")
    arrow(8.0, 8.3, 8.0, 7.65)
    ax.text(8.18, 8.0, "Ha", fontsize=8, color='#44ff88', fontweight='bold')
    arrow(10.0, 8.7, 13.5, 8.7, color='#ff6b6b')
    ax.text(11.3, 8.88, "Yo'q", fontsize=8, color='#ff6b6b', fontweight='bold')
    box(13.5, 8.7, 2.8, 0.85, "NR metrikalar:\nSNR + BRISQUE\n+ Entropy ΔH", '#4da6ff')

    diamond(8.0, 7.35, 4.5, 0.8, "Vazifa turi nima?")

    # Spektral
    arrow(5.75, 7.35, 3.2, 7.35, color='#06d6a0')
    ax.text(3.9, 7.52, "Spektral\ntasniflash", fontsize=7.5, color='#06d6a0')
    box(2.2, 7.35, 2.8, 0.8, "SAM (birlamchi)\n+ SSIM + NCC", '#06d6a0')

    # Pan-sharpening
    arrow(8.0, 6.95, 8.0, 6.3)
    ax.text(8.18, 6.65, "Pan-sharp / fusion", fontsize=7.5, color='#ffd166')
    box(8.0, 6.0, 3.1, 0.75, "ERGAS (standart)\n+ SAM + NCC", '#ffd166')

    # Kalibratsiya
    arrow(10.25, 7.35, 12.5, 6.6, color='#4da6ff')
    ax.text(10.8, 7.12, "Kalibratsiya /\natm. korreksiya", fontsize=7.5, color='#4da6ff')
    box(12.8, 6.3, 2.8, 0.75, "NCC (birlamchi)\n+ PSNR + SNR", '#4da6ff')

    # Umumiy
    arrow(5.75, 6.75, 3.8, 5.8, color='#ff6b6b')
    ax.text(4.2, 6.5, "Umumiy sifat", fontsize=7.5, color='#ff6b6b')
    box(3.2, 5.5, 2.8, 0.75, "SSIM + PSNR\n+ SAM + NCC", '#ff6b6b')

    arrow(8.0, 5.62, 8.0, 4.98)
    diamond(8.0, 4.68, 4.5, 0.8, "Buzilish turi ma'lummi?")
    ax.text(8.18, 4.32, "Ha", fontsize=8, color='#44ff88', fontweight='bold')
    arrow(8.0, 4.28, 8.0, 3.65)

    buzz = [
        (1.6,  "Gaussian\nshovqin",    "PSNR + SNR\n+ Entropy ΔH",        '#ff6b6b'),
        (4.3,  "Blur",                 "SSIM + SAM\n+ BRISQUE + Entropy ΔH↓", '#4da6ff'),
        (8.0,  "Kompressiya",          "ERGAS + BRISQUE\n+ NCC",          '#ffd166'),
        (11.7, "Stripe\nshovqin",      "BRISQUE + SSIM\n+ PSNR",          '#06d6a0'),
        (14.4, "Tuz-Murch",            "PSNR + ERGAS + NCC\n⚠ SAM yolg'iz\nishlatma!", '#c77dff'),
    ]
    for bx, blabel, mrec, bc in buzz:
        box(bx, 3.3, 2.6, 0.75, blabel, bc, fs=7.5)
        box(bx, 2.2, 2.6, 0.95, mrec, bc, '#0d1117', fs=7.0)
        arrow(bx, 2.92, bx, 2.67, color=bc)

    ax.text(8.0, 1.28,
            "💡 Entropy ΔH:\n"
            "   Blur → ΔH katta (entropy kamaytiradi)   "
            "Gaussian/Tuz-Murch → ΔH katta (entropy oshiradi)\n"
            "   → Entropy ΔH yolg'iz buzilish turini farqlamaydi; boshqa metrika bilan birga ishlating.",
            ha='center', fontsize=8, color='#4da6ff',
            bbox=dict(facecolor='#0d1a2a', edgecolor='#4da6ff',
                      boxstyle='round,pad=0.4', alpha=0.92))

    ax.text(8.0, 0.32,
            "⚠  Qoida: Kamida 2–3 metrikani birga qo'llang — "
            "bitta metrika asosida qaror qabul qilmang!",
            ha='center', fontsize=9, color='#ffcc00', fontweight='bold',
            bbox=dict(facecolor='#1a1a00', edgecolor='#ffcc00',
                      boxstyle='round,pad=0.4'))

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 6-RASM: Metrikalar divergensiyasi
# ═══════════════════════════════════════════════════════════════

def fig6_divergence(results, n_images, out="figure6_divergence.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    sl = short_labels(); xr = np.arange(len(LABELS))
    vals_raw, nm = normalize_matrix(results)
    ml_colors = ["#4da6ff", "#44ff88", "#ff6b6b", "#ffd166",
                 "#ff9944", "#c77dff", "#00e5ff", "#ffb347"]
    m_markers = ["o", "s", "^", "D", "v", "*", "P", "X"]
    tm_idx = next(i for i, g in enumerate(GROUPS) if g == "Tuz-Murch")

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        f"8 Metrika Divergensiyasi — Korrelyatsiya va Paradoks Tahlili  (n={n_images})",
        fontsize=13, fontweight='bold', color=TEXT, y=0.97)
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.07, right=0.97,
                           top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)

    # Panel 1: Normalizatsiya
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(BG2)
    ax1.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)
    for sp in ax1.spines.values():
        sp.set_edgecolor(GRID)
    for j, (m, n) in enumerate(zip(METRICS, nm.T)):
        ms = 10 if m_markers[j] in ("*", "P", "X") else 6
        ax1.plot(xr, n, color=ml_colors[j], marker=m_markers[j], markersize=ms,
                 linewidth=1.8, label=m, alpha=0.9,
                 markeredgecolor='white', markeredgewidth=0.5)
    ax1.axvspan(tm_idx - 0.5, tm_idx + 0.5, alpha=0.12, color='#c77dff', zorder=0)
    ax1.text(tm_idx, 1.12,
             f"⚠ Paradoks:\nPSNR={results[LABELS[tm_idx]]['PSNR']:.1f} dB\n"
             f"SAM={results[LABELS[tm_idx]]['SAM']:.2f}°\n"
             f"NCC={results[LABELS[tm_idx]]['NCC']:.4f}",
             ha='center', fontsize=7.5, color='#c77dff', fontweight='bold')
    ax1.set_xticks(xr)
    ax1.set_xticklabels(sl, fontsize=7.5, color=MUTED, rotation=40, ha='right')
    ax1.set_ylim(-0.05, 1.25)
    ax1.set_ylabel("Normalizatsiya qilingan sifat", color=MUTED, fontsize=9)
    ax1.set_title("Barcha 8 Metrika — Normalizatsiya Qilingan Ko'rsatkichlar",
                  color=TEXT, fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7.5, frameon=True, loc='upper left', ncol=4,
               facecolor='#1e2530', edgecolor=GRID, labelcolor=TEXT)
    ax1.tick_params(colors=MUTED)

    # Panel 2: PSNR vs SAM scatter
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG2)
    ax2.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID)
    for i, (n, g) in enumerate(zip(LABELS, GROUPS)):
        pv = results[n]["PSNR"]; sv = results[n]["SAM"]
        nv = results[n]["NCC"]
        ms = 130 if g == "Tuz-Murch" else 60
        ax2.scatter(pv, sv, s=ms, color=GRP_COLOR[g],
                    zorder=4, edgecolors='white', linewidths=0.7)
        if g == "Tuz-Murch":
            ax2.annotate(
                f"⚠ PSNR↓ SAM↓\nNCC={nv:.4f}",
                xy=(pv, sv), xytext=(pv - 7, sv + 2.5),
                fontsize=7.5, color='#c77dff', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c77dff', lw=1.0))
    ax2.set_xlabel("PSNR (dB)", color=MUTED, fontsize=9)
    ax2.set_ylabel("SAM (°)", color=MUTED, fontsize=9)
    ax2.set_title("PSNR vs SAM Divergensiyasi", color=TEXT, fontsize=10, fontweight='bold')
    ax2.tick_params(colors=MUTED)

    # Panel 3: Divergensiya bar chart
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(BG2)
    ax3.grid(True, axis='y', color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)
    for sp in ax3.spines.values():
        sp.set_edgecolor(GRID)
    diverg = [nm[i, :].max() - nm[i, :].min() for i in range(len(LABELS))]
    bars = ax3.bar(xr, diverg, color=[GRP_COLOR[g] for g in GROUPS],
                   alpha=0.88, edgecolor=BG, linewidth=0.5)
    for bar, dv, lbl in zip(bars, diverg, LABELS):
        ax3.text(bar.get_x() + bar.get_width() / 2, dv + 0.01,
                 f"{dv:.2f}", ha='center', va='bottom', fontsize=8.5,
                 color=TEXT, fontweight='bold')
        if "Tuz-Murch" in lbl:
            bar.set_edgecolor('#c77dff'); bar.set_linewidth(3)
    ax3.set_xticks(xr)
    ax3.set_xticklabels(sl, fontsize=8, color=MUTED, rotation=40, ha='right')
    ax3.set_ylabel("Maksimal divergensiya\n(8 metrika orasida)", color=MUTED, fontsize=9)
    ax3.set_title(
        "Har Bir Buzilish Turi Uchun Metrikalar Orasidagi Divergensiya (0=mos, 1=maksimal farq)",
        color=TEXT, fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 1.15); ax3.tick_params(colors=MUTED)
    patches = [mpatches.Patch(facecolor=v, label=k) for k, v in GRP_COLOR.items()]
    ax3.legend(handles=patches, fontsize=8, frameon=True, loc='upper left',
               facecolor='#1e2530', edgecolor=GRID, labelcolor=TEXT, ncol=5)

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# 7-RASM: Samaradorlik matritsasi
# ═══════════════════════════════════════════════════════════════

def fig7_effectiveness(results, n_images, out="figure7_effectiveness.png"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vals_raw, nm = normalize_matrix(results)
    eff = np.zeros(nm.shape, dtype=int)
    eff[nm > 0.75] = 3
    eff[(nm > 0.50) & (nm <= 0.75)] = 2
    eff[(nm > 0.25) & (nm <= 0.50)] = 1
    eff[nm <= 0.25] = 0

    # Tuz-Murch SAM paradoksi — majburiy 0
    tm_row = next(i for i, g in enumerate(GROUPS) if g == "Tuz-Murch")
    sam_col = METRICS.index("SAM")
    eff[tm_row, sam_col] = 0

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    fig.subplots_adjust(left=0.19, right=0.88, top=0.88, bottom=0.14)
    fig.suptitle(
        f"Metrikalar Samaradorlik Matritsasi  (8 Metrika, n={n_images}, avtomatik baho)",
        fontsize=12, fontweight='bold', color=TEXT, y=0.97)
    fig.text(0.53, 0.93,
             "3=★ A'lo (NM>0.75)  ·  2=✓ Yaxshi  ·  1=⚠ O'rtacha  ·  0=❌ Yanglishi mumkin  "
             "·  Ballar normalizatsiyalangan qiymatdan avtomatik hisoblangan",
             ha='center', fontsize=8, color=MUTED)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'eff', ['#c0392b', '#e67e22', '#f1c40f', '#27ae60'], N=4)
    im = ax.imshow(eff, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    lbl_map = {0: "❌\n0", 1: "⚠\n1", 2: "✓\n2", 3: "★\n3"}
    for i in range(len(LABELS)):
        for j in range(len(METRICS)):
            v = eff[i, j]
            tc = 'black' if v >= 2 else TEXT
            ax.text(j, i, lbl_map[v], ha='center', va='center',
                    fontsize=9, color=tc, fontweight='bold')

    # SAM paradoksi ramkasi
    ax.add_patch(plt.Rectangle((sam_col - 0.5, tm_row - 0.5), 1, 1,
                                linewidth=3, edgecolor='#c77dff',
                                facecolor='none', zorder=5))
    ax.text(sam_col, tm_row + 0.7, "⚠ SAM paradoksi",
            ha='center', fontsize=7.5, color='#c77dff', fontweight='bold')

    # Yangi metrikalar ajratuvchi
    ax.axvline(5.5, color='#ffd166', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(6.5, -0.8, "Yangi metrikalar",
            ha='center', fontsize=8, color='#ffd166', fontweight='bold')

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS, color=TEXT, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(LABELS)))
    sl = [l.replace("Gaussian", "G").replace("Kompressiya", "K")
          .replace("Stripe", "Str").replace("Tuz-Murch", "Tuz-M") for l in LABELS]
    ax.set_yticklabels(sl, fontsize=9)
    for ytick, g in zip(ax.get_yticklabels(), GROUPS):
        ytick.set_color(GRP_COLOR[g])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.018, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(["0: Yanglishi\nmumkin", "1: O'rtacha", "2: Yaxshi", "3: A'lo"])
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✓ {out}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT Residential: haqiqiy hisob + CSV + 7 rasm (8 metrika)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Misol:
  python generate_all_figures_v3.py \\
      --folder /data/EuroSATallBands/Residential \\
      --n-images 50 --seed 42

  # Faqat CSV (rasmlar yo'q):
  python generate_all_figures_v3.py \\
      --folder /data/EuroSATallBands/Residential --csv-only

MUHIM: --folder ko'rsatilmasa, dastur xato berib to'xtaydi.
       Sintetik data ISHLATILMAYDI.
""")
    parser.add_argument("--folder",    required=True,
                        help="EuroSATallBands/Residential papkasi (MAJBURIY)")
    parser.add_argument("--n-images",  type=int, default=50,
                        help="Nechta tasvir ishlatilsin (default=50)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Tasodifiy tanlov uchun seed (default=42)")
    parser.add_argument("--csv-only",  action="store_true",
                        help="Faqat CSV saqlash, rasmlar chizilmaydi")
    parser.add_argument("--csv-out",   default="metrics_results.csv",
                        help="CSV fayl nomi (default=metrics_results.csv)")
    args = parser.parse_args()

    print("=" * 72)
    print("EuroSAT Residential — 8 Metrika Hisoblash + CSV + 7 Rasm")
    print("Metrikalar: PSNR  SSIM  SAM  ERGAS  SNR  BRISQUE  Entropy_dH  NCC")
    print(f"Dataset:    {args.folder}")
    print(f"Tasvirlar:  {args.n_images} ta (seed={args.seed})")
    print("=" * 72)

    # ── Dataset yuklash ──
    print("\nDataset yuklanmoqda...")
    images = load_eurosat(args.folder, n=args.n_images, seed=args.seed)

    # ── Metrikalar hisoblash ──
    results, band_res, first_img = compute_all_metrics(images, seed=args.seed)
    n_images = len(images)

    # ── CSV saqlash ──
    save_csv(results, band_res, n_images, args.folder, out=args.csv_out)

    if args.csv_only:
        print("\n--csv-only rejim: rasmlar chizilmadi.")
        print("=" * 72)
        return

    # ── Rasmlar ──
    print("\nRasmlar yaratilmoqda...")
    import matplotlib
    matplotlib.use('Agg')

    fig1_distortions(first_img, results)          # faqat birinchi tasvir
    fig2_metrics(results, n_images)               # barcha tasvirlar o'rtachasi
    fig3_bands(band_res, n_images)                # barcha tasvirlar o'rtachasi
    fig4_heatmap(results, n_images)               # barcha tasvirlar o'rtachasi
    fig5_flowchart()                              # statik diagram
    fig6_divergence(results, n_images)            # barcha tasvirlar o'rtachasi
    fig7_effectiveness(results, n_images)         # barcha tasvirlar o'rtachasi

    print("\n" + "=" * 72)
    print("Barcha ishlar muvaffaqiyatli tugadi!")
    print(f"  CSV:     {args.csv_out}")
    print("  Rasmlar: figure1_distortions.png ... figure7_effectiveness.png")
    print(f"\n  n={n_images} ta haqiqiy Sentinel-2 tasvir · seed={args.seed}")
    print("  1-rasm: birinchi tanlangan tasvir")
    print("  2–7-rasm: barcha tasvirlar bo'yicha o'rtacha metrikalar")
    print("=" * 72)


if __name__ == "__main__":
    main()