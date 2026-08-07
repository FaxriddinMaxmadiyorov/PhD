"""
Sifatni baholash moduli (kanal bo'yicha)
==========================================

Tozalangan tasvirning har bir kanali uchun beshta referenssiz
ko'rsatkichni hisoblaydi va ularni moslashuvchan sifat indeksiga (AQI)
birlashtiradi:

    NIQE tipidagi tabiiylik ko'rsatkichi
    BRISQUE tipidagi tabiiylik ko'rsatkichi
    gistogramma entropiyasi
    o'rtacha gradient kattaligi
    qoldiq shovqin darajasi sigma*

NIQE va BRISQUE haqida izoh
----------------------------
Asl NIQE va BRISQUE inson bahosi bilan o'rgatilgan modelga tayanadi.
Bu yerda ularning matematik yadrosi - MSCN koeffitsiyentlari va
ularning umumlashgan Gauss taqsimoti parametrlari - qo'llanilgan,
pretrained model o'rniga esa buzilmagan sintetik sahnalardan
hisoblangan statistikadan masofa ishlatilgan. Bu pilot bosqich uchun
soddalashtirilgan variant bo'lib, keyingi ishda haqiqiy buzilmagan
tasvirlar korpusi bilan almashtirilishi kerak.
"""

import numpy as np
from scipy import ndimage
from scipy.special import gamma as gamma_fn

from noise_characterization import gradient_magnitude
from noise_level_estimation import estimate_noise_level_per_band

_GAM_RANGE = np.arange(0.2, 10.0, 0.001)
_R_GAM = (gamma_fn(2.0 / _GAM_RANGE) ** 2) / (
    gamma_fn(1.0 / _GAM_RANGE) * gamma_fn(3.0 / _GAM_RANGE)
)


def histogram_entropy(band: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropiyasi, sobit sondagi bin bo'yicha gistogrammadan.

    Eslatma: skimage.measure.shannon_entropy uzluksiz (float) tasvirlarda
    har bir piksel qiymatini alohida belgi deb hisoblab, deyarli doim
    log2(piksellar_soni) ga yaqin, ya'ni foydasiz natija beradi.
    """
    hist, _ = np.histogram(band, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _local_mean_std(band: np.ndarray, sigma: float = 7.0 / 6.0):
    mu = ndimage.gaussian_filter(band, sigma)
    mu_sq = ndimage.gaussian_filter(band ** 2, sigma)
    return mu, np.sqrt(np.clip(mu_sq - mu ** 2, 0.0, None))


def mscn_coefficients(band: np.ndarray, c: float = 1e-3) -> np.ndarray:
    """Mahalliy kontrast bo'yicha normallashtirilgan koeffitsiyentlar."""
    mu, sigma_local = _local_mean_std(band)
    return (band - mu) / (sigma_local + c)


def _ggd_shape_and_variance(coeffs: np.ndarray):
    """Umumlashgan Gauss taqsimotining shakl parametri va dispersiyasi."""
    coeffs = coeffs.ravel()
    variance = float(np.mean(coeffs ** 2))
    mean_abs = float(np.mean(np.abs(coeffs)))
    if mean_abs < 1e-8:
        return 2.0, variance
    rho = variance / (mean_abs ** 2)
    return float(_GAM_RANGE[int(np.argmin(np.abs(rho - _R_GAM)))]), variance


def _pairwise_products(coeffs: np.ndarray):
    return (
        coeffs[:, :-1] * coeffs[:, 1:],
        coeffs[:-1, :] * coeffs[1:, :],
        coeffs[:-1, :-1] * coeffs[1:, 1:],
        coeffs[1:, :-1] * coeffs[:-1, 1:],
    )


def extract_naturalness_features(band: np.ndarray) -> np.ndarray:
    """10 o'lchamli tabiiylik vektori: MSCN va to'rt yo'nalishdagi ko'paytmalar."""
    coeffs = mscn_coefficients(band)
    alpha0, var0 = _ggd_shape_and_variance(coeffs)
    features = [alpha0, var0]
    for product in _pairwise_products(coeffs):
        alpha, var = _ggd_shape_and_variance(product)
        features.extend([alpha, var])
    return np.array(features)


def _patchify(band: np.ndarray, patch_size: int = 32):
    h, w = band.shape
    patches = [
        band[i:i + patch_size, j:j + patch_size]
        for i in range(0, h - patch_size + 1, patch_size)
        for j in range(0, w - patch_size + 1, patch_size)
    ]
    return patches if patches else [band]


def build_natural_reference(scene_fn, n_scenes: int = 20, size: int = 96,
                             n_bands: int = 6, patch_size: int = 32):
    """
    Buzilmagan sahnalar patch'laridan tabiiylik referens statistikasini
    hisoblaydi va normallashtirish chegaralarini kalibrlaydi.

    Patch darajasida ishlash muhim: butun sahnalar bo'yicha statistika
    olinsa, namunalar soni kam va bir-biriga o'xshash bo'lib, standart
    og'ish sun'iy ravishda kichik chiqadi va keyingi z-ball hisoblari
    beqaror bo'ladi.

    scene_fn(size, n_bands, seed) imzosiga ega bo'lishi kerak.
    """
    all_features = []
    for s in range(n_scenes):
        scene = scene_fn(size, n_bands, seed=s)
        for band_idx in range(scene.shape[2]):
            band = scene[:, :, band_idx]
            for patch in _patchify(band, patch_size):
                all_features.append(extract_naturalness_features(patch))

    features = np.array(all_features)
    ref_mean = features.mean(axis=0)
    ref_std = features.std(axis=0) + 1e-3

    z = (features - ref_mean) / ref_std
    rms_z = np.sqrt(np.mean(z ** 2, axis=1))
    abs_z = np.mean(np.abs(z), axis=1)

    # Buzilmagan patch'larning o'zi referensdan qanday tarqalishini
    # o'lchab, 95-protsentilning uch baravari "qabul qilinadigan
    # buzilish" chegarasi sifatida olinadi. Bu chegaralarni qo'lda
    # tanlash o'rniga ma'lumotdan kalibrlab beradi.
    niqe_max = float(np.percentile(rms_z, 95) * 3.0)
    brisque_max = float(np.percentile(abs_z, 95) * 3.0)

    ref_entropy = float(np.mean([
        histogram_entropy(scene_fn(size, n_bands, seed=s)[:, :, b])
        for s in range(min(n_scenes, 10)) for b in range(n_bands)
    ]))
    ref_gradient = float(np.mean([
        np.mean(gradient_magnitude(scene_fn(size, n_bands, seed=s)[:, :, b]))
        for s in range(min(n_scenes, 10)) for b in range(n_bands)
    ]))

    return {
        "ref_mean": ref_mean, "ref_std": ref_std,
        "niqe_max": niqe_max, "brisque_max": brisque_max,
        "ref_entropy": ref_entropy, "ref_gradient": ref_gradient,
    }


def niqe_style_score(band, ref_mean, ref_std, patch_size: int = 32) -> float:
    """Patch'lar bo'yicha o'rtachalangan RMS z-masofa."""
    return float(np.mean([
        np.sqrt(np.mean(((extract_naturalness_features(p) - ref_mean) / ref_std) ** 2))
        for p in _patchify(band, patch_size)
    ]))


def brisque_style_score(band, ref_mean, ref_std, patch_size: int = 32) -> float:
    """
    Patch'lar bo'yicha o'rtachalangan mutlaq z-masofa (chegaralanmagan).

    Ball ataylab sobit oraliqqa qisilmaydi: kalibrlangan chegaradan
    yuqori sun'iy chegara qo'yilsa, ko'pchilik tasvirlarda ball to'yinib
    qoladi va AQI ichidagi tegishli komponent doim nolga aylanadi, ya'ni
    beshta ko'rsatkichdan biri amalda ishlamay qoladi.
    """
    return float(np.mean([
        np.mean(np.abs((extract_naturalness_features(p) - ref_mean) / ref_std))
        for p in _patchify(band, patch_size)
    ]))


def compute_metrics_per_band(image: np.ndarray, noise_types, reference: dict) -> dict:
    """Har bir kanal uchun beshta ko'rsatkichni hisoblaydi."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]

    rm, rs = reference["ref_mean"], reference["ref_std"]
    n_bands = image.shape[2]

    niqe, brisque, entropy, gradient = [], [], [], []
    for i in range(n_bands):
        band = image[:, :, i]
        niqe.append(niqe_style_score(band, rm, rs))
        brisque.append(brisque_style_score(band, rm, rs))
        entropy.append(histogram_entropy(band))
        gradient.append(float(np.mean(gradient_magnitude(band))))

    sigma = estimate_noise_level_per_band(image, noise_types)

    return {
        "niqe": np.array(niqe), "brisque": np.array(brisque),
        "entropy": np.array(entropy), "gradient": np.array(gradient),
        "sigma_star": sigma,
    }


def compute_aqi_per_band(metrics: dict, reference: dict,
                          sigma_max: float = 0.12) -> np.ndarray:
    """
    Har bir kanal uchun AQI (0..1, katta qiymat yaxshi sifat).

    Buzilish ko'rsatkichlari (niqe, brisque, sigma*) uchun to'g'ridan-
    to'g'ri normallashtirish, entropiya va gradient uchun esa buzilmagan
    sahnaning qiymatidan nisbiy chetlanish jarimasi qo'llaniladi. Bu
    ajratish zarur, chunki entropiya va gradient shovqin bilan ham,
    haqiqiy tekstura bilan ham bir yo'nalishda o'sadi va ularni bevosita
    sifat mezoni sifatida ishlatish ortiqcha silliqlashtirilgan tasvirni
    xato ravishda yuqori baholashga olib keladi.
    """
    q_niqe = 1.0 - np.clip(metrics["niqe"] / reference["niqe_max"], 0, 1)
    q_brisque = 1.0 - np.clip(metrics["brisque"] / reference["brisque_max"], 0, 1)
    q_sigma = 1.0 - np.clip(metrics["sigma_star"] / sigma_max, 0, 1)

    e_dev = np.abs(metrics["entropy"] - reference["ref_entropy"]) / max(reference["ref_entropy"], 1e-6)
    g_dev = np.abs(metrics["gradient"] - reference["ref_gradient"]) / max(reference["ref_gradient"], 1e-6)
    q_entropy = 1.0 - np.clip(e_dev, 0, 1)
    q_gradient = 1.0 - np.clip(g_dev, 0, 1)

    return np.mean(np.vstack([q_niqe, q_brisque, q_sigma, q_entropy, q_gradient]), axis=0)


if __name__ == "__main__":
    from multispectral_data import make_multispectral_scene, apply_multispectral_noise
    from adaptive_denoising import adaptive_denoise_per_band

    print("Tabiiylik referensi hisoblanmoqda...")
    reference = build_natural_reference(make_multispectral_scene, n_scenes=15,
                                         size=96, n_bands=6)
    print(f"Kalibrlangan chegaralar: niqe_max={reference['niqe_max']:.2f}, "
          f"brisque_max={reference['brisque_max']:.2f}")

    rng = np.random.default_rng(8)
    clean = make_multispectral_scene(128, 6, seed=41)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)
    types = [t["type_true"] for t in truth]

    from noise_level_estimation import estimate_noise_level_per_band
    sigmas = estimate_noise_level_per_band(noisy, types)
    denoised = adaptive_denoise_per_band(noisy, types, sigmas)

    m_noisy = compute_metrics_per_band(noisy, types, reference)
    m_clean = compute_metrics_per_band(denoised, types, reference)
    aqi_noisy = compute_aqi_per_band(m_noisy, reference)
    aqi_clean = compute_aqi_per_band(m_clean, reference)

    print(f"\n{'kanal':>6}{'AQI shovqinli':>16}{'AQI tozalangan':>17}"
          f"{'NIQE oldin':>13}{'NIQE keyin':>13}")
    for i in range(len(types)):
        print(f"{i:>6}{aqi_noisy[i]:>16.3f}{aqi_clean[i]:>17.3f}"
              f"{m_noisy['niqe'][i]:>13.2f}{m_clean['niqe'][i]:>13.2f}")