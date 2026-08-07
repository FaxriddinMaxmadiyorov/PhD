"""
Shovqin darajasini baholash moduli (kanal bo'yicha)
=====================================================

Har bir kanal uchun, shu kanalning shovqin turiga moslashtirilgan
holda, robust shovqin darajasini (sigma*) hisoblaydi.

Turga qarab uch xil strategiya qo'llaniladi:

    gaussian       -> wavelet MAD baholovchisi bevosita
    poisson, mixed -> Anscombe transformatsiyasi, so'ngra MAD, so'ngra
                      delta usuli orqali dastlabki shkalaga qaytarish
    salt_pepper    -> avval median filtr bilan impulslar chetlashtiriladi,
                      so'ngra qolgan fon shovqiniga MAD

Kanallar bo'yicha o'rtachalash qo'llanilmaydi: har bir kanalning
yoritilganligi va o'qish zanjiri boshqacha bo'lgani uchun ularning
shovqin darajalari bir necha barobar farq qilishi mumkin.
"""

import numpy as np
import pywt
from scipy import ndimage

from noise_characterization import flat_region_mask


def wavelet_mad_sigma(band: np.ndarray, wavelet: str = "db8") -> float:
    """
    Donoho-Johnstone MAD baholovchisi: bir darajali 2D DWT ning
    diagonal (HH) qism polosasi koeffitsiyentlari orqali.
    """
    _, (_, _, cd) = pywt.dwt2(band, wavelet)
    return float(np.median(np.abs(cd)) / 0.6745)


def anscombe_transform(band: np.ndarray) -> np.ndarray:
    """Puasson shovqinining signalga bog'liq dispersiyasini barqarorlashtiradi."""
    return 2.0 * np.sqrt(np.clip(band, 0.0, None) + 3.0 / 8.0)


def _poisson_sigma_original_scale(sigma_stabilized: float, local_mean: float) -> float:
    """
    Anscombe fazosidagi bahoni dastlabki shkalaga qaytaradi (delta usuli):
        f(x) = 2*sqrt(x + 3/8),  f'(x) = 1/sqrt(x + 3/8)
        =>  sigma_x ~= sigma_f * sqrt(x + 3/8)
    """
    return sigma_stabilized * np.sqrt(max(local_mean, 0.0) + 3.0 / 8.0)


def _repair_impulses(band: np.ndarray, k: float = 3.0, size: int = 3) -> np.ndarray:
    """
    Faqat impulsiv deb aniqlangan piksellarni median qiymat bilan
    almashtiradi, qolgan piksellarga tegmaydi.

    Butun tasvirni median filtrdan o'tkazish noto'g'ri bo'lar edi: median
    filtr impulslar bilan birga fon shovqinini ham sezilarli darajada
    silliqlaydi va keyingi MAD bahosini bir necha barobar past chiqaradi.
    """
    median = ndimage.median_filter(band, size=size)
    residual = band - median
    mad = np.median(np.abs(residual - np.median(residual)))
    robust_sigma = 1.4826 * mad

    if robust_sigma <= 0:
        return band.copy()

    impulse_mask = np.abs(residual) > k * robust_sigma
    repaired = band.copy()
    repaired[impulse_mask] = median[impulse_mask]
    return repaired


def estimate_band_sigma(band: np.ndarray, noise_type: str,
                         wavelet: str = "db8",
                         flat_percentile: float = 60.0) -> float:
    """Bitta kanal uchun sigma* ni turga moslashtirilgan holda hisoblaydi."""
    if noise_type == "salt_pepper":
        # Impulslar chetlashtirilgach, qolgan fon shovqini baholanadi.
        # Fon shovqini odatda Puasson va Gauss aralashmasi bo'lgani uchun
        # Anscombe yo'li orqali hisoblanadi.
        repaired = _repair_impulses(band)
        sigma_stabilized = wavelet_mad_sigma(anscombe_transform(repaired), wavelet=wavelet)
        mask = flat_region_mask(repaired, flat_percentile=flat_percentile)
        local_mean = float(np.mean(repaired[mask])) if mask.any() else float(np.mean(repaired))
        return _poisson_sigma_original_scale(sigma_stabilized, local_mean)

    if noise_type in ("poisson", "mixed"):
        sigma_stabilized = wavelet_mad_sigma(anscombe_transform(band), wavelet=wavelet)
        mask = flat_region_mask(band, flat_percentile=flat_percentile)
        local_mean = float(np.mean(band[mask])) if mask.any() else float(np.mean(band))
        return _poisson_sigma_original_scale(sigma_stabilized, local_mean)

    return wavelet_mad_sigma(band, wavelet=wavelet)


def estimate_noise_level_per_band(image: np.ndarray, noise_types,
                                   wavelet: str = "db8",
                                   flat_percentile: float = 60.0) -> np.ndarray:
    """
    Har bir kanal uchun sigma* massivini qaytaradi.

    image       : (H, W) yoki (H, W, C)
    noise_types : uzunligi C ga teng turlar ro'yxati, yoki bitta satr
                  (u holda barcha kanallarga bir xil tur qo'llaniladi)
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]

    n_bands = image.shape[2]
    if isinstance(noise_types, str):
        noise_types = [noise_types] * n_bands

    return np.array([
        estimate_band_sigma(image[:, :, i], noise_types[i],
                            wavelet=wavelet, flat_percentile=flat_percentile)
        for i in range(n_bands)
    ])


def estimate_noise_level(image: np.ndarray, noise_type, **kwargs) -> float:
    """
    Kanallar bo'yicha o'rtachalangan yagona sigma*.
    Faqat taqqoslash tajribalarida ishlatiladi.
    """
    return float(np.mean(estimate_noise_level_per_band(image, noise_type, **kwargs)))


if __name__ == "__main__":
    from multispectral_data import make_multispectral_scene, apply_multispectral_noise

    rng = np.random.default_rng(4)
    clean = make_multispectral_scene(size=128, n_bands=6, seed=21)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)

    types = [t["type_true"] for t in truth]
    sigmas = estimate_noise_level_per_band(noisy, types)
    averaged = float(np.mean(sigmas))

    print("Kanal bo'yicha sigma* baholash (haqiqiy turlar bilan)")
    print("-" * 74)
    print(f"{'kanal':>6}{'turi':>14}{'sigma_haqiqiy':>16}{'sigma*':>12}{'nisbiy xato':>16}")
    for i, t in enumerate(truth):
        err = abs(sigmas[i] - t["sigma_true"]) / t["sigma_true"] * 100
        print(f"{i:>6}{t['type_true']:>14}{t['sigma_true']:>16.5f}"
              f"{sigmas[i]:>12.5f}{err:>15.1f}%")

    print(f"\nKanal bo'yicha sigma* tarqalishi: "
          f"{sigmas.min():.5f} ... {sigmas.max():.5f} "
          f"({sigmas.max() / max(sigmas.min(), 1e-9):.1f} barobar farq)")
    print(f"Kanallar bo'yicha o'rtachalangan yagona qiymat: {averaged:.5f}")
    print("Yagona qiymat eng past va eng yuqori kanallarning ikkalasiga ham mos kelmaydi.")