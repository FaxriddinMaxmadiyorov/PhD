"""
Moslashuvchan shovqinsizlantirish moduli (kanal bo'yicha)
===========================================================

Har bir kanalga o'z shovqin turiga mos usul va o'z sigma* qiymatiga
sozlangan parametr bilan ishlov beradi.

Tur -> usul mosligi:
    gaussian     -> Wavelet (BayesShrink)
    poisson      -> Non-Local Means
    mixed        -> BM3D
    salt_pepper  -> impulslarni tiklash, so'ngra fon shovqini uchun BM3D

Impulsiv kanalda ikki bosqichli ishlov zarur: median almashtirish faqat
impulslarni yo'q qiladi, fon shovqini esa saqlanib qoladi va uni alohida
bostirish kerak. Bir bosqichli median filtrlash esa impulslarni yo'qotsa
ham, fon shovqinini to'liq bostirmaydi va ayni vaqtda detalni ortiqcha
silliqlaydi.
"""

import numpy as np
import bm3d
from skimage.restoration import denoise_wavelet, denoise_nl_means

from noise_level_estimation import _repair_impulses, wavelet_mad_sigma

DENOISER_BY_TYPE = {
    "gaussian": "wavelet",
    "poisson": "nlm",
    "mixed": "bm3d",
    "salt_pepper": "impulse+bm3d",
}


def select_denoiser(noise_type: str) -> str:
    if noise_type not in DENOISER_BY_TYPE:
        raise ValueError(f"noma'lum shovqin turi: {noise_type}")
    return DENOISER_BY_TYPE[noise_type]


def denoise_band(band: np.ndarray, noise_type: str, sigma: float) -> np.ndarray:
    """Bitta kanalni uning turiga mos usul bilan tozalaydi."""
    sigma = float(max(sigma, 1e-4))

    if noise_type == "gaussian":
        out = denoise_wavelet(band, sigma=sigma, mode="soft",
                               method="BayesShrink", rescale_sigma=True)

    elif noise_type == "poisson":
        out = denoise_nl_means(band, h=1.15 * sigma, sigma=sigma,
                                patch_size=5, patch_distance=6, fast_mode=True)

    elif noise_type == "mixed":
        out = bm3d.bm3d(band, sigma_psd=sigma)

    elif noise_type == "salt_pepper":
        repaired = _repair_impulses(band)
        out = bm3d.bm3d(repaired, sigma_psd=sigma)

    else:
        raise ValueError(f"noma'lum shovqin turi: {noise_type}")

    return np.clip(out, 0.0, 1.0)


def adaptive_denoise_per_band(image: np.ndarray, noise_types, sigmas) -> np.ndarray:
    """
    Har bir kanalni o'z turi va o'z sigma qiymati bilan tozalaydi.

    image       : (H, W) yoki (H, W, C)
    noise_types : uzunligi C ga teng ro'yxat yoki bitta satr
    sigmas      : uzunligi C ga teng massiv yoki bitta son
    """
    image = np.asarray(image, dtype=np.float64)
    squeeze = image.ndim == 2
    if squeeze:
        image = image[:, :, None]

    n_bands = image.shape[2]
    if isinstance(noise_types, str):
        noise_types = [noise_types] * n_bands
    sigmas = np.broadcast_to(np.asarray(sigmas, dtype=np.float64).reshape(-1), (n_bands,))

    out = np.stack([
        denoise_band(image[:, :, i], noise_types[i], sigmas[i])
        for i in range(n_bands)
    ], axis=-1)

    return out[:, :, 0] if squeeze else out


# Eski interfeys bilan moslik (taqqoslash tajribalarida ishlatiladi)
def adaptive_denoise(image, noise_type, sigma):
    return adaptive_denoise_per_band(image, noise_type, sigma)


def hybrid_denoise_per_band(image: np.ndarray, noise_types) -> np.ndarray:
    """
    GIBRID STRATEGIYA - tavsiya etiladigan asosiy variant.

    Shovqin turini tasniflash natijasi faqat BITTA qaror uchun
    ishlatiladi: kanalda impulsiv buzilish bormi yoki yo'q. Impulsiv deb
    tasniflangan kanalda avval impulslar median asosida tuzatiladi,
    so'ngra barcha kanallarga bir xilda BM3D qo'llaniladi; har bir kanal
    uchun sigma o'sha kanalning o'z MAD bahosidan olinadi.

    Nima uchun aynan shunday
    ------------------------
    Kanal darajasidagi qiyosiy tajriba shuni ko'rsatdiki, to'liq
    tur-marshrutlash (har bir turga alohida usul) foydani faqat impulsiv
    kanallarda beradi: u yerda ustunlik +12,3 dB va barcha holatlarda
    musbat. Impulsiv bo'lmagan kanallarda esa u universal BM3D dan
    o'rtacha 1,0 dB YOMONROQ ishlaydi va holatlarning atigi 27 foizida
    ustun keladi. Sababi shundaki, Gauss, Puasson va aralash rejimlar
    uzluksiz spektr hosil qiladi va BM3D ularning barchasida deyarli
    maqbul natija beradi; bu spektr ichida usulni almashtirish yutuq
    keltirmaydi, tasniflash xatosi esa zarar keltiradi.

    Impulsiv buzilish esa sifat jihatidan boshqacha: u chegaralashga
    asoslangan usullar bilan yo'qotilmaydi va uni alohida aniqlash
    haqiqatan zarur. Gibrid variant shu ikki kuzatuvni birlashtiradi.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]
    if isinstance(noise_types, str):
        noise_types = [noise_types] * image.shape[2]

    out = np.empty_like(image)
    for i in range(image.shape[2]):
        band = image[:, :, i]
        if noise_types[i] == "salt_pepper":
            band = _repair_impulses(band)
        sigma = wavelet_mad_sigma(band)
        out[:, :, i] = np.clip(bm3d.bm3d(band, sigma_psd=max(sigma, 1e-4)), 0, 1)

    return out if image.shape[2] > 1 else out[:, :, 0]


if __name__ == "__main__":
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from multispectral_data import make_multispectral_scene, apply_multispectral_noise
    from noise_level_estimation import estimate_noise_level_per_band

    rng = np.random.default_rng(6)
    clean = make_multispectral_scene(size=128, n_bands=6, seed=31)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)

    types = [t["type_true"] for t in truth]
    sigmas = estimate_noise_level_per_band(noisy, types)
    denoised = adaptive_denoise_per_band(noisy, types, sigmas)
    hybrid = hybrid_denoise_per_band(noisy, types)

    print("Kanal bo'yicha shovqinsizlantirish (haqiqiy turlar bilan)")
    print("-" * 78)
    print(f"{'kanal':>6}{'turi':>14}{'usul':>15}{'PSNR oldin':>13}"
          f"{'PSNR keyin':>13}{'yutuq':>10}")
    for i, t in enumerate(truth):
        before = psnr(clean[:, :, i], noisy[:, :, i], data_range=1.0)
        after = psnr(clean[:, :, i], denoised[:, :, i], data_range=1.0)
        print(f"{i:>6}{t['type_true']:>14}{select_denoiser(t['type_true']):>15}"
              f"{before:>13.2f}{after:>13.2f}{after - before:>+10.2f}")

    print(f"\nButun tasvir (tur-marshrutlash): {psnr(clean, noisy, data_range=1.0):.2f} -> "
          f"{psnr(clean, denoised, data_range=1.0):.2f} dB")
    print(f"Butun tasvir (gibrid):           {psnr(clean, noisy, data_range=1.0):.2f} -> "
          f"{psnr(clean, hybrid, data_range=1.0):.2f} dB")