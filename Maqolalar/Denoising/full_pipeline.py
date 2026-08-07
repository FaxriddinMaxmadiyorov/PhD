"""
To'liq pipeline: kanal darajasida moslashuvchan shovqinsizlantirish
====================================================================

Ushbu fayl tizimning yakuniy ko'rinishini amalga oshiradi. Uning
tuzilishi dastlabki loyihadan sezilarli farq qiladi, chunki ayrim
modullar tajriba natijalariga ko'ra soddalashtirildi yoki o'chirildi.
Quyida har bir qaror va uning asosi keltirilgan.

Yakuniy oqim
------------
    Multispektrli tenzor (H, W, C)
        -> har bir kanal uchun xususiyat vektori [a, b, impuls, kurtoz]
        -> k-NN tasniflash (kanal darajasida)
        -> strategiya:
             "hybrid"  (standart) - tasniflash faqat impulsiv kanalni
                                    ajratish uchun; impulsiv kanalda
                                    impulslar tuzatiladi, so'ngra
                                    BARCHA kanalga bir xilda BM3D
             "routing"            - har bir turga alohida usul
                                    (qiyosiy tajribalar uchun)
        -> sifat ko'rsatkichlari (tashxis uchun)
        -> tozalangan tenzor + kanal bo'yicha hisobot

Dizayn qarorlari va ularning asosi
-----------------------------------
1. Kanal bo'yicha alohida qayta ishlash SAQLANDI.
   Xususiyatlarni kanallar bo'yicha o'rtachalash 9,1-11,7 dB yo'qotishga
   olib keladi, chunki tanlangan yagona tur kanallarning atigi 36-42
   foiziga to'g'ri keladi.

2. To'rt yo'nalishli tur-marshrutlash STANDART BO'LMAY QOLDI.
   Kanal darajasidagi tahlil ustunlik faqat impulsiv kanallardan kelib
   chiqishini ko'rsatdi (+12,3 dB, 100% holatda musbat). Impulsiv
   bo'lmagan kanallarda marshrutlash universal BM3D dan o'rtacha 1,0 dB
   YOMONROQ ishlaydi. Sababi: Gauss, Puasson va aralash rejimlar uzluksiz
   spektr hosil qiladi va BM3D ularning barchasida deyarli maqbul.

3. GIBRID strategiya standart qilib olindi.
   Uchala shovqin darajasida ham eng yuqori natija:
       daraja 0,3: BM3D 30,55 | marshrutlash 37,44 | gibrid 38,49 dB
       daraja 0,6: BM3D 32,50 | marshrutlash 36,26 | gibrid 37,28 dB
       daraja 0,9: BM3D 33,09 | marshrutlash 33,75 | gibrid 34,94 dB

4. Kanallararo ierarxik regulyarizatsiya O'CHIRILDI.
   Halol taqqoslashda aniqlikni monoton pasaytiradi: 71,4% -> 65,9% ->
   60,3%. U aniqlanishi kerak bo'lgan kanallararo farqni bostiradi.

5. Yopiq tsiklli qayta sozlash O'CHIRILDI, faqat TASHXIS sifatida qoldi.
   Hisoblash hajmini qariyb ikki barobar oshirgan holda sifatni atigi
   0,06 dB ga o'zgartiradi (+14,10 -> +14,16 dB). Sifat ko'rsatkichlari
   hisoblanadi va past sifatli kanallar belgilanadi, ammo avtomatik
   qayta ishlash amalga oshirilmaydi.

MUHIM CHEKLOV (haqiqiy ma'lumot)
---------------------------------
Impulsivlik detektori haqiqiy Sentinel-2 L1C mahsulotlariga o'tmaydi:
30 ta EuroSAT patchida o'rtacha impuls ulushi 0,105 chiqdi, bu sintetik
haqiqatan impulsiv kanallardagi qiymatdan (0,082) ham yuqori. Sabab
shundaki, binolar, yo'llar va dala chegaralari median qoldig'ida katta
chetlashuv hosil qiladi. Yakkalik sharti (impulse_fraction_isolated)
yolg'on ijobiylikni 0,038 gacha kamaytiradi, ammo bartaraf etmaydi.

Shu sababli:
  - standart L1C mahsulotlari uchun strategy="bm3d" tavsiya etiladi;
  - "hybrid" impulsiv buzilish haqiqatan kutiladigan ma'lumotlarga
    (xom mahsulot, eski arxivlar, uzatish xatoliklari) mo'ljallangan.
"""

import numpy as np

from noise_type_classification import features_from_image
from noise_level_estimation import estimate_noise_level_per_band, wavelet_mad_sigma
from adaptive_denoising import (
    adaptive_denoise_per_band, hybrid_denoise_per_band, select_denoiser,
)
from quality_feedback import compute_metrics_per_band, compute_aqi_per_band

import bm3d


def _denoise_bm3d_only(cube: np.ndarray) -> np.ndarray:
    """Bazaviy variant: barcha kanalga bir xilda BM3D, sigma o'z MAD bahosidan."""
    out = np.empty_like(cube)
    for i in range(cube.shape[2]):
        band = cube[:, :, i]
        sigma = wavelet_mad_sigma(band)
        out[:, :, i] = np.clip(bm3d.bm3d(band, sigma_psd=max(sigma, 1e-4)), 0, 1)
    return out


def run_pipeline(noisy_cube: np.ndarray, classifier, reference: dict = None,
                  strategy: str = "hybrid",
                  use_isolated_impulse: bool = False,
                  aqi_threshold: float = 0.75) -> dict:
    """
    Tizimning asosiy kirish nuqtasi.

    noisy_cube : (H, W, C) yoki (H, W) - qiymatlar [0, 1] oralig'ida
    classifier : o'rgatilgan NoiseTypeClassifier
    reference  : build_natural_reference() natijasi. None bo'lsa sifat
                 ko'rsatkichlari hisoblanmaydi.
    strategy   : "hybrid"  - tavsiya etiladigan standart variant
                 "routing" - to'rt yo'nalishli tur-marshrutlash
                 "bm3d"    - tasniflashsiz, barcha kanalga BM3D
    use_isolated_impulse : True bo'lsa, impulsivlik belgisi yakkalik
                 sharti bilan hisoblanadi. Haqiqiy tasvirlar uchun
                 tavsiya etiladi, sintetik ma'lumotda esa zarur emas.

    Qaytaradi: {"denoised", "band_types", "band_sigmas", "band_denoisers",
                "metrics", "aqi", "low_quality_bands", "strategy"}
    """
    cube = np.asarray(noisy_cube, dtype=np.float64)
    squeeze = cube.ndim == 2
    if squeeze:
        cube = cube[:, :, None]

    n_bands = cube.shape[2]

    # --- 1-2. Xarakterlash va kanal darajasida tasniflash ---
    if strategy == "bm3d":
        band_types = ["n/a"] * n_bands
        band_sigmas = np.array([wavelet_mad_sigma(cube[:, :, i]) for i in range(n_bands)])
    else:
        features = features_from_image(cube, isolated_impulse=use_isolated_impulse)
        band_types = list(classifier.predict_bands(features))
        band_sigmas = estimate_noise_level_per_band(cube, band_types)

    # --- 3. Shovqinsizlantirish ---
    if strategy == "hybrid":
        denoised = hybrid_denoise_per_band(cube, band_types)
        band_denoisers = ["median+bm3d" if t == "salt_pepper" else "bm3d"
                          for t in band_types]
    elif strategy == "routing":
        denoised = adaptive_denoise_per_band(cube, band_types, band_sigmas)
        band_denoisers = [select_denoiser(t) for t in band_types]
    elif strategy == "bm3d":
        denoised = _denoise_bm3d_only(cube)
        band_denoisers = ["bm3d"] * n_bands
    else:
        raise ValueError(f"noma'lum strategiya: {strategy}")

    # --- 4. Sifat ko'rsatkichlari (faqat tashxis uchun) ---
    metrics = aqi = None
    low_quality = []
    if reference is not None:
        types_for_metrics = band_types if strategy != "bm3d" else ["gaussian"] * n_bands
        metrics = compute_metrics_per_band(denoised, types_for_metrics, reference)
        aqi = compute_aqi_per_band(metrics, reference)
        low_quality = [i for i, v in enumerate(np.atleast_1d(aqi)) if v < aqi_threshold]

    if squeeze:
        denoised = denoised[:, :, 0]

    return {
        "denoised": denoised,
        "band_types": band_types,
        "band_sigmas": band_sigmas,
        "band_denoisers": band_denoisers,
        "metrics": metrics,
        "aqi": aqi,
        "low_quality_bands": low_quality,
        "strategy": strategy,
    }


if __name__ == "__main__":
    import pickle, os
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from multispectral_data import make_multispectral_scene, apply_multispectral_noise
    from noise_type_classification import NoiseTypeClassifier, build_calibration_set

    if os.path.exists("clf.pkl"):
        classifier = pickle.load(open("clf.pkl", "rb"))
    else:
        print("Klassifikator o'rgatilmoqda...")
        x, y = build_calibration_set(n_scenes=60, size=96, n_bands=6)
        classifier = NoiseTypeClassifier(k=21).fit(x, y)

    def ssim_mb(a, b):
        return float(np.mean([ssim(a[:, :, i], b[:, :, i], data_range=1.0)
                              for i in range(a.shape[2])]))

    rng = np.random.default_rng(21)
    clean = make_multispectral_scene(96, 6, seed=9001)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)

    print("Uchta strategiyani solishtirish")
    print("-" * 62)
    print(f"{'strategiya':<12}{'PSNR, dB':>12}{'SSIM':>10}{'impulsiv kanallar':>20}")
    for strat in ["bm3d", "routing", "hybrid"]:
        res = run_pipeline(noisy, classifier, strategy=strat)
        n_imp = sum(1 for t in res["band_types"] if t == "salt_pepper")
        print(f"{strat:<12}{psnr(clean, res['denoised'], data_range=1.0):>12.2f}"
              f"{ssim_mb(clean, res['denoised']):>10.3f}{n_imp:>20}")

    print(f"\nShovqinli tenzor: {psnr(clean, noisy, data_range=1.0):.2f} dB")
    res = run_pipeline(noisy, classifier, strategy="hybrid")
    print("\nKanal bo'yicha hisobot (gibrid):")
    print(f"{'kanal':>6}{'haqiqiy tur':>14}{'aniqlangan':>14}{'sigma*':>10}{'usul':>14}")
    for i, t in enumerate(truth):
        print(f"{i:>6}{t['type_true']:>14}{res['band_types'][i]:>14}"
              f"{res['band_sigmas'][i]:>10.5f}{res['band_denoisers'][i]:>14}")