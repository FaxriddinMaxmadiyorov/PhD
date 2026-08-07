"""
Multispektrli sahna va shovqin generatori
===========================================

Multispektrli tasvirning ikkita muhim xossasini modellashtiradi:

1. Kanallar orasidagi kuchli fazoviy korrelyatsiya. Yer usti strukturasi
   (dala chegaralari, suv havzasi, relyef) barcha kanallarda bir xil
   joyda turadi, faqat kontrast va yorqinlik farq qiladi.

2. Kanallar bo'yicha turlicha shovqin rejimi. Har bir kanalning
   yoritilganligi va sensor sezgirligi har xil bo'lgani uchun foton
   oqimi ham har xil bo'ladi. Yorug' kanalda foton ko'p, shuning uchun
   Puasson tashkil etuvchisi ustunlik qiladi; qorong'i kanalda foton
   kam, shuning uchun o'qish shovqini (Gauss) ustun bo'ladi. Demak
   bitta tasvirning turli kanallari amalda turli shovqin turiga
   tegishli bo'ladi.

Aynan ikkinchi xossa kanallar bo'yicha o'rtachalashni noto'g'ri qiladi:
o'rtachalash turli rejimlarni bitta soniga siqib, ikkalasiga ham mos
kelmaydigan oraliq qiymat beradi.
"""

import numpy as np
from scipy import ndimage


def make_spatial_structure(size: int, seed: int) -> np.ndarray:
    """
    Kanallar uchun umumiy fazoviy struktura (0..1 oralig'ida).
    To'rt arxetip aralashmasi: davriy naqsh, bo'lakli-tekis mintaqalar,
    ko'p masshtabli tekstura va yo'nalishli gradient.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    freq = rng.uniform(10, 30)
    phase = rng.uniform(0, 2 * np.pi)
    periodic = np.sin(xx / freq + phase) * np.cos(yy / (freq * 1.2))

    piecewise = np.zeros_like(xx)
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.uniform(0, size, 2)
        radius = rng.uniform(size / 10, size / 4)
        piecewise[(xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2] = rng.uniform(-1, 1)

    fractal = np.zeros_like(xx)
    amplitude = 1.0
    for octave in range(4):
        coarse = rng.normal(0, 1, (max(2, size >> (3 - octave)),) * 2)
        upscaled = ndimage.zoom(coarse, size / coarse.shape[0], order=1)[:size, :size]
        fractal += amplitude * upscaled
        amplitude *= 0.5
    fractal /= (np.std(fractal) + 1e-8)

    angle = rng.uniform(0, np.pi)
    gradient = (np.cos(angle) * xx + np.sin(angle) * yy) / size

    w = rng.dirichlet([1.0, 1.0, 1.0, 0.5])
    scene = w[0] * periodic + w[1] * piecewise + w[2] * fractal + w[3] * gradient
    return (scene - scene.min()) / (np.ptp(scene) + 1e-8)


def make_multispectral_scene(size: int, n_bands: int, seed: int) -> np.ndarray:
    """
    (size, size, n_bands) shaklidagi buzilmagan multispektrli sahna.

    Umumiy struktura barcha kanallarda saqlanadi, har bir kanal esa o'z
    yorqinlik darajasi va kontrastiga ega bo'ladi. Yorqinlik darajasining
    kanallar bo'yicha keng tarqalishi (0,08 dan 0,75 gacha) muhim: aynan
    shu tarqalish keyinchalik kanallar bo'yicha turlicha shovqin rejimini
    keltirib chiqaradi.
    """
    rng = np.random.default_rng(seed + 9973)
    structure = make_spatial_structure(size, seed)

    bands = []
    for _ in range(n_bands):
        brightness = rng.uniform(0.08, 0.75)   # kanalning o'rtacha yorqinligi
        contrast = rng.uniform(0.15, 0.45)     # kanalning kontrasti
        band = brightness + contrast * (structure - 0.5)
        bands.append(np.clip(band, 0.01, 1.0))

    return np.stack(bands, axis=-1)


def apply_multispectral_noise(clean: np.ndarray, rng: np.random.Generator,
                               level: float = 0.6,
                               impulse_bands: float = 0.0):
    """
    Har bir kanalga o'z shovqin rejimini qo'llaydi.

    Kanalning yorqinligiga qarab foton miqyosi (Puasson kuchi) va o'qish
    shovqini (Gauss kuchi) belgilanadi, shu sababli hosil bo'ladigan
    shovqin turi kanaldan kanalga tabiiy ravishda o'zgaradi.

    impulse_bands - impulsiv buzilish qo'shiladigan kanallar ulushi
    (uzatish xatoliklari odatda barcha kanalga emas, ayrimlariga tegadi).

    Qaytaradi: (noisy, per_band_truth), bunda per_band_truth har bir kanal
    uchun haqiqiy shovqin parametrlari lug'ati.
    """
    size, _, n_bands = clean.shape
    noisy = np.zeros_like(clean)
    truth = []

    impulse_mask_bands = rng.random(n_bands) < impulse_bands

    for i in range(n_bands):
        band = clean[:, :, i]
        mean_level = float(band.mean())

        # Foton miqyosi yorqinlikka proportsional: yorug' kanalda foton ko'p.
        # Qo'shimcha ravishda kanalning kvant samaradorligi ham hisobga
        # olinadi, chunki spektral kanallar sezgirligi bir xil emas.
        quantum_efficiency = rng.uniform(0.3, 3.0)
        photon_scale = (60.0 + 1400.0 * mean_level) * quantum_efficiency
        photon_scale *= (1.0 - 0.6 * level)

        # O'qish shovqini signaldan mustaqil. Uning kattaligi kanalning
        # o'qish zanjiri sifatiga bog'liq va kanallar bo'yicha bir tartibga
        # farq qilishi mumkin: ayrim kanallar sovutilgan va kam shovqinli,
        # boshqalari sezilarli o'qish shovqiniga ega.
        read_quality = 10.0 ** rng.uniform(-1.0, 0.9)
        read_sigma = (0.004 + 0.030 * level) * read_quality

        out = rng.poisson(np.clip(band, 0, None) * photon_scale) / photon_scale
        out = out + rng.normal(0, read_sigma, band.shape)

        impulse_prob = 0.0
        if impulse_mask_bands[i]:
            impulse_prob = 0.02 + 0.10 * level
            m = rng.random(band.shape) < impulse_prob
            out[m] = rng.choice([0.0, 1.0], size=int(m.sum()))

        noisy[:, :, i] = np.clip(out, 0, 1)

        # Haqiqiy (bilingan) parametrlar: NLF koeffitsiyentlari
        #   Puasson qismi:  v = mu / photon_scale  ->  a_true = 1 / photon_scale
        #   Gauss qismi:    v = read_sigma^2       ->  b_true = read_sigma^2
        a_true = 1.0 / photon_scale
        b_true = read_sigma ** 2
        truth.append({
            "a_true": a_true,
            "b_true": b_true,
            "impulse_true": impulse_prob,
            "mean_level": mean_level,
            "photon_scale": photon_scale,
            "read_sigma": read_sigma,
            # umumiy shovqin standart og'ishi o'rtacha intensivlikda
            "sigma_true": float(np.sqrt(a_true * mean_level + b_true)),
            "type_true": _dominant_type(a_true, b_true, mean_level, impulse_prob),
        })

    return noisy, truth


def _dominant_type(a_true: float, b_true: float, mean_level: float,
                    impulse_prob: float) -> str:
    """
    Kanalning haqiqiy shovqin turi: qaysi tashkil etuvchi o'rtacha
    intensivlikda dispersiyaning katta qismini beradi.

    Impulsiv buzilish mavjud bo'lsa u ustun deb qaraladi, chunki uning
    qayta ishlashga ta'siri qolgan tashkil etuvchilardan kuchliroq.
    """
    if impulse_prob > 0:
        return "salt_pepper"

    poisson_var = a_true * mean_level
    gauss_var = b_true
    ratio = poisson_var / (gauss_var + 1e-12)

    if ratio > 4.0:
        return "poisson"
    if ratio < 0.25:
        return "gaussian"
    return "mixed"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clean = make_multispectral_scene(size=96, n_bands=6, seed=1)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)

    print(f"Sahna: {clean.shape},  shovqinli: {noisy.shape}\n")
    print(f"{'kanal':>6}{'yorqinlik':>12}{'a_true':>12}{'b_true':>12}"
          f"{'sigma_true':>12}{'turi':>14}")
    for i, t in enumerate(truth):
        print(f"{i:>6}{t['mean_level']:>12.3f}{t['a_true']:>12.6f}{t['b_true']:>12.6f}"
              f"{t['sigma_true']:>12.4f}{t['type_true']:>14}")

    types = [t["type_true"] for t in truth]
    print(f"\nBitta tasvirdagi turlar: {sorted(set(types))}")
    print("Bu aynan kanallar bo'yicha o'rtachalash yo'qotadigan farq.")