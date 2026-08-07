"""
Shovqinni xarakterlash moduli (kanal bo'yicha)
================================================

Multispektrli tasvirning HAR BIR KANALI uchun alohida shovqin xususiyat
vektorini hisoblaydi:  [a, b, impulse, kurtosis]

Nima uchun kanal bo'yicha
--------------------------
Shovqin turi bu yerda a/b nisbati orqali ta'riflanadi, ya'ni Puasson va
Gauss tashkil etuvchilari ulushlari nisbati bilan. Bu nisbat kanalning
yorqinligiga va sensor sezgirligiga bevosita bog'liq: yorug' kanalda
foton oqimi katta bo'lgani uchun Puasson tashkil etuvchisi ustunlik
qiladi, qorong'i kanalda esa o'qish shovqini ustun bo'ladi. Shu sababli
bitta tasvirning turli kanallari turli shovqin turiga tegishli bo'lishi
mumkin va kanallar bo'yicha o'rtachalash bu farqni yo'q qiladi.

Ierarxik regulyarizatsiya
--------------------------
Kanallarni to'liq mustaqil baholashning zaif tomoni bor: ayrim kanalda
tekis mintaqa kam bo'lsa (bulutli yoki tekstura boy sahna), NLF
regressiyasi beqaror chiqadi. Buning oldini olish uchun har bir kanal
bahosi barcha kanallarning umumiy o'rtachasiga qisman tortiladi, tortish
kuchi esa shu kanaldagi ishonchli namunalar soniga bog'liq qilinadi.
Bu James-Stein tipidagi qisqartirish bo'lib, kanallar butunlay mustaqil
emasligi (bir sensor, bir sahna) faktiga tayanadi.
"""

import numpy as np
from scipy import ndimage, stats


def gradient_magnitude(band: np.ndarray) -> np.ndarray:
    """Sobel operatorlari yordamida gradient kattaligi."""
    gx = ndimage.sobel(band, axis=1)
    gy = ndimage.sobel(band, axis=0)
    return np.hypot(gx, gy)


def flat_region_mask(band: np.ndarray, flat_percentile: float = 30.0) -> np.ndarray:
    """Gradienti eng past piksellarni (tekis mintaqalarni) belgilaydi."""
    grad = gradient_magnitude(band)
    return grad <= np.percentile(grad, flat_percentile)


def local_mean_variance(band: np.ndarray, window: int = 7):
    """Mahalliy o'rtacha va dispersiya (uniform filtr orqali)."""
    mean = ndimage.uniform_filter(band, size=window)
    mean_sq = ndimage.uniform_filter(band ** 2, size=window)
    return mean, np.clip(mean_sq - mean ** 2, 0.0, None)


def fit_nlf_band(band: np.ndarray, window: int = 7, flat_percentile: float = 60.0,
                  n_bins: int = 12, envelope_percentile: float = 15.0):
    """
    Bitta kanal uchun shovqin darajasi funksiyasi:  v = a*mu + b

    Pastki chegara (lower envelope) usuli
    --------------------------------------
    (mu, v) juftliklariga to'g'ridan-to'g'ri regressiya qo'llash a
    koeffitsiyentini tizimli ravishda oshirib yuboradi, chunki "tekis"
    deb belgilangan mintaqalarda ham qoldiq tekstura saqlanib qoladi va
    u dispersiyani faqat OSHIRADI, hech qachon kamaytirmaydi. Demak
    tarqoqlikning pastki chegarasi sof shovqin darajasiga mos keladi.

    Shu sababli intensivlik oralig'i teng to'ldirilgan binlarga
    bo'linadi, har bir binda dispersiyaning past protsentili olinadi va
    chiziqli model aynan shu nuqtalarga moslashtiriladi. Bu tekstura
    ifloslanishini sezilarli darajada kamaytiradi.

    Qaytaradi: (a, b, n_samples, mu_span)
    """
    mean, variance = local_mean_variance(band, window=window)
    mask = flat_region_mask(band, flat_percentile=flat_percentile)

    mu_vals, v_vals = mean[mask], variance[mask]
    n = int(mu_vals.size)

    if n < 50:
        return 0.0, float(np.mean(v_vals)) if n else 0.0, n, 0.0

    mu_span = float(np.percentile(mu_vals, 95) - np.percentile(mu_vals, 5))
    if mu_span < 1e-6:
        # Intensivlik deyarli o'zgarmas: qiyalikni ajratib bo'lmaydi.
        return 0.0, float(np.percentile(v_vals, envelope_percentile)), n, mu_span

    edges = np.percentile(mu_vals, np.linspace(0.0, 100.0, n_bins + 1))
    xs, ys = [], []
    for i in range(n_bins):
        sel = (mu_vals >= edges[i]) & (mu_vals <= edges[i + 1])
        if int(sel.sum()) < 30:
            continue
        xs.append(float(mu_vals[sel].mean()))
        ys.append(float(np.percentile(v_vals[sel], envelope_percentile)))

    if len(xs) < 3:
        return 0.0, float(np.percentile(v_vals, envelope_percentile)), n, mu_span

    a, b = np.polyfit(np.array(xs), np.array(ys), deg=1)

    # a va b dispersiyaning tashkil etuvchilari bo'lgani uchun manfiy
    # bo'la olmaydi; baholash xatoligi ularni manfiy chiqarishi mumkin.
    return float(max(a, 0.0)), float(max(b, 0.0)), n, mu_span


def impulse_fraction(band: np.ndarray, k: float = 3.0, size: int = 3) -> float:
    """Median qoldig'i robust chegaradan oshgan piksellar ulushi."""
    residual = band - ndimage.median_filter(band, size=size)
    mad = np.median(np.abs(residual - np.median(residual)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        return 0.0
    return float(np.mean(np.abs(residual) > k * robust_sigma))


def kurtosis_feature(band: np.ndarray, size: int = 3) -> float:
    """Median qoldig'ining ortiqcha kurtozi."""
    residual = band - ndimage.median_filter(band, size=size)
    return float(stats.kurtosis(residual.ravel(), fisher=True, bias=False))


def _shrink_toward_global(values, weights, strength: float = 1.0):
    """
    James-Stein tipidagi qisqartirish: har bir kanal bahosi umumiy
    o'rtachaga qisman tortiladi.

        v_hat_i = w_i * v_i + (1 - w_i) * v_global

    bunda w_i = n_i / (n_i + strength * n_ref). Kanalda ishonchli namuna
    ko'p bo'lsa o'z bahosiga ishoniladi, kam bo'lsa umumiy o'rtachaga
    suyaniladi.

    MUHIM: tajriba shuni ko'rsatdiki, ushbu qisqartirish shovqin turini
    aniqlash aniqligini YOMONLASHTIRADI (strength=0 da 76,4%, 0,3 da
    66,7%, 1,0 da 65,3%). Sabab shundaki, qisqartirish kanallar orasidagi
    farqni kamaytiradi, holbuki butun yondashuvning maqsadi aynan shu
    farqni aniqlashdir. Shu sababli standart qiymat sifatida strength=0
    (qisqartirishsiz) qabul qilingan; funksiya faqat qiyosiy tajribalar
    uchun saqlab qolingan.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if values.size <= 1:
        return values

    reference = np.median(weights[weights > 0]) if np.any(weights > 0) else 1.0
    w = weights / (weights + strength * reference + 1e-12)
    global_value = float(np.average(values, weights=weights + 1e-12))
    return w * values + (1.0 - w) * global_value


def characterize_noise_per_band(image: np.ndarray, window: int = 7,
                                 flat_percentile: float = 60.0,
                                 impulse_k: float = 3.0,
                                 shrinkage: float = 0.0,
                                 isolated_impulse: bool = False) -> dict:
    """
    Har bir kanal uchun xususiyat vektorini hisoblaydi.

    image: (H, W) yoki (H, W, C)

    Qaytaradi:
      {
        "features":  (C, 4) - har bir kanal uchun [a, b, impulse, kurtosis],
        "n_samples": (C,)   - NLF fitida ishtirok etgan piksellar soni,
        "mu_span":   (C,)   - intensivlik oralig'i kengligi
      }

    shrinkage=0 bo'lsa kanallar to'liq mustaqil baholanadi.

    isolated_impulse=True bo'lsa, impulsivlik belgisi yakkalik sharti
    bilan hisoblanadi. Haqiqiy sun'iy yo'ldosh tasvirlari uchun tavsiya
    etiladi, chunki oddiy variant tasvirdagi binolar va dala
    chegaralarini impulsiv buzilish deb belgilaydi.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]

    n_bands = image.shape[2]
    a_raw, b_raw, ns, spans, imp, kur = [], [], [], [], [], []

    for i in range(n_bands):
        band = image[:, :, i]
        a, b, n, span = fit_nlf_band(band, window=window,
                                      flat_percentile=flat_percentile)
        a_raw.append(a)
        b_raw.append(b)
        ns.append(n)
        spans.append(span)
        if isolated_impulse:
            imp.append(impulse_fraction_isolated(band, k=impulse_k))
        else:
            imp.append(impulse_fraction(band, k=impulse_k))
        kur.append(kurtosis_feature(band))

    # Ishonch og'irligi: namunalar soni va intensivlik oralig'i kengligi
    # birgalikda NLF fitining ishonchliligini belgilaydi.
    ns_arr = np.asarray(ns, dtype=np.float64)
    span_arr = np.asarray(spans, dtype=np.float64)
    confidence = ns_arr * (span_arr / (span_arr.max() + 1e-12))

    if shrinkage > 0 and n_bands > 1:
        a_adj = _shrink_toward_global(a_raw, confidence, strength=shrinkage)
        b_adj = _shrink_toward_global(b_raw, confidence, strength=shrinkage)
    else:
        a_adj, b_adj = np.asarray(a_raw), np.asarray(b_raw)

    features = np.column_stack([a_adj, b_adj, imp, kur])
    return {"features": features, "n_samples": ns_arr, "mu_span": span_arr}


def characterize_noise(image: np.ndarray, **kwargs) -> dict:
    """
    Kanallar bo'yicha o'rtachalangan yagona xususiyat vektori.

    DIQQAT: ushbu funksiya faqat taqqoslash tajribalarida, ya'ni kanal
    bo'yicha yondashuvning ustunligini ko'rsatish uchun ishlatiladi.
    Asosiy pipeline characterize_noise_per_band dan foydalanadi, chunki
    o'rtachalash turli kanallardagi turli shovqin rejimlarini bitta
    qiymatga siqib, ularning hech biriga mos kelmaydigan natija beradi.
    """
    result = characterize_noise_per_band(image, **kwargs)
    mean_features = result["features"].mean(axis=0)
    return {
        "a": float(mean_features[0]),
        "b": float(mean_features[1]),
        "impulse": float(mean_features[2]),
        "kurtosis": float(mean_features[3]),
        "feature_vector": mean_features,
    }


if __name__ == "__main__":
    from multispectral_data import make_multispectral_scene, apply_multispectral_noise

    rng = np.random.default_rng(3)
    clean = make_multispectral_scene(size=128, n_bands=6, seed=11)
    noisy, truth = apply_multispectral_noise(clean, rng, level=0.6, impulse_bands=0.2)

    res = characterize_noise_per_band(noisy)
    avg = characterize_noise(noisy)

    print("Kanal bo'yicha baholangan va haqiqiy qiymatlar")
    print("-" * 84)
    print(f"{'kanal':>6}{'a_baho':>12}{'a_haqiqiy':>12}{'b_baho':>12}"
          f"{'b_haqiqiy':>12}{'imp':>8}{'turi':>14}")
    for i, t in enumerate(truth):
        f = res["features"][i]
        print(f"{i:>6}{f[0]:>12.6f}{t['a_true']:>12.6f}{f[1]:>12.6f}"
              f"{t['b_true']:>12.6f}{f[2]:>8.3f}{t['type_true']:>14}")

    print(f"\nKanallar bo'yicha o'rtachalangan (eski usul): "
          f"a={avg['a']:.6f}, b={avg['b']:.6f}")
    print("Yagona qiymat yuqoridagi turli rejimlarning hech biriga mos kelmaydi.")


def impulse_fraction_isolated(band: np.ndarray, k: float = 3.0, size: int = 3,
                               max_neighbor_frac: float = 0.35) -> float:
    """
    Impulsivlik ulushining yakkalik sharti bilan hisoblangan varianti.

    Oddiy impulse_fraction() median qoldig'i robust chegaradan oshgan
    barcha piksellarni impulsiv deb hisoblaydi. Haqiqiy sun'iy yo'ldosh
    tasvirlarida bu jiddiy muammoga olib keladi: binolar, yo'llar va dala
    chegaralari ham qoldiqda katta chetlashuv hosil qiladi va detektor
    ularni impulsiv deb belgilaydi.

    Ushbu variant qo'shimcha shart qo'yadi: haqiqiy impulsiv buzilish
    fazoviy jihatdan YAKKA bo'ladi, tekstura esa BOG'LANGAN. Shu sababli
    belgilangan pikselning qo'shnilari ham belgilangan bo'lsa, u tekstura
    deb hisoblanadi va inobatga olinmaydi.

    O'lchangan natijalar (30 ta EuroSAT patchi, 13 kanal):
        oddiy variant   - o'rtacha ulush 0,105
        yakkalik sharti - o'rtacha ulush 0,038

    Ya'ni yakkalik sharti yolg'on ijobiylikni qariyb uch barobar
    kamaytiradi, ammo uni to'liq bartaraf etmaydi. Haqiqiy ma'lumotda
    impulsiv buzilishni ishonchli aniqlash ochiq masala bo'lib qolmoqda.
    """
    residual = band - ndimage.median_filter(band, size=size)
    mad = np.median(np.abs(residual - np.median(residual)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        return 0.0

    flagged = np.abs(residual) > k * robust_sigma
    if not flagged.any():
        return 0.0

    # Har bir piksel uchun qo'shnilarining belgilangan ulushi (o'zisiz)
    local = ndimage.uniform_filter(flagged.astype(np.float64), size=3)
    neighbor_frac = (local * 9.0 - 1.0) / 8.0

    isolated = flagged & (neighbor_frac <= max_neighbor_frac)
    return float(isolated.mean())