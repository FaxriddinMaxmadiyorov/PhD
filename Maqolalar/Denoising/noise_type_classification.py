"""
Shovqin turini klassifikatsiyalash moduli (kanal bo'yicha)
============================================================

Har bir kanalning xususiyat vektorini [a, b, impulse, kurtosis] qabul
qilib, shu kanalning shovqin turini aniqlaydi:

    gaussian | poisson | mixed | salt_pepper

Kalibrlash to'plami multispektrli sintetik sahnalardan yig'iladi va
har bir KANAL alohida namuna hisoblanadi, chunki tur kanal darajasidagi
xossadir.

Xususiyatlarni kengaytirish
----------------------------
Xom [a, b, impulse, kurtosis] vektoriga ikkita hosila belgi qo'shiladi:
Puasson va Gauss dispersiyalari nisbatining logarifmi hamda kanalning
o'rtacha intensivligi. Nisbat aynan turni belgilovchi kattalik bo'lgani
uchun uni bevosita belgi sifatida berish klassifikator ishini
yengillashtiradi; o'rtacha intensivlik esa nisbatni to'g'ri talqin
qilish uchun zarur kontekst beradi, chunki Puasson dispersiyasi
intensivlikka proportsionaldir.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from noise_characterization import characterize_noise_per_band
from multispectral_data import make_multispectral_scene, apply_multispectral_noise

NOISE_TYPES = ["gaussian", "poisson", "mixed", "salt_pepper"]


def expand_features(raw_features: np.ndarray, mean_levels: np.ndarray) -> np.ndarray:
    """
    (C, 4) xom xususiyatlarni (C, 6) kengaytirilgan xususiyatlarga
    aylantiradi: [a, b, impulse, kurtosis, log(Puasson/Gauss), mean_level].
    """
    raw_features = np.atleast_2d(raw_features)
    mean_levels = np.asarray(mean_levels, dtype=np.float64).reshape(-1)

    a = raw_features[:, 0]
    b = raw_features[:, 1]

    poisson_var = np.clip(a * mean_levels, 0.0, None)
    gauss_var = np.clip(b, 0.0, None)
    log_ratio = np.log10((poisson_var + 1e-9) / (gauss_var + 1e-9))

    return np.column_stack([raw_features, log_ratio, mean_levels])


def features_from_image(image: np.ndarray, **kwargs):
    """
    Tasvirdan kanal bo'yicha kengaytirilgan xususiyat matritsasini
    hisoblaydi. Qaytaradi: (C, 6) massiv.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]

    char = characterize_noise_per_band(image, **kwargs)
    mean_levels = image.reshape(-1, image.shape[2]).mean(axis=0)
    return expand_features(char["features"], mean_levels)


def build_calibration_set(n_scenes: int = 60, size: int = 96, n_bands: int = 6,
                           seed: int = 42):
    """
    Multispektrli sahnalardan kanal darajasidagi kalibrlash to'plamini
    yig'adi. Har bir sahna n_bands ta namuna beradi.
    """
    rng = np.random.default_rng(seed)
    features, labels = [], []

    for s in range(n_scenes):
        clean = make_multispectral_scene(size, n_bands, seed=int(rng.integers(0, 10 ** 6)))
        level = float(rng.uniform(0.15, 1.0))
        impulse_bands = float(rng.choice([0.0, 0.15, 0.3]))
        noisy, truth = apply_multispectral_noise(clean, rng, level=level,
                                                  impulse_bands=impulse_bands)

        x = features_from_image(noisy)
        for i, t in enumerate(truth):
            features.append(x[i])
            labels.append(t["type_true"])

    return np.array(features), np.array(labels)


class NoiseTypeClassifier:
    """Standartlashtirish + k eng yaqin qo'shni klassifikatori."""

    def __init__(self, k: int = 21):
        self.k = k
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.model.fit(self.scaler.fit_transform(features), labels)
        self._fitted = True
        return self

    def predict_bands(self, features: np.ndarray) -> np.ndarray:
        """(C, 6) xususiyat matritsasidan har bir kanal uchun tur."""
        if not self._fitted:
            raise RuntimeError("Klassifikator o'rgatilmagan")
        return self.model.predict(self.scaler.transform(np.atleast_2d(features)))

    def predict_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Tasvirdan to'g'ridan-to'g'ri kanallar bo'yicha turlar ro'yxati."""
        return self.predict_bands(features_from_image(image, **kwargs))


if __name__ == "__main__":
    print("Kalibrlash to'plami yig'ilmoqda (multispektrli sahnalar)...")
    x, y = build_calibration_set(n_scenes=120, size=96, n_bands=6, seed=42)
    print(f"Namunalar (kanal darajasida): {x.shape[0]}, belgilar: {x.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)
    print("Sinflar taqsimoti: " + ", ".join(f"{u}={c}" for u, c in zip(unique, counts)))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=0
    )
    clf = NoiseTypeClassifier(k=21).fit(x_train, y_train)
    y_pred = clf.model.predict(clf.scaler.transform(x_test))

    print(f"\nAniqlik: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Chalkashlik matritsasi (qator = haqiqiy, ustun = bashorat):")
    print(NOISE_TYPES)
    print(confusion_matrix(y_test, y_pred, labels=NOISE_TYPES))
    print()
    print(classification_report(y_test, y_pred, labels=NOISE_TYPES, zero_division=0))