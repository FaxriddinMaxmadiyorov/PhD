"""
Furye almashtirish demo: davriy shovqinni Furye sohada aniqlash va tozalash.

Ishlatish:
    pip install numpy scipy matplotlib
    python fourier_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

np.random.seed(0)

# ============================================================
# 1-QADAM: Sun'iy tasvir yasaymiz (markazda kvadrat obyekt)
# ============================================================
img = np.zeros((256, 256))
img[80:176, 80:176] = 200            # markazga kvadrat qo'yamiz
img = ndimage.gaussian_filter(img, 3)  # chetlarini biroz silliqlaymiz

# ============================================================
# 2-QADAM: Davriy shovqin qo'shamiz (vertikal chiziqlar)
# har 8 pikselda takrorlanuvchi sinus to'lqin
# ============================================================
x = np.arange(256)
stripes = 40 * np.sin(2 * np.pi * x / 8)      # 8 = to'lqin davri
img_noisy = np.clip(img + stripes[np.newaxis, :], 0, 255)

# ============================================================
# 3-QADAM: Furye almashtirish (fazoviy soha -> chastota soha)
# fftshift markazni o'rtaga keltiradi
# ============================================================
F = np.fft.fftshift(np.fft.fft2(img_noisy))
spectrum = np.log1p(np.abs(F))    # log — ko'rish uchun qulay

# Shovqin spektrda qayerda? Chiziq davri 8 => markazdan 256/8 = 32 piksel
# Markaz (128,128), demak nuqtalar (128, 128-32)=(128,96) va (128,160)

# ============================================================
# 4-QADAM: Shovqin nuqtalarini o'chiramiz
# ============================================================
F_clean = F.copy()
for (py, px) in [(128, 96), (128, 160)]:   # ikkala simmetrik nuqta
    F_clean[py-3:py+4, px-3:px+4] = 0

# ============================================================
# 5-QADAM: Teskari almashtirish (chastota -> fazoviy soha)
# ============================================================
img_recovered = np.abs(np.fft.ifft2(np.fft.ifftshift(F_clean)))

# ============================================================
# Natijalarni chizamiz
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img_noisy, cmap='gray')
axes[0, 0].set_title("1) Shovqinli tasvir\n(kvadrat + davriy chiziqlar)")

axes[0, 1].imshow(spectrum, cmap='gray')
axes[0, 1].set_title("2) Furye spektri\n(shovqin = 2 ta yorug' nuqta)")

axes[1, 0].imshow(np.log1p(np.abs(F_clean)), cmap='gray')
axes[1, 0].set_title("3) Spektr\n(shovqin nuqtalari o'chirildi)")

axes[1, 1].imshow(img_recovered, cmap='gray')
axes[1, 1].set_title("4) Tozalangan tasvir\n(chiziqlar yo'qoldi)")

for a in axes.ravel():
    a.axis('off')

plt.tight_layout()
plt.savefig('fourier_result.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# Miqdoriy tekshiruv
# ============================================================
print(f"Bitta qatordagi tebranish (standart og'ish):")
print(f"  Shovqinli:   {img_noisy[40, :].std():.1f}")
print(f"  Tozalangan:  {img_recovered[40, :].std():.1f}")
print(f"Shovqin nuqtasi markazdan masofa: 256/8 = {256//8} piksel")