import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRLAR
# ==========================================
image_path = "image.png"      # <-- PNG rasmingiz
target_mse = 500                 # Maqsad MSE

# ==========================================
# RASMNI O'QISH (RGB yoki grayscale bo'lishi mumkin)
# ==========================================
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Rasm topilmadi: {image_path}")

# Rangli bo'lsa grayscale ga o'tkazish
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img.astype(np.float32)

# ==========================================
# MSE
# ==========================================
def mse(a, b):
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)

# ==========================================
# Kvantlash
# ==========================================
def quantize(image, levels):
    step = 255 / (levels - 1)
    q = np.round(image / step) * step
    return np.clip(q, 0, 255)

# ==========================================
# Gaussian Blur
# ==========================================
def blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

# ==========================================
# Gaussian Noise
# ==========================================
def add_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)

# ==========================================
# Binary Search: Kvantlash
# ==========================================
def fit_quantization(target):

    low = 2
    high = 256

    best = img.copy()

    for _ in range(25):

        levels = (low + high) // 2

        q = quantize(img, levels)

        e = mse(img, q)

        if abs(e - target) < abs(mse(img, best) - target):
            best = q

        if e > target:
            low = levels + 1
        else:
            high = levels - 1

    return best

# ==========================================
# Binary Search: Blur
# ==========================================
def fit_blur(target):

    low = 0.01
    high = 20

    best = img.copy()

    for _ in range(30):

        sigma = (low + high) / 2

        b = blur(img, sigma)

        e = mse(img, b)

        if abs(e - target) < abs(mse(img, best) - target):
            best = b

        if e < target:
            low = sigma
        else:
            high = sigma

    return best

# ==========================================
# Binary Search: Noise
# ==========================================
def fit_noise(target):

    low = 0
    high = 80

    best = img.copy()

    for _ in range(30):

        sigma = (low + high) / 2

        n = add_noise(img, sigma)

        e = mse(img, n)

        if abs(e - target) < abs(mse(img, best) - target):
            best = n

        if e < target:
            low = sigma
        else:
            high = sigma

    return best

# ==========================================
# Buzilishlarni yaratish
# ==========================================
quant = fit_quantization(target_mse)
blurred = fit_blur(target_mse)
noise = fit_noise(target_mse)

print("Quantization MSE =", mse(img, quant))
print("Blur MSE         =", mse(img, blurred))
print("Noise MSE        =", mse(img, noise))

# ==========================================
# Plot
# ==========================================
fig, ax = plt.subplots(1, 4, figsize=(17,5))

ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
# ax[0].set_title("a) Etalon", fontsize=15)
ax[0].axis("off")

ax[1].imshow(quant, cmap="gray", vmin=0, vmax=255)
# ax[1].set_title(f"b) Kvantlash\nMSE = {mse(img, quant):.1f}", fontsize=14)
ax[1].axis("off")

ax[2].imshow(blurred, cmap="gray", vmin=0, vmax=255)
# ax[2].set_title(f"v) Xiralashish\nMSE = {mse(img, blurred):.1f}", fontsize=14)
ax[2].axis("off")

ax[3].imshow(noise, cmap="gray", vmin=0, vmax=255)
# ax[3].set_title(f"g) Shovqin\nMSE = {mse(img, noise):.1f}", fontsize=14)
ax[3].axis("off")

plt.tight_layout()
# plt.show()

import os

plt.tight_layout(rect=[0, 0, 1, 0.93])

# Kiruvchi rasm nomidan foydalanish
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = f"{base_name}_mse_comparison.png"

plt.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")

print(f"Rasm saqlandi: {output_path}")

plt.show()