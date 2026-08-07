import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARAMETRLAR
# ==========================================
image_path = "image.png"

# Atmosfera modeli:
# g = f * t + A
t = 0.65      # atmosfera o'tkazuvchanligi
A = 90        # path radiance

# ==========================================
# FONT
# ==========================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 15,
    "savefig.dpi": 600
})

# ==========================================
# RASM
# ==========================================
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(image_path)

img = img.astype(np.float32)

# ==========================================
# Atmosfera modeli
# ==========================================
hazy = img * t + A
hazy = np.clip(hazy, 0, 255)

# ==========================================
# Figure
# ==========================================
fig, ax = plt.subplots(
    1,
    3,
    figsize=(17,5),
    constrained_layout=True
)

# ----------------------------------------------------
# Original
# ----------------------------------------------------
ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("a) Toza sahna f(x,y)")
ax[0].axis("off")

# ----------------------------------------------------
# Haze
# ----------------------------------------------------
ax[1].imshow(hazy, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("b) Atmosfera tumani\n$g=f\\cdot t+A$")
ax[1].axis("off")

# ----------------------------------------------------
# Histogram
# ----------------------------------------------------
ax[2].hist(
    img.ravel(),
    bins=64,
    color="gray",
    alpha=0.85,
    label="Toza"
)

ax[2].hist(
    hazy.ravel(),
    bins=64,
    color="lightgray",
    alpha=0.95,
    label="Tumanli"
)

ax[2].set_title("c) Gistogramma siqiladi\n(kontrast pasayadi)")
ax[2].set_xlabel("Intensivlik")
ax[2].set_ylabel("Piksel soni")

ax[2].set_xlim(0,255)
ax[2].legend()

# ==========================================
# SAVE
# ==========================================
base = os.path.splitext(os.path.basename(image_path))[0]

output = base + "_atmosfera_modeli.png"

fig.savefig(
    output,
    dpi=600,
    facecolor="white"
)

print("Saved:", output)

plt.show()