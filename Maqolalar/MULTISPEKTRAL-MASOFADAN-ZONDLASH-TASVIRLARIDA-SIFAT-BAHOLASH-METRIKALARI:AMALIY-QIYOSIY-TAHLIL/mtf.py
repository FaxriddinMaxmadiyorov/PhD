import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MTF egri chiziqlari
# ==========================================

f = np.linspace(0, 1, 500)

# Ikki xil MTF (namunaviy)
mtf_good = np.exp(-3 * f)
mtf_bad = np.exp(-7 * f)

# ==========================================
# Sinusoidal test pattern
# ==========================================

width = 900
height = 250

x = np.linspace(0, 1, width)

# Chapdan o'ngga chastota oshadi
frequency = 2 + 18 * x

phase = 2 * np.pi * np.cumsum(frequency) / width

# Kontrast ham pasayadi (MTF effekti)
contrast = np.exp(-2.2 * x)

signal = 0.5 + 0.5 * contrast * np.sin(phase)

pattern = np.tile(signal, (height, 1))

# ==========================================
# Plot
# ==========================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(13, 4),
    gridspec_kw={"width_ratios": [1, 1]}
)

# ------------------------------------------
# Chap grafik
# ------------------------------------------

ax[0].plot(
    f,
    mtf_good,
    color="black",
    linewidth=1.8,
    label="Yaxshi tizim (keng MTF)"
)

ax[0].plot(
    f,
    mtf_bad,
    color="gray",
    linestyle="--",
    linewidth=2.0,
    label="Yomon tizim (tor MTF)"
)

ax[0].axhline(
    0.5,
    color="gray",
    linestyle=":",
    linewidth=1.2
)

ax[0].set_xlim(-0.05, 1.02)
ax[0].set_ylim(-0.05, 1.05)

ax[0].set_xlabel("Fazoviy chastota", fontsize=12)
ax[0].set_ylabel("MTF (kontrast uzatish)", fontsize=12)

# ax[0].set_title("a) MTF egri chizig'i", fontsize=13)

ax[0].legend(loc="upper right", fontsize=11)

# ------------------------------------------
# O'ng rasm
# ------------------------------------------

ax[1].imshow(
    pattern,
    cmap="gray",
    aspect="auto",
    vmin=0,
    vmax=1
)

# ax[1].set_title(
#     "b) Chastota oshgani sari kontrast pasayadi",
#     fontsize=13
# )

ax[1].axis("off")

plt.tight_layout()

# ==========================================
# Saqlash
# ==========================================

output_file = "mtf_demo.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Saqlandi: {output_file}")

plt.show()