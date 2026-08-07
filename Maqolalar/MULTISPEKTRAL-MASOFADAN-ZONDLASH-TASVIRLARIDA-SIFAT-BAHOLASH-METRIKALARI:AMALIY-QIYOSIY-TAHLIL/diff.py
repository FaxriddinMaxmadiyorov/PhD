import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Matplotlib style
# ==========================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12
})

# ==========================================
# Gradient hosil qilish
# ==========================================

width = 600
height = 220

gradient = np.tile(
    np.linspace(0, 255, width),
    (height, 1)
)

# ==========================================
# Bit-depth bo'yicha kvantlash
# ==========================================

def quantize(img, bits):
    """
    img : [0,255]
    bits : bit-depth
    """
    levels = 2 ** bits

    q = np.round(img * (levels - 1) / 255)
    q = q * 255 / (levels - 1)

    return q


images = [
    gradient,
    quantize(gradient, 4),
    quantize(gradient, 2),
    quantize(gradient, 1),
]

titles = [
    "8-bit (256 daraja)",
    "4-bit (16 daraja)",
    "2-bit (4 daraja)",
    "1-bit (2 daraja)",
]

# ==========================================
# Plot
# ==========================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(14, 3.8)
)

for ax, img, title in zip(axes, images, titles):

    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=255,
        aspect="auto"
    )

    # ax.set_title(title, fontsize=13)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()

# ==========================================
# Save
# ==========================================

outfile = "bit_depth_comparison.png"

plt.savefig(
    outfile,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Saqlandi: {outfile}")

plt.show()