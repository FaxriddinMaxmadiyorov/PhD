import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Matplotlib style
# ==========================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.0
})

# ==========================================
# Spektral ma'lumotlar (namunaviy)
# ==========================================

wavelength = np.array([
    450, 490, 560, 670,
    705, 740, 780, 842,
    865, 945, 1375,
    1610, 2190
])

vegetation = np.array([
    0.03, 0.04, 0.08, 0.05,
    0.20, 0.35, 0.38, 0.45,
    0.46, 0.30, 0.02,
    0.25, 0.15
])

soil = np.array([
    0.10, 0.12, 0.16, 0.22,
    0.26, 0.28, 0.30, 0.32,
    0.33, 0.28, 0.03,
    0.38, 0.35
])

# ==========================================
# Sentinel-2 spektral kanallari
# ==========================================

bands = {
    "B2": 490,
    "B3": 560,
    "B4": 665,
    "B5": 705,
    "B6": 740,
    "B7": 783,
    "B8": 842,
    "B11": 1610,
    "B12": 2190
}

# ==========================================
# Plot
# ==========================================

fig, ax = plt.subplots(figsize=(10.5, 5))

# Qizil chekka (Red Edge)
ax.axvspan(
    680,
    750,
    color="0.94",
    zorder=0
)

# Sentinel-2 kanallari
for name, wl in bands.items():

    ax.axvline(
        wl,
        color="0.75",
        linestyle=":",
        linewidth=0.8,
        zorder=0
    )

    ax.text(
        wl,
        0.535,
        name,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35"
    )

# O'simlik
ax.plot(
    wavelength,
    vegetation,
    "-o",
    color="black",
    linewidth=2.2,
    markersize=6,
    label="O'simlik"
)

# Tuproq
ax.plot(
    wavelength,
    soil,
    "--s",
    color="0.5",
    linewidth=2.2,
    markersize=6,
    label="Tuproq"
)

# Red Edge yozuvi
# ax.text(
#     815,
#     0.505,
#     "Qizil chekka",
#     fontsize=11,
#     style="italic",
#     ha="center",
#     va="top"
# )

# O'qlar
ax.set_xlim(400, 2225)
ax.set_ylim(0, 0.52)

ax.set_xlabel("To'lqin uzunligi (nm)", fontsize=14)
ax.set_ylabel("Reflektans", fontsize=14)

# Grid
ax.grid(
    True,
    linestyle="-",
    linewidth=0.5,
    alpha=0.25
)

# Legend
leg = ax.legend(
    loc="upper left",
    frameon=True
)

leg.get_frame().set_alpha(0.95)

plt.tight_layout()

# ==========================================
# Saqlash
# ==========================================

outfile = "vegetation_soil_spectral_signature.png"

plt.savefig(
    outfile,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Saqlandi: {outfile}")

plt.show()