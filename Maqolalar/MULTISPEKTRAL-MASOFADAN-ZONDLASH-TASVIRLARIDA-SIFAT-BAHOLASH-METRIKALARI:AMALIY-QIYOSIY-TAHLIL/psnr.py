import numpy as np
import matplotlib.pyplot as plt

# ==========================
# PSNR formulasi (8-bit tasvir)
# ==========================
MAX_I = 255

mse = np.linspace(1, 2000, 1000)
psnr = 10 * np.log10((MAX_I ** 2) / mse)

# ==========================
# Grafik
# ==========================
plt.figure(figsize=(10, 4.8))

plt.plot(mse, psnr, color="black", linewidth=1.8)

plt.xlabel("MSE", fontsize=13)
plt.ylabel("PSNR (dB)", fontsize=13)

plt.title(
    "PSNR va MSE orasidagi teskari logarifmik bog'lanish (8-bit tasvir uchun)",
    fontsize=14
)

# Grid
plt.grid(True, alpha=0.25)

# 30 va 40 dB chiziqlari
plt.axhline(40, color="gray", linestyle=":", linewidth=1.2)
plt.axhline(30, color="gray", linestyle="--", linewidth=1.2)

# Izohlar
plt.text(
    1200,
    40.7,
    "40 dB (odatda yuqori sifat)",
    fontsize=11,
    va="bottom"
)

plt.text(
    1200,
    30.7,
    "30 dB (odatda qoniqarli sifat chegarasi)",
    fontsize=11,
    va="bottom"
)

plt.text(
    1200,
    21,
    "odatda qoniqarsiz sifat",
    fontsize=11
)

plt.xlim(0, 2000)
plt.ylim(13.5, 49)

plt.tight_layout()

# ==========================
# Saqlash
# ==========================
output_file = "psnr_vs_mse.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Saqlandi: {output_file}")

plt.show()