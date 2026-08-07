import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm

# ==========================================
# Matplotlib sozlamalari
# ==========================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12
})

# ==========================================
# MSCN qiymatlari
# ==========================================

x = np.linspace(-3.5, 3.5, 1000)

# ==========================================
# Generalized Gaussian Distribution (GGD)
#
# beta (shape):
#   beta < 2  -> o'tkir cho'qqi, og'ir dumlar
#   beta = 2  -> normal taqsimot
#   beta > 2  -> yassi cho'qqi
# ==========================================

# Toza tasvir
beta_clean = 1.2
sigma_clean = 0.75
clean = gennorm.pdf(x, beta_clean, loc=0, scale=sigma_clean)

# Shovqin qo'shilgan
beta_noise = 1.8
sigma_noise = 0.90
noise = gennorm.pdf(x, beta_noise, loc=0, scale=sigma_noise)

# Xiralashgan
beta_blur = 2.8
sigma_blur = 0.75
blur = gennorm.pdf(x, beta_blur, loc=0, scale=sigma_blur)

# ==========================================
# Kurtoz (Pearson)
# ==========================================

def kurtosis(beta):
    from scipy.special import gamma
    return gamma(5/beta) * gamma(1/beta) / gamma(3/beta)**2

k_clean = kurtosis(beta_clean)
k_noise = kurtosis(beta_noise)
k_blur = kurtosis(beta_blur)

# ==========================================
# Plot
# ==========================================

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    x,
    clean,
    color="black",
    linewidth=2.6,
    label=f"Toza tasvir   (kurtoz = {k_clean:.2f})"
)

ax.plot(
    x,
    noise,
    color="0.35",
    linewidth=2.0,
    linestyle=(0,(6,2)),
    label=f"Shovqin qo'shilgan   (kurtoz = {k_noise:.2f})"
)

ax.plot(
    x,
    blur,
    color="0.65",
    linewidth=2.0,
    linestyle=(0,(2,2)),
    label=f"Xiralashtirilgan   (kurtoz = {k_blur:.2f})"
)

ax.set_xlim(-3.5,3.5)

ax.set_xlabel("MSCN koeffitsienti qiymati", fontsize=14)
ax.set_ylabel("Zichlik", fontsize=14)

ax.grid(alpha=0.25)

ax.legend(loc="upper right", frameon=True)

plt.tight_layout()

# ==========================================
# Save
# ==========================================

outfile = "mscn_ggd_distribution.png"

plt.savefig(
    outfile,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print(f"Saqlandi: {outfile}")

plt.show()