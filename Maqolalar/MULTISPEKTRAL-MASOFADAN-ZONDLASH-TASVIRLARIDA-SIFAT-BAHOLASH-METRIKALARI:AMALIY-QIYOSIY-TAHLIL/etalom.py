"""
Tasvir sifatini baholash metrikalarining tasnifi (blok-sxema).
Butun rasm kulrang shkalada chiziladi.

Ishga tushirish:  python tasnif_rasm.py
Natija:           rasm_tasnif.png  (joriy papkada)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

fig, ax = plt.subplots(figsize=(11.5, 4.2))
ax.axis('off')


def box(x, y, w, h, matn, fon='0.95', shrift=8.5, qalin=False):
    """Chetlari qora, foni kulrang to'rtburchak va uning ichidagi matn."""
    ax.add_patch(plt.Rectangle((x, y), w, h,
                               facecolor=fon, edgecolor='0.1', lw=1.3))
    ax.text(x + w/2, y + h/2, matn, ha='center', va='center',
            fontsize=shrift, fontweight='bold' if qalin else 'normal')


def strelka(x1, y1, x2, y2):
    """(x1,y1) dan (x2,y2) ga strelka."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='0.1', lw=1.2))


# ---------- 1-daraja: umumiy tushuncha ----------
box(0.31, 0.80, 0.38, 0.15, "Tasvir sifatini baholash metrikalari",
    fon='0.82', shrift=11, qalin=True)

# ---------- 2-daraja: ikki sinf ----------
box(0.02, 0.46, 0.525, 0.15,
    "Etalonli", fon='0.90', shrift=10)
box(0.60, 0.46, 0.370, 0.15,
    "Etalonsiz", fon='0.90', shrift=10)

strelka(0.45, 0.80, 0.283, 0.615)
strelka(0.55, 0.80, 0.785, 0.615)

# ---------- 3-daraja: etalonli metrikalar ----------
fr = [(0.020, 0.120, "Piksel darajasida\nMSE, PSNR"),
      (0.155, 0.120, "Strukturaviy\nSSIM, MS-SSIM"),
      (0.290, 0.100, "Fizik asosli\nMTF"),
      (0.405, 0.140, "MZ ga xos\nSAM, ERGAS, Q4")]
for x, w, matn in fr:
    box(x, 0.08, w, 0.20, matn, fon='1.0', shrift=8.5)
    strelka(0.283, 0.46, x + w/2, 0.285)

# ---------- 3-daraja: etalonsiz metrikalar ----------
nr = [(0.600, 0.170, "Tabiiy statistika\nBRISQUE, NIQE"),
      (0.790, 0.180, "Xususiyat asosli\ngradient, Furye, kontrast")]
for x, w, matn in nr:
    box(x, 0.08, w, 0.20, matn, fon='1.0', shrift=8.5)
    strelka(0.785, 0.46, x + w/2, 0.285)

ax.set_xlim(0, 1)
ax.set_ylim(0.05, 0.98)

plt.tight_layout()
plt.savefig('rasm_tasnif.png', dpi=140, bbox_inches='tight')
plt.close()

print("Saqlandi: rasm_tasnif.png")