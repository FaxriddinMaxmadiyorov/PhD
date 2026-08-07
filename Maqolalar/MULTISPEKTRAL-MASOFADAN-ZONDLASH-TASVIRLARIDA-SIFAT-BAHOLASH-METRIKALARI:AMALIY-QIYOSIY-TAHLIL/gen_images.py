"""
CNN Maqolasi uchun 4 ta Rasm Generatsiyasi
============================================
Ishlatish:
    python generate_figures.py

Kerakli fayllar (bir papkada bo'lishi kerak):
    cnn_results.csv       — test natijalar (cnn_train.py chiqishi)
    cnn_loss_curve data   — to'g'ridan ma'lumot yo'q, quyida izoh bor

Chiqish fayllari:
    figure1_dataset.png   — Dataset generatsiya jarayoni
    figure2_architecture.png — CNN arxitektura blok-diagrammasi
    figure3_loss.png      — Train/val loss egri chiziqlari
    figure4_scatter.png   — Actual vs Predicted PSNR scatter plot

MUHIM: figure3 uchun cnn_train.py dan loss qiymatlarini
quyidagi o'zgaruvchilarga kiriting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import csv
import os

# ─────────────────────────────────────────────
# SOZLAMALAR
# ─────────────────────────────────────────────

# Figure 3 uchun loss qiymatlarini shu yerga kiriting
# (cnn_train.py natijasidan ko'chiring)
TRAIN_RMSE = [
    10.78, 10.01, 9.88, 9.74, 9.62,   # Faza 1 (freeze)
    8.13, 6.75, 6.31, 6.06, 5.81,      # Faza 2 (fine-tune)
    5.63, 5.50, 5.25, 5.12, 4.99,
    4.86, 4.77, 4.66, 4.60, 4.54,
]

VAL_RMSE = [
    10.16, 9.90, 9.72, 9.59, 9.58,    # Faza 1
    6.63, 6.24, 6.35, 5.60, 5.56,      # Faza 2
    5.35, 5.35, 5.37, 5.13, 5.53,
    5.05, 5.05, 4.98, 4.99, 4.98,
]

FREEZE_EPOCHS = 5   # Faza 1 nechta epoch

CSV_FILE = "cnn_results.csv"  # cnn_train.py chiqishi

# ─────────────────────────────────────────────
# JURNAL USLUBI
# ─────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# Grayscale ranglari
DARK   = '#1a1a1a'
MID    = '#555555'
LIGHT  = '#aaaaaa'
WHITE  = '#ffffff'
BOX_FILL = '#f0f0f0'
BOX_EDGE = '#333333'


# ═══════════════════════════════════════════════════════
# FIGURE 1 — Dataset generatsiya jarayoni
# ═══════════════════════════════════════════════════════

def figure1_dataset():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(xc, yc, w, h, lines, fill=BOX_FILL, edge=BOX_EDGE, lw=1.2):
        b = FancyBboxPatch((xc - w/2, yc - h/2), w, h,
                           boxstyle='round,pad=0.1', linewidth=lw,
                           edgecolor=edge, facecolor=fill)
        ax.add_patch(b)
        n = len(lines)
        for i, line in enumerate(lines):
            dy = (i - (n-1)/2) * 0.38
            weight = 'bold' if i == 0 else 'normal'
            fs = 10 if i == 0 else 9
            ax.text(xc, yc + dy, line, ha='center', va='center',
                    fontsize=fs, fontweight=weight, color=DARK)

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=DARK,
                                   lw=1.2, mutation_scale=12))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, label, fontsize=9, color=MID,
                    va='center', style='italic')

    # USTUN 1: EuroSAT (x=1.8)
    box(1.8, 8.5, 2.8, 1.1, ['EuroSAT Sentinel-2', '"Residential" sinfi'])
    arrow(1.8, 7.95, 1.8, 7.2)
    ax.text(2.05, 7.58, 'seed=42', fontsize=8.5, color=MID, style='italic')

    box(1.8, 6.75, 2.8, 1.0, ['3 000 ta tasvir', '64x64, 6 kanal'])
    arrow(1.8, 6.25, 1.8, 5.5)
    ax.text(2.05, 5.88, 'normalize', fontsize=8.5, color=MID, style='italic')

    box(1.8, 5.05, 2.8, 1.0, ['Normalizatsiya', '[0, 1], 2-98 persentil'])

    # O'Q: normalizatsiya → buzilish
    arrow(3.2, 5.05, 4.3, 5.05)
    ax.text(3.5, 5.38, 'x12 buzilish', fontsize=9, color=MID, style='italic')

    # USTUN 2: 12 buzilish turlari (x=6.1)
    dist_list = [
        'Gaussian (s=0.02-0.10)',
        'Gaussian blur (s=1-3)',
        'DCT siqish (20-60%)',
        'Chiziqli shovqin (s=0.02-0.05)',
        'Salt-and-pepper (p=0.02)',
    ]
    ax.text(6.1, 7.4, '12 buzilish turi:', ha='center',
            fontsize=10.5, fontweight='bold', color=DARK)
    for i, d in enumerate(dist_list):
        yy = 6.8 - i * 0.72
        b = FancyBboxPatch((4.4, yy - 0.28), 3.4, 0.56,
                           boxstyle='round,pad=0.05', linewidth=0.8,
                           edgecolor=MID, facecolor=WHITE)
        ax.add_patch(b)
        ax.text(6.1, yy, d, ha='center', va='center',
                fontsize=9, color=DARK)
        

    # # O'Q: buzilish → PSNR hisoblash
    # arrow(7.8, 5.05, 8.7, 5.05)
    # ax.text(8.25, 5.40, 'PSNR hisoblash', fontsize=8.5, color=MID, style='italic')
    # ax.text(8.25, 4.75, 'PSNR hisoblash', fontsize=8.5, color=MID, style='italic')

    # USTUN 3: 36 000 namuna (x=10.1) — o'ngga siljitildi
    box(10.1, 7.3, 2.0, 1.0, ['36 000', 'namuna'], lw=1.4)

    # Vertikal o'q: 36 000 → split
    arrow(10.1, 6.8, 10.1, 6.15)
    ax.text(10.35, 6.48, '70/15/15', fontsize=8.5, color=MID, style='italic')

    # Dataset split — alohida qatorga
    splits = [('Train 70%', '25 200'), ('Val 15%', '5 400'), ('Test 15%', '5 400')]
    xs = [8.8, 10.1, 11.4]
    for (label, n), xc in zip(splits, xs):
        b = FancyBboxPatch((xc - 0.6, 4.5), 1.2, 1.3,
                           boxstyle='round,pad=0.07', lw=0.9,
                           edgecolor=BOX_EDGE, facecolor=BOX_FILL)
        ax.add_patch(b)
        ax.text(xc, 5.25, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=DARK)
        ax.text(xc, 4.78, n, ha='center', va='center',
                fontsize=8.5, color=MID)
        # kichik o'q 36 000 dan split bloklarga
        ax.annotate('', xy=(xc, 5.8), xytext=(10.1, 6.12),
                    arrowprops=dict(arrowstyle='->', color=LIGHT,
                                   lw=0.7, mutation_scale=8))

    ax.set_title('1-rasm. Dataset generatsiya jarayoni',
                 fontsize=11.5, pad=10, color=DARK)
    plt.tight_layout()
    plt.savefig('figure1_dataset.png', facecolor=WHITE)
    plt.close()
    print("* figure1_dataset.png")


def figure2_architecture():
    fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    def block(xc, yc, w, h, lines, fill=BOX_FILL, edge=BOX_EDGE, lw=1.0):
        b = FancyBboxPatch((xc - w/2, yc - h/2), w, h,
                           boxstyle='round,pad=0.08', lw=lw,
                           edgecolor=edge, facecolor=fill)
        ax.add_patch(b)
        n = len(lines)
        for i, line in enumerate(lines):
            dy = (i - (n-1)/2) * 0.30
            weight = 'bold' if i == 0 else 'normal'
            fs = 9.5 if i == 0 else 8.5
            ax.text(xc, yc + dy, line, ha='center', va='center',
                    fontsize=fs, fontweight=weight, color=DARK)

    def arr(y1, y2, xc=3.0):
        ax.annotate('', xy=(xc, y2), xytext=(xc, y1),
                    arrowprops=dict(arrowstyle='->', color=DARK,
                                   lw=1.0, mutation_scale=10))

    xc = 3.0
    # Input
    block(xc, 10.0, 3.2, 0.7,
          ['Kirish', '6 × 64 × 64 (B2,B3,B4,B8,B11,B12)'],
          fill='#e8e8e8', lw=1.4)
    arr(9.65, 9.25)

    # Conv1
    block(xc, 9.0, 3.2, 0.6,
          ['Conv1 (6→64, 7×7) + BN + ReLU'],
          fill=BOX_FILL)
    arr(8.7, 8.35)

    # MaxPool
    block(xc, 8.1, 3.2, 0.45,
          ['MaxPool (3×3, stride=2)'],
          fill=WHITE, edge=MID, lw=0.8)
    arr(7.88, 7.45)

    # ResBlocks
    resblocks = [
        ('ResBlock ×2', '64 kanal', 7.15),
        ('ResBlock ×2', '128 kanal, stride=2', 6.35),
        ('ResBlock ×2', '256 kanal, stride=2', 5.55),
        ('ResBlock ×2', '512 kanal, stride=2', 4.75),
    ]
    for title, sub, yy in resblocks:
        block(xc, yy, 3.2, 0.6, [title, sub], fill=BOX_FILL)
        if yy > 4.75:
            arr(yy - 0.3, yy - 0.65)

    arr(4.45, 4.05)

    # GAP
    block(xc, 3.8, 3.2, 0.45,
          ['Global Average Pooling  →  512-d'],
          fill=WHITE, edge=MID, lw=0.8)

    # Separator
    ax.plot([0.5, 5.5], [3.53, 3.53], color=MID, lw=0.8, ls='--')
    ax.text(xc, 3.38, 'Regression head', ha='center', fontsize=8.5,
            color=MID, style='italic')

    arr(3.55, 3.1)

    # FC1
    block(xc, 2.85, 3.2, 0.6,
          ['FC (512 → 256) + ReLU'],
          fill=BOX_FILL, lw=1.0)
    arr(2.55, 2.2)

    # Dropout
    block(xc, 1.95, 3.2, 0.45,
          ['Dropout  (p = 0.3)'],
          fill=WHITE, edge=MID, lw=0.8)
    arr(1.73, 1.3)

    # FC2
    block(xc, 1.05, 3.2, 0.6,
          ['FC (256 → 1)'],
          fill=BOX_FILL, lw=1.0)
    arr(0.75, 0.2)

    # Output
    block(xc, -0.05, 3.2, 0.6,
          ['Bashorat qilingan PSNR (dB)'],
          fill='#e8e8e8', lw=1.4)

    ax.set_title('2-rasm. CNN arxitekturasi (ResNet18 asosi, moslashtirilgan)',
                 fontsize=10, pad=8, color=DARK)

    plt.tight_layout()
    plt.savefig('figure2_architecture.png', facecolor=WHITE)
    plt.close()
    print("✓ figure2_architecture.png")


# ═══════════════════════════════════════════════════════
# FIGURE 3 — Train/Val Loss egri chiziqlari
# ═══════════════════════════════════════════════════════

def figure3_loss():
    epochs = list(range(1, len(TRAIN_RMSE) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, TRAIN_RMSE, color=DARK, lw=1.8,
            marker='o', markersize=3, label='Train RMSE', zorder=3)
    ax.plot(epochs, VAL_RMSE, color=MID, lw=1.8,
            marker='s', markersize=3, linestyle='--',
            label='Validatsiya RMSE', zorder=3)

    # Faza chegarasi
    ax.axvline(FREEZE_EPOCHS + 0.5, color=LIGHT, lw=1.2,
               linestyle=':', zorder=2)
    ax.text(FREEZE_EPOCHS + 0.65, max(TRAIN_RMSE) * 0.92,
            'Fine-tune\nboshlandi',
            fontsize=8.5, color=MID, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc=WHITE,
                      ec=LIGHT, lw=0.8))

    # Eng yaxshi val RMSE belgisi
    best_val = min(VAL_RMSE)
    best_ep  = VAL_RMSE.index(best_val) + 1
    ax.annotate(f'Val RMSE = {best_val:.2f} dB\n(epoch {best_ep})',
                xy=(best_ep, best_val),
                xytext=(best_ep - 4, best_val + 1.5),
                fontsize=8.5, color=DARK,
                arrowprops=dict(arrowstyle='->', color=DARK, lw=0.9),
                bbox=dict(boxstyle='round,pad=0.3', fc=WHITE,
                          ec=LIGHT, lw=0.8))

    ax.set_xlabel('Epoch', labelpad=5)
    ax.set_ylabel('RMSE (dB)', labelpad=5)
    ax.set_title('3-rasm. Train va validatsiya RMSE egri chiziqlari',
                 pad=8)
    ax.legend(loc='upper right', framealpha=0.9,
              edgecolor=LIGHT, facecolor=WHITE)
    ax.set_xlim(0.5, len(epochs) + 0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('figure3_loss.png', facecolor=WHITE)
    plt.close()
    print("✓ figure3_loss.png")


# ═══════════════════════════════════════════════════════
# FIGURE 4 — Actual vs Predicted Scatter Plot
# ═══════════════════════════════════════════════════════

DISTORTION_MARKERS = {
    'Blur_s1':          ('o', 'Blur σ=1'),
    'Blur_s2':          ('o', 'Blur σ=2'),
    'Blur_s3':          ('o', 'Blur σ=3'),
    'Gaussian_s002':    ('s', 'Gaussian σ=0.02'),
    'Gaussian_s005':    ('s', 'Gaussian σ=0.05'),
    'Gaussian_s010':    ('s', 'Gaussian σ=0.10'),
    'Compress_20':      ('^', 'Compress 20%'),
    'Compress_40':      ('^', 'Compress 40%'),
    'Compress_60':      ('^', 'Compress 60%'),
    'Stripe_s002':      ('D', 'Stripe σ=0.02'),
    'Stripe_s005':      ('D', 'Stripe σ=0.05'),
    'SaltPepper_p002':  ('x', 'Salt-and-pepper'),
}

DIST_GROUPS = {
    'Blur':        ['Blur_s1', 'Blur_s2', 'Blur_s3'],
    'Gaussian':    ['Gaussian_s002', 'Gaussian_s005', 'Gaussian_s010'],
    'Compress':    ['Compress_20', 'Compress_40', 'Compress_60'],
    'Stripe':      ['Stripe_s002', 'Stripe_s005'],
    'Salt-pepper': ['SaltPepper_p002'],
}
GROUP_GRAY = {
    'Blur':        '#1a1a1a',
    'Gaussian':    '#555555',
    'Compress':    '#888888',
    'Stripe':      '#bbbbbb',
    'Salt-pepper': '#444444',
}


def figure4_scatter():
    if not os.path.exists(CSV_FILE):
        print(f"⚠  {CSV_FILE} topilmadi — figure4 o'tkazib yuborildi.")
        print("   cnn_train.py ni ishga tushiring, keyin qayta bajaring.")
        return

    data = {}
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                dist = row['distortion']
                actual = float(row['actual_psnr'])
                cnn    = float(row['cnn_predicted'])
                brisq  = float(row['brisque_predicted'])
                data.setdefault(dist, {'actual': [], 'cnn': [], 'brisque': []})
                data[dist]['actual'].append(actual)
                data[dist]['cnn'].append(cnn)
                data[dist]['brisque'].append(brisq)
            except (KeyError, ValueError):
                continue

    if not data:
        print(f"⚠  {CSV_FILE} bo'sh yoki noto'g'ri format.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, pred_key, title_suffix in [
        (ax1, 'cnn',     'CNN (ResNet18)  MAE = 2.98 dB'),
        (ax2, 'brisque', 'BRISQUE  MAE = 21.36 dB'),
    ]:
        all_actual, all_pred = [], []

        for group, dists in DIST_GROUPS.items():
            gc = GROUP_GRAY[group]
            marker_key = dists[0]
            marker = DISTORTION_MARKERS.get(marker_key, ('o', group))[0]

            act_all, pred_all = [], []
            for d in dists:
                if d in data:
                    act_all  += data[d]['actual']
                    pred_all += data[d][pred_key]

            if act_all:
                ax.scatter(act_all, pred_all,
                           marker=marker, color=gc, alpha=0.55,
                           s=15, label=group, linewidths=0.4,
                           edgecolors='none')
                all_actual += act_all
                all_pred   += pred_all

        if all_actual:
            mn = min(min(all_actual), min(all_pred))
            mx = max(max(all_actual), max(all_pred))
            pad = (mx - mn) * 0.04
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                    color=DARK, lw=1.0, ls='--', zorder=5,
                    label='Ideal (y=x)')
            ax.set_xlim(mn - pad, mx + pad)
            ax.set_ylim(mn - pad, mx + pad)

        ax.set_xlabel('Haqiqiy PSNR (dB)', labelpad=5)
        ax.set_ylabel('Bashorat PSNR (dB)', labelpad=5)
        ax.set_title(title_suffix, fontsize=10.5, pad=6)
        ax.set_aspect('equal', adjustable='box')
        if ax == ax1:
            ax.legend(loc='upper left', fontsize=8,
                    framealpha=0.9, edgecolor=LIGHT,
                    facecolor=WHITE, markerscale=1.4)

    fig.suptitle('4-rasm. Haqiqiy va bashorat qilingan PSNR taqqoslanishi',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig('figure4_scatter.png', facecolor=WHITE, bbox_inches='tight')
    plt.close()
    print("✓ figure4_scatter.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("4 ta rasm generatsiya qilinmoqda...")
    print("=" * 50)

    figure1_dataset()
    figure2_architecture()
    figure3_loss()
    figure4_scatter()

    print()
    print("Chiqish fayllari:")
    for f in ['figure1_dataset.png', 'figure2_architecture.png',
              'figure3_loss.png', 'figure4_scatter.png']:
        exists = "✓" if os.path.exists(f) else "✗ (topilmadi)"
        print(f"  {f}  {exists}")
    print()
    print("Eslatma: figure3 uchun TRAIN_RMSE va VAL_RMSE")
    print("  qiymatlarini cnn_train.py chiqishidan ko'chirib")
    print("  skript boshidagi o'zgaruvchilarga kiriting.")