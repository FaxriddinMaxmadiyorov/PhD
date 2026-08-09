"""
1-rasm: uchta metrikaning xom qiymatlar diapazoni sezilarli darajada
farq qilishini ko'rsatuvchi grafik. Bu diapazon nomutanosibligi
normalizatsiyasiz o'rtachalashda nima uchun bitta metrika (bu holda
Laplasian energiyasi) natijani "yutib yuborishini" tushuntiradi.
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data.npz')
M1, M2, M3 = data['M1'], data['M2'], data['M3']

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
labels = ['M1: MAD\n(shovqin bahosi)', 'M2: Laplasian\nenergiyasi', 'M3: Shannon\nentropiyasi']
values = [M1, M2, M3]

for ax, vals, lab in zip(axes, values, labels):
    ax.boxplot(vals, patch_artist=True,
               boxprops=dict(facecolor='0.85', edgecolor='0.2'),
               medianprops=dict(color='0.1', linewidth=1.5),
               whiskerprops=dict(color='0.2'), capprops=dict(color='0.2'))
    ax.set_title(lab, fontsize=9)
    ax.set_xticks([])
    ax.set_yscale('log')
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', linestyle=':', color='0.7', linewidth=0.6, which='both')

fig.suptitle("Uch metrikaning xom qiymatlar diapazoni (logarifmik shkala, miqyoslar mos kelmaydi)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('fig1_metric_ranges.png', dpi=200)
print("saved fig1")