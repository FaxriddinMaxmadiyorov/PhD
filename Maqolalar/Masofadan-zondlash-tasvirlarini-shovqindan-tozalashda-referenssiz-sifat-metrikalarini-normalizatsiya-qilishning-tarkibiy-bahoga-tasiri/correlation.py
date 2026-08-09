"""
2-rasm: har bir alohida metrika, normalizatsiyasiz tarkibiy baho va
normalizatsiya qilingan tarkibiy bahoning referens PSNR bilan Pirson
korrelyatsiyasini (mutlaq qiymatda) solishtiruvchi ustunli diagramma.
"""
import json
import matplotlib.pyplot as plt

with open('results.json', encoding='utf-8') as f:
    res = json.load(f)['correlations']

labels = ['M1\n(MAD)', 'M2\n(Laplasian)', 'M3\n(Entropiya)',
          'Tarkibiy\n(xom)', 'Tarkibiy\n(normalizatsiya)']
keys = ['M1_MAD', 'M2_LapEnergy', 'M3_Entropy', 'Composite_raw', 'Composite_normalized']
values = [abs(res[k]['pearson_vs_PSNR']) for k in keys]
colors = ['0.75', '0.75', '0.75', '0.45', '0.15']

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(labels, values, color=colors, edgecolor='0.1', linewidth=0.8)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
            ha='center', va='bottom', fontsize=9)

ax.set_ylabel("|Pirson korrelyatsiyasi| (PSNR bilan)", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("Normalizatsiyaning tarkibiy bahoga ta'siri", fontsize=10)
ax.grid(axis='y', linestyle=':', color='0.7', linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('fig2_correlation_comparison.png', dpi=200)
print("saved fig2")