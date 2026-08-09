"""
Tajriba: normalizatsiya/standartlashtirishning tasvir sifatini baholash
metrikalarini birlashtirishga ta'sirini ko'rsatish.

Uslubiyat:
1. Sintetik "ko'p to'qimali" bazaviy tasvir yaratiladi (masofadan zondlash
   tasvirlariga xos struktura: gradient + davriy tekstura + chekka bloklar).
2. Bazaviy tasvirga turli darajadagi additiv Gauss shovqini qo'shiladi
   (sigma = 2..40), natijada sifat pasayib boruvchi tasvirlar to'plami olinadi.
3. Har bir shovqinli tasvir uchun (referenssiz) uchta mustaqil metrika
   hisoblanadi:
     M1 - MAD asosidagi lokal shovqin bahosi (tipik diapazon: 0-40)
     M2 - Laplasian energiyasi (yuqori chastota energiyasi, diapazon: 0-6000)
     M3 - Shannon entropiyasi (diapazon: 0-8)
4. Referens sifat ko'rsatkichi sifatida asl (shovqinsiz) tasvirga nisbatan
   PSNR hisoblanadi (faqat tekshirish/validatsiya uchun, metrikalarning
   o'ziga kirmaydi).
5. Uchta metrikadan tarkibiy (composite) baho ikki usulda tuziladi:
     (a) xom qiymatlarni oddiy o'rtacha qilib (normalizatsiyasiz)
     (b) har bir metrikani [0;1] oralig'iga chiziqli normalizatsiya qilib,
         so'ngra o'rtacha olib
   Ikkala tarkibiy bahoning PSNR bilan Spirmen va Pirson korrelyatsiyasi
   hisoblanadi va solishtiriladi.
"""
"""
Tajriba: normalizatsiya/standartlashtirishning tasvir sifatini baholash
metrikalarini birlashtirishga ta'sirini ko'rsatish (REAL TASVIRLAR ASOSIDA).

Ma'lumotlar: 21 ta rural va 21 ta urban aerofotosurat (jami 42 ta, 1024x1024,
RGB), Tezis.zip dan olingan. Ba'zi tasvirlarda no-data (qora) chekkalar bor -
bular avtomatik aniqlanib kesib tashlanadi.

Uslubiyat:
1. Har bir tasvir kulrang shkalaga o'tkaziladi, qora chekkalar kesiladi,
   markazdan 512x512 o'lchamga keltiriladi (metrikalar taqqoslanuvchan
   bo'lishi uchun bir xil o'lchamga normallashtiriladi).
2. Har bir tasvirga turli darajadagi additiv Gauss shovqini qo'shiladi
   (sigma = 2..40), natijada sifat pasayib boruvchi tasvirlar to'plami olinadi.
3. Har bir shovqinli tasvir uchun (referenssiz) uchta mustaqil metrika
   hisoblanadi:
     M1 - MAD asosidagi lokal shovqin bahosi
     M2 - Laplasian energiyasi (yuqori chastota energiyasi)
     M3 - Shannon entropiyasi
4. Referens sifat ko'rsatkichi sifatida asl (shovqinsiz, lekin kesilgan/
   o'lchami moslashtirilgan) tasvirga nisbatan PSNR hisoblanadi.
5. Uchta metrikadan tarkibiy (composite) baho ikki usulda tuziladi:
     (a) xom qiymatlarni oddiy o'rtacha qilib (normalizatsiyasiz)
     (b) har bir metrikani datasetning o'zi bo'yicha [0;1] oralig'iga
         chiziqli normalizatsiya qilib, so'ngra o'rtacha olib
   Ikkala tarkibiy bahoning PSNR bilan Spirmen va Pirson korrelyatsiyasi
   hisoblanadi va solishtiriladi.
"""
import numpy as np
from scipy import stats
from scipy.signal import convolve2d
from PIL import Image
import glob, json, os

rng = np.random.default_rng(42)

IMG_DIR = 'dataset'
TARGET_SIZE = 512

# ---------- 1. Tasvirlarni yuklash va tayyorlash ----------
def load_and_prepare(path, size=TARGET_SIZE, black_thresh=5):
    im = np.array(Image.open(path).convert('L'), dtype=float)
    row_ok = im.mean(axis=1) > black_thresh
    col_ok = im.mean(axis=0) > black_thresh
    if row_ok.any() and col_ok.any():
        im = im[np.ix_(row_ok, col_ok)]
    h, w = im.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    im = im[top:top+s, left:left+s]
    im = np.array(Image.fromarray(im.astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=float)
    return im

image_files = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))
dataset = []
for f in image_files:
    label = 'urban' if 'urban' in os.path.basename(f) else 'rural'
    dataset.append(dict(path=f, label=label, base=load_and_prepare(f)))
print(f"Yuklandi: {len(dataset)} ta tasvir "
      f"({sum(1 for d in dataset if d['label']=='urban')} urban, "
      f"{sum(1 for d in dataset if d['label']=='rural')} rural)")

# ---------- 2. Shovqin darajalari ----------
sigmas = [2, 4, 6, 9, 12, 16, 20, 25, 30, 35, 40]
N_REALIZATIONS = 3  # har bir (tasvir, sigma) juftligi uchun

def add_noise(img, sigma):
    noisy = img + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255)

# ---------- 3. Referenssiz metrikalar ----------
def laplacian(img):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return convolve2d(img, k, mode='valid')

def mad_noise_estimate(img):
    lap = laplacian(img)
    return np.median(np.abs(lap - np.median(lap))) * 1.4826

def laplacian_energy(img):
    lap = laplacian(img)
    return float(np.var(lap))

def shannon_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)) * (255 / 256))

def psnr(clean, noisy):
    mse = np.mean((clean - noisy) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ---------- 4. Ma'lumotlarni yig'ish ----------
records = []
for d in dataset:
    base = d['base']
    for s in sigmas:
        for r_idx in range(N_REALIZATIONS):
            noisy = add_noise(base, s)
            m1 = mad_noise_estimate(noisy)
            m2 = laplacian_energy(noisy)
            m3 = shannon_entropy(noisy)
            p = psnr(base, noisy)
            records.append(dict(image=os.path.basename(d['path']), label=d['label'],
                                 sigma=s, realization=r_idx,
                                 M1_MAD=m1, M2_LapEnergy=m2, M3_Entropy=m3, PSNR=p))

print(f"Jami namunalar soni: {len(records)}")

M1 = np.array([r['M1_MAD'] for r in records])
M2 = np.array([r['M2_LapEnergy'] for r in records])
M3 = np.array([r['M3_Entropy'] for r in records])
PSNR = np.array([r['PSNR'] for r in records])

# ---------- 5. Tarkibiy baho: normalizatsiyasiz vs normalizatsiya bilan ----------
def minmax(v):
    return (v - v.min()) / (v.max() - v.min())

composite_raw = (M1 + M2 + M3) / 3.0
composite_norm = (minmax(M1) + minmax(M2) + minmax(M3)) / 3.0

def corr(a, b):
    sp = stats.spearmanr(a, b)
    pe = stats.pearsonr(a, b)
    pe_r = pe.statistic if hasattr(pe, 'statistic') else pe[0]
    return sp.correlation, pe_r

results = {}
for name, vec in [('M1_MAD', M1), ('M2_LapEnergy', M2), ('M3_Entropy', M3),
                   ('Composite_raw', composite_raw), ('Composite_normalized', composite_norm)]:
    sp, pe = corr(vec, PSNR)
    results[name] = dict(spearman_vs_PSNR=round(float(sp), 4), pearson_vs_PSNR=round(float(pe), 4),
                          min=round(float(vec.min()), 3), max=round(float(vec.max()), 3))

print(json.dumps(results, indent=2, ensure_ascii=False))

image_ids = {os.path.basename(d['path']): i for i, d in enumerate(dataset)}
IMG_ID = np.array([image_ids[r['image']] for r in records])

np.savez('data.npz',
          sigmas=np.array(sigmas), M1=M1, M2=M2, M3=M3, PSNR=PSNR,
          composite_raw=composite_raw, composite_norm=composite_norm,
          image_id=IMG_ID, n_images=len(dataset))

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(dict(n_images=len(dataset), n_samples=len(records), correlations=results), f,
               ensure_ascii=False, indent=2)