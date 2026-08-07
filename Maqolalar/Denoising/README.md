# Multispektral tasvirlarda kanal darajasida moslashuvchan shovqinsizlantirish

Multispektral sun'iy yo'ldosh tasvirlarida shovqin turi va darajasini
**har bir spektral kanal uchun alohida** aniqlab, shu asosda
shovqinsizlantirishni amalga oshiruvchi tizim.

## O'rnatish

```bash
pip install numpy scipy scikit-learn scikit-image PyWavelets bm3d tifffile matplotlib
```

Linux'da tizim Python'i ishlatilsa `--break-system-packages` qo'shiladi.

## Ishga tushirish (buyruq satridan)

Eng oson yo'l — `run.py` skripti. Tif fayllar istalgan papkada tursa
bo'ladi, yo'lini argument sifatida berasiz:

```bash
python3 run.py rasmlar/River_1.tif          # bitta fayl
python3 run.py rasmlar/*.tif                # shablon
python3 run.py rasmlar/                     # butun papka
python3 run.py rasmlar/ -o natijalar/       # chiqish papkasini belgilash
python3 run.py rasmlar/River_1.tif --strategy hybrid
python3 run.py rasmlar/ --strategy hybrid --isolated-impulse
```

`--isolated-impulse` bayrog'i impulsivlikni yakkalik sharti bilan
aniqlaydi va haqiqiy tasvirlarda yolg'on ijobiylikni kamaytiradi.
`hybrid` rejimini haqiqiy L1C mahsulotlariga qo'llasangiz, uni
qo'shishni tavsiya etamiz.

Har bir kirish fayli uchun uchta natija hosil bo'ladi:

| Fayl | Mazmuni |
|---|---|
| `*_denoised.tif` | tozalangan tenzor, uint16 formatda (QGIS va SNAP ochadi) |
| `*_comparison.png` | original, tozalangan va olib tashlangan shovqin xaritasi |
| `*_report.txt` | kanal bo'yicha hisobot: turi, sigma*, usul, AQI |

Boshqa hech qanday tayyorgarlik qadami kerak emas — `run.py` ni
to'g'ridan-to'g'ri ishga tushiraverasiz.

Klassifikator faqat `hybrid` va `routing` rejimlarida kerak bo'ladi va
o'sha paytda avtomatik o'rgatilib, `clf.pkl` sifatida saqlanadi; keyingi
safar u qayta ishlatiladi. Standart `bm3d` rejimi tasniflashni butunlay
chetlab o'tadi, shuning uchun unda klassifikator umuman o'rgatilmaydi.

`full_pipeline.py` ni alohida ishga tushirish shart emas: u kutubxona
moduli bo'lib, `run_pipeline()` funksiyasini beradi. Uning `__main__`
bloki faqat namoyish uchun (uchala strategiyani sintetik ma'lumotda
solishtiradi) va hech narsa saqlamaydi.

## Tezkor boshlash (Python kodidan)

```python
import numpy as np, tifffile, pickle
from noise_type_classification import NoiseTypeClassifier, build_calibration_set
from full_pipeline import run_pipeline
from save_output import save_cube_tif, save_comparison_figure

# 1. Klassifikatorni o'rgatish (bir marta, keyin saqlab qo'yiladi)
X, y = build_calibration_set(n_scenes=120, size=96, n_bands=6)
clf = NoiseTypeClassifier(k=21).fit(X, y)
pickle.dump(clf, open("clf.pkl", "wb"))

# 2. Tasvirni yuklash: uint16 -> [0, 1]
cube = np.clip(tifffile.imread("River_1.tif").astype(np.float64) / 10000.0, 0, 1)

# 3. Ishga tushirish
res = run_pipeline(cube, clf, strategy="bm3d")

# 4. Saqlash
save_cube_tif(res["denoised"], "River_1_denoised.tif")
save_comparison_figure(cube, res["denoised"], "River_1_comparison.png")
```

## Qaysi strategiyani tanlash

| strategy | Qachon ishlatiladi |
|---|---|
| `"bm3d"` | **Haqiqiy Sentinel-2 L1C mahsulotlari uchun tavsiya etiladi.** Tasniflashni chetlab o'tadi, faqat kanal bo'yicha sigma* baholab BM3D qo'llaydi. |
| `"hybrid"` | Impulsiv buzilish haqiqatan kutilganda (xom mahsulot, eski arxivlar, uzatish xatoliklari). Sintetik ma'lumotda eng yuqori natija. |
| `"routing"` | Faqat qiyosiy tajribalar uchun. Amalda tavsiya etilmaydi. |

**Nima uchun haqiqiy ma'lumotga `bm3d`:** impulsivlik detektori L1C
mahsulotlarida ishonchsiz. 30 ta EuroSAT patchida o'rtacha impuls ulushi
0,105 chiqdi, bu sintetik haqiqatan impulsiv kanallardagi qiymatdan
(0,082) ham yuqori. Sabab: binolar, yo'llar va dala chegaralari median
qoldig'ida katta chetlashuv hosil qiladi. Natijada tasniflash 13
kanaldan 12 tasini xato ravishda impulsiv deb belgilashi mumkin.

## Chiqish

`run_pipeline()` lug'at qaytaradi:

| Kalit | Turi | Mazmuni |
|---|---|---|
| `denoised` | ndarray | tozalangan tenzor, kirish bilan bir xil shaklda |
| `band_types` | list | har bir kanalning aniqlangan turi |
| `band_sigmas` | ndarray | har bir kanal uchun baholangan sigma* |
| `band_denoisers` | list | qaysi kanalga qaysi usul qo'llanildi |
| `metrics` | dict | beshta xom sifat ko'rsatkichi (reference berilganda) |
| `aqi` | ndarray | kanal bo'yicha umumiy sifat balli, 0..1 |
| `low_quality_bands` | list | chegaradan past kanallar indekslari |
| `strategy` | str | ishlatilgan rejim |

## Fayllar tuzilishi

| Fayl | Vazifasi |
|---|---|
| `multispectral_data.py` | sintetik multispektrli sahna va sensor shovqini generatori |
| `noise_characterization.py` | NLF fiti (`v = a*mu + b`), impulsivlik, kurtoz |
| `noise_type_classification.py` | xususiyat vektori va k-NN klassifikator |
| `noise_level_estimation.py` | MAD asosida sigma* baholash, Anscombe transformatsiyasi |
| `adaptive_denoising.py` | denoiserlar va gibrid strategiya |
| `quality_feedback.py` | referenssiz sifat ko'rsatkichlari va AQI |
| `full_pipeline.py` | barcha bosqichlarni birlashtiruvchi asosiy modul |
| `evaluate.py` | maqoladagi sonli natijalarni ishlab chiqaruvchi tajribalar |
| `real_data_validation.py` | haqiqiy EuroSAT ma'lumotlarida tasdiqlash |
| `save_output.py` | natijani faylga saqlash va vizual solishtirish |
| `run.py` | buyruq satridan ishga tushirish skripti |

Har bir modul mustaqil ishga tushiriladigan `__main__` blokiga ega:

```bash
python3 multispectral_data.py      # generator namunasi
python3 noise_type_classification.py  # klassifikator aniqligi
python3 full_pipeline.py           # uchala strategiyani solishtirish
python3 real_data_validation.py    # EuroSAT tahlili
```

## Qayta ishlash oqimi

```
Multispektrli tenzor (H, W, C)
    |
    +-- har bir kanal uchun alohida:
    |     [1] xarakterlash:  v = a*mu + b, impulsivlik, kurtoz
    |     [2] k-NN tasniflash (kanal darajasida)
    |     [3] sigma* baholash (MAD, Puasson uchun Anscombe)
    |
    +-- [4] shovqinsizlantirish:
    |         impulsiv kanal -> median tuzatish -> BM3D
    |         qolganlari     -> BM3D
    |         sigma har bir kanalning o'z bahosidan
    |
    +-- [5] sifat ko'rsatkichlari (faqat tashxis, avtomatik qayta ishlash yo'q)
    |
    +-- tozalangan tenzor + kanal bo'yicha hisobot
```

## O'lchangan natijalar

Sintetik ma'lumotda (6 kanal, 96x96):

| Ko'rsatkich | Qiymat |
|---|---|
| Kanal darajasida tur aniqligi | 75,5% |
| sigma* baholash nisbiy xatoligi | 1,5-5,4% |
| Gibrid vs universal BM3D | +1,9 dan +8,0 dB |
| Gibrid vs tur-marshrutlash | +1,0 dan +1,2 dB |

Haqiqiy EuroSAT ma'lumotida (30 patch, 13 kanal):

| Tekshiruv | Natija |
|---|---|
| sigma* va nativ fazoviy ruxsat bog'liqligi | Spirmen rho = -0,851 (p = 0,0002) |
| Sinf ichidagi izchillik (ichki/umumiy tarqoqlik) | 0,42 |
| Impulsivlik detektori | ishonchsiz (yolg'on ijobiylik yuqori) |

## Ma'lum cheklovlar

**Impulsivlik detektori haqiqiy ma'lumotga o'tmaydi.** Yakkalik sharti
(`impulse_fraction_isolated`) yolg'on ijobiylikni 0,105 dan 0,038 ga
kamaytiradi, ammo bartaraf etmaydi.

**Past shovqinda `b` koeffitsiyenti oshib chiqadi.** sigma < 0,03
bo'lganda baho haqiqiy qiymatdan 2,5-3 barobar yuqori. Sabab: shovqin
dispersiyasi qoldiq struktura dispersiyasi bilan taqqoslanadigan
darajaga yetadi va ularni bitta tasvirdan ajratib bo'lmaydi.

**Sifat baholash moduli sintetik referensga tayanadi.**
`build_natural_reference()` sintetik toza sahnalardan statistika oladi,
shuning uchun haqiqiy tasvirlarda AQI qiymatlari past chiqadi. Uni
to'g'ri ishlatish uchun referensni haqiqiy buzilmagan tasvirlar
korpusidan qayta hisoblash kerak.

**Klassifikator sintetik ma'lumotda o'rgatiladi.** Haqiqiy tasvirda
shovqinning haqiqiy turi noma'lum bo'lgani uchun yorliq yo'q. Bu domen
siljishiga olib keladi.

## Kirish ma'lumotiga qo'yiladigan talablar

Qiymatlar **[0, 1] oralig'ida** bo'lishi shart. Sentinel-2 L1C uchun
xom `uint16` qiymatlar 10000 ga bo'linadi.

Processing Baseline 04.00 (2022-yil yanvaridan keyingi mahsulotlar)
uchun qo'shimcha `BOA_ADD_OFFSET` siljishini hisobga olish kerak;
metama'lumotdagi `QUANTIFICATION_VALUE` va offset qiymatlarini o'qing.
EuroSAT (2019) eski to'plam bo'lgani uchun oddiy bo'lish yetarli.