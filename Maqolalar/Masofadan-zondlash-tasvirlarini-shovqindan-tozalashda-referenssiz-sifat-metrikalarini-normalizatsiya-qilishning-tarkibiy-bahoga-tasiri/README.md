# Masofadan zondlash tasvirlarini shovqindan tozalashda referenssiz sifat metrikalarini normalizatsiya qilishning tarkibiy bahoga ta'siri

Ushbu papka maqoladagi tajribani qayta ishlab chiqarish (reproduce) uchun kerakli barcha skriptlarni o'z ichiga oladi.

## Maqsad

Bir nechta referenssiz (NR-IQA) sifat metrikasini (turlicha o'lchov birligi va diapazonga ega) oddiy o'rtacha bilan birlashtirganda, normalizatsiya bosqichisiz eng katta diapazonli metrika natijani "yutib yuborishini", normalizatsiyadan keyin esa barcha metrikalar teng hissa qo'shib, tarkibiy bahoning haqiqiy sifat (PSNR) bilan korrelyatsiyasi ortishini tajriba orqali ko'rsatish.

## Papka tuzilishi

```
.
├── README.md                  – ushbu fayl
├── experiment.py               – asosiy tajriba: metrikalarni hisoblash, normalizatsiya, korrelyatsiya
├── metrics_range.py            – 1-rasm: metrikalarning xom qiymatlar diapazoni (quti-mo'ylov diagrammasi)
├── correlation.py              – 2-rasm: normalizatsiyadan oldin/keyin PSNR bilan korrelyatsiya solishtiruvi
└── dataset/                     – 42 ta manba tasvir (21 urban + 21 rural, .png)
    ├── 1_urban.png
    ├── 1_rural.png
    └── ...
```

## Talablar

```bash
pip install numpy scipy matplotlib pillow --break-system-packages
```

Python 3.10+ tavsiya etiladi.

## Ishga tushirish tartibi

Barcha buyruqlar shu papka ichida, ketma-ket bajariladi:

```bash
python3 experiment.py       # data.npz va results.json hosil qiladi
python3 metrics_range.py    # fig1_metric_ranges.png hosil qiladi
python3 correlation.py      # fig2_correlation_comparison.png hosil qiladi
```

`experiment.py` avval ishga tushishi shart — u ikkinchi va uchinchi skript foydalanadigan `data.npz` va `results.json` fayllarini yaratadi.

## Metodologiya qisqacha

1. **Ma'lumot**: `dataset/` papkasidagi 42 ta tasvir kulrang shkalaga o'tkaziladi, no-data (qora) chekkalari avtomatik kesiladi, 512×512 o'lchamga keltiriladi.
2. **Shovqin**: har bir tasvirga 11 ta σ darajasida (2–40) additiv Gauss shovqini qo'shiladi, har bir (tasvir, σ) juftligi uchun 3 ta mustaqil realizatsiya olinadi → jami 42 × 11 × 3 = **1386 ta namuna**.
3. **Metrikalar** (har biri referenssiz, faqat shovqinli tasvirdan hisoblanadi):
   - **M1** – MAD asosidagi lokal shovqin bahosi
   - **M2** – Laplasian energiyasi (yuqori chastota)
   - **M3** – Shannon entropiyasi
4. **Referens**: PSNR (asl toza tasvir bilan shovqinli tasvir orasida) – faqat tekshirish/validatsiya uchun, metrikalarning o'ziga kiritilmaydi.
5. **Tarkibiy baho**: (a) xom qiymatlar o'rtachasi, (b) har bir metrika [0;1] ga normalizatsiya qilingandan keyingi o'rtacha. Ikkalasining PSNR bilan Pirson/Spirmen korrelyatsiyasi solishtiriladi.

## Asosiy natija

| Ko'rsatkich | |r| (PSNR bilan, Pirson) |
|---|---|
| M1 (MAD) | 0,924 |
| M2 (Laplasian) | 0,829 |
| M3 (Entropiya) | 0,788 |
| Tarkibiy baho (xom) | 0,830 |
| **Tarkibiy baho (normalizatsiya qilingan)** | **0,893** |

Normalizatsiyasiz tarkibiy baho amalda M2 bilan bir xil xatti-harakat namoyon etadi (chunki uning diapazoni eng katta); normalizatsiyadan keyin barcha uch metrika hisobga kiradi va korrelyatsiya sezilarli yaxshilanadi.

## Takrorlanuvchanlik

Tasodifiy son generatori qat'iy belgilangan (`np.random.default_rng(42)`), shuning uchun skriptni qayta ishga tushirish bir xil natijalarni beradi.

## Eslatma

`dataset/` papkasidagi tasvirlar sun'iy yo'ldosh tasviri emas, balki yuqori ajratilgan aerofotosuratlar – ular Sentinel-2/EuroSAT kabi masofadan zondlash tasvirlariga xos tekstura xilma-xilligini (shahar, qishloq xo'jaligi, suv, o'simlik qoplami) aks ettiruvchi test poligoni sifatida tanlangan.