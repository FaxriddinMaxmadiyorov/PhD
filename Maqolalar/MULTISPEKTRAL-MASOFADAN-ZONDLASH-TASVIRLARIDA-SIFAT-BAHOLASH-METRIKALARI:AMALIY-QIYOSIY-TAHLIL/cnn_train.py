"""
CNN Classifier — Multispektral Tasvirlarda Buzilish Turini Aniqlash
====================================================================
ResNet18 asosida 5 sinf klassifikatsiya:
    0 — Gaussian shovqin
    1 — Blur
    2 — DCT siqish
    3 — Chiziqli shovqin (Stripe)
    4 — Salt-and-pepper

Ishlatish:
    python cnn_classifier.py

Dataset: dataset/EuroSATallBands/Residential/
Chiqish:
    classifier_model.pth       — saqlangan model
    classifier_results.txt     — accuracy, per-class natijalar
    classifier_confusion.png   — confusion matrix rasmi
    classifier_loss.png        — loss egri chiziqlari
"""

import random
import warnings
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# SOZLAMALAR
# ═══════════════════════════════════════════════

DATASET_FOLDER = "dataset/EuroSATallBands/Residential"
N_IMAGES       = 3000
SEED           = 42
BAND_IDX       = (1, 2, 3, 7, 10, 11)   # B2,B3,B4,B8,B11,B12

BATCH_SIZE     = 64       # Classifikatsiya uchun kattaroq batch
EPOCHS_FREEZE  = 5
EPOCHS_FINETUNE= 15
LR_HEAD        = 1e-3
LR_BACKBONE    = 1e-4
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════
# SINF NOMLARI
# ═══════════════════════════════════════════════

CLASS_NAMES = ['Gaussian', 'Blur', 'Compress', 'Stripe', 'Salt-pepper']

DIST_TO_CLASS = {
    'Gaussian_s002':    0,
    'Gaussian_s005':    0,
    'Gaussian_s010':    0,
    'Blur_s1':          1,
    'Blur_s2':          1,
    'Blur_s3':          1,
    'Compress_20':      2,
    'Compress_40':      2,
    'Compress_60':      2,
    'Stripe_s002':      3,
    'Stripe_s005':      3,
    'SaltPepper_p002':  4,
}

# ═══════════════════════════════════════════════
# BUZILISH FUNKSIYALARI
# ═══════════════════════════════════════════════

def add_gaussian(img, sigma):
    return np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32), 0, 1)

def add_blur(img, sigma):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma)
    return np.clip(out, 0, 1).astype(np.float32)

def add_compression(img, loss_pct):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        d = dctn(img[:, :, c].astype(np.float64), norm='ortho')
        d[np.abs(d) < np.percentile(np.abs(d), loss_pct)] = 0
        out[:, :, c] = np.clip(idctn(d, norm='ortho'), 0, 1)
    return out.astype(np.float32)

def add_stripe(img, sigma):
    out = img.copy()
    for c in range(img.shape[2]):
        out[:, :, c] = np.clip(
            out[:, :, c] + np.random.normal(0, sigma, (1, img.shape[1])).astype(np.float32), 0, 1)
    return out

def add_sp(img, prob):
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < prob / 2]     = 0.0
    out[mask > 1 - prob / 2] = 1.0
    return out

DISTORTION_LIST = [
    ('Gaussian_s002',   lambda img: add_gaussian(img, 0.02)),
    ('Gaussian_s005',   lambda img: add_gaussian(img, 0.05)),
    ('Gaussian_s010',   lambda img: add_gaussian(img, 0.10)),
    ('Blur_s1',         lambda img: add_blur(img, 1)),
    ('Blur_s2',         lambda img: add_blur(img, 2)),
    ('Blur_s3',         lambda img: add_blur(img, 3)),
    ('Compress_20',     lambda img: add_compression(img, 20)),
    ('Compress_40',     lambda img: add_compression(img, 40)),
    ('Compress_60',     lambda img: add_compression(img, 60)),
    ('Stripe_s002',     lambda img: add_stripe(img, 0.02)),
    ('Stripe_s005',     lambda img: add_stripe(img, 0.05)),
    ('SaltPepper_p002', lambda img: add_sp(img, 0.02)),
]

# ═══════════════════════════════════════════════
# DATASET YUKLASH
# ═══════════════════════════════════════════════

def load_eurosat(folder, n=3000, seed=42, band_idx=(1,2,3,7,10,11)):
    try:
        import tifffile
    except ImportError:
        print("pip install tifffile")
        raise

    folder = Path(folder)
    tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.TIF"))
    random.seed(seed)
    chosen = random.sample(tifs, min(n, len(tifs)))
    print(f"  {len(chosen)} ta tasvir yuklanmoqda...")

    images = []
    for p in chosen:
        try:
            raw = tifffile.imread(str(p)).astype(np.float32)
            if raw.ndim == 3 and raw.shape[0] < raw.shape[-1]:
                raw = np.transpose(raw, (1, 2, 0))
            if raw.ndim == 2:
                raw = raw[:, :, np.newaxis]
            if raw.shape[2] >= max(band_idx) + 1:
                bands = raw[:, :, list(band_idx)]
            elif raw.shape[2] >= 6:
                bands = raw[:, :, :6]
            else:
                bands = raw
            for b in range(bands.shape[2]):
                band = bands[:, :, b].astype(np.float64)
                p2, p98 = np.percentile(band, [2, 98])
                rng = p98 - p2
                if rng < 1.0:
                    mn, mx = float(band.min()), float(band.max())
                    rng2 = mx - mn
                    bands[:, :, b] = 0.0 if rng2 < 1e-6 else np.clip((band - mn) / rng2, 0, 1)
                else:
                    bands[:, :, b] = np.clip((band - p2) / rng, 0, 1)
            images.append(bands.astype(np.float32))
        except Exception as e:
            print(f"  Xato: {p.name}: {e}")

    print(f"  ✓ {len(images)} ta tasvir yuklandi")
    return images

# ═══════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════

class DistortionDataset(Dataset):
    """
    Har bir namuna: (buzilgan_tasvir_tensor [6,64,64], class_label int)
    """
    def __init__(self, samples, augment=False):
        self.samples = samples   # list of (img_np, class_id, dist_name)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, class_id, dist_name = self.samples[idx]
        if self.augment:
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=1).copy()
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=0).copy()
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        label  = torch.tensor(class_id, dtype=torch.long)
        return tensor, label, dist_name

# ═══════════════════════════════════════════════
# NAMUNALAR GENERATSIYASI
# ═══════════════════════════════════════════════

def generate_samples(images, seed=42):
    print("\nNamunalar generatsiya qilinmoqda...")
    samples = []
    for img_i, ref in enumerate(images):
        np.random.seed(seed + img_i)
        for dist_name, fn in DISTORTION_LIST:
            dist = fn(ref)
            class_id = DIST_TO_CLASS[dist_name]
            samples.append((dist, class_id, dist_name))
        if (img_i + 1) % 500 == 0:
            print(f"  {img_i + 1}/{len(images)} tasvir...")
    print(f"  ✓ Jami {len(samples)} namuna")
    return samples

def split_samples(samples, train_r=0.70, val_r=0.15, seed=42):
    by_dist = defaultdict(list)
    for s in samples:
        by_dist[s[2]].append(s)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for dist_name, lst in by_dist.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        train += lst[:n_train]
        val   += lst[n_train:n_train + n_val]
        test  += lst[n_train + n_val:]

    print(f"\nDataset: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test

# ═══════════════════════════════════════════════
# CNN MODELI — CLASSIFIER
# ═══════════════════════════════════════════════

class DistortionClassifier(nn.Module):
    def __init__(self, in_channels=6, num_classes=5, dropout=0.3):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")

        # 3 → 6 kanal
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight
            new_conv.weight /= 2.0
        backbone.conv1 = new_conv

        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )

        # Classifier head (regression emas, classification)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),   # 5 sinf
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)   # [batch, 5] — CrossEntropy uchun

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone muzlatildi")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  Backbone muzdan chiqarildi")

# ═══════════════════════════════════════════════
# O'QITISH
# ═══════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(imgs)
    return total_loss / total, correct / total

def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Faza 1
    print(f"\n{'='*55}")
    print(f"FAZA 1: Backbone muzlatilgan — {EPOCHS_FREEZE} epoch")
    print(f"{'='*55}")
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FREEZE)

    for epoch in range(1, EPOCHS_FREEZE + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)
        print(f"  Epoch {epoch:2d}/{EPOCHS_FREEZE}  "
              f"train_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}  "
              f"loss={tr_loss:.3f}/{vl_loss:.3f}")

    # Faza 2
    print(f"\n{'='*55}")
    print(f"FAZA 2: To'liq fine-tune — {EPOCHS_FINETUNE} epoch")
    print(f"{'='*55}")
    model.unfreeze_backbone()
    optimizer = optim.Adam([
        {'params': model.head.parameters(),     'lr': LR_HEAD},
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)
        print(f"  Epoch {epoch:2d}/{EPOCHS_FINETUNE}  "
              f"train_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}  "
              f"loss={tr_loss:.3f}/{vl_loss:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ✓ Eng yaxshi model (val_acc={vl_acc:.4f})")

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  ✓ Eng yaxshi model yuklandi (val_acc={best_val_acc:.4f})")

    return train_losses, val_losses, train_accs, val_accs

# ═══════════════════════════════════════════════
# BAHOLASH
# ═══════════════════════════════════════════════

def evaluate_test(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_dists = [], [], []
    with torch.no_grad():
        for imgs, labels, dnames in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_dists.extend(dnames)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).mean()

    # Per-class accuracy
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[name] = (all_preds[mask] == all_labels[mask]).mean()

    # Confusion matrix
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    return overall_acc, per_class, cm, all_preds, all_labels

# ═══════════════════════════════════════════════
# GRAFIKLAR
# ═══════════════════════════════════════════════

def plot_loss(train_losses, val_losses, train_accs, val_accs, freeze_ep):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(train_losses) + 1)

    for ax, tr, vl, ylabel, title in [
        (ax1, train_losses, val_losses, "Loss (CrossEntropy)", "Loss egri chiziqlari"),
        (ax2, train_accs,  val_accs,  "Accuracy",             "Aniqlik egri chiziqlari"),
    ]:
        ax.plot(epochs, tr, color='#1a1a1a', lw=1.8, marker='o',
                markersize=3, label='Train')
        ax.plot(epochs, vl, color='#555555', lw=1.8, marker='s',
                markersize=3, linestyle='--', label='Validatsiya')
        ax.axvline(freeze_ep + 0.5, color='#aaaaaa', lw=1.0,
                   linestyle=':', label='Fine-tune boshlandi')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9, edgecolor='#cccccc')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('classifier_loss.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ classifier_loss.png")

def plot_confusion(cm, class_names):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Greys', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Bashorat qilingan sinf", labelpad=8)
    ax.set_ylabel("Haqiqiy sinf", labelpad=8)
    ax.set_title("Confusion matrix (normallashtirilgan)", pad=10)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else '#1a1a1a'
            ax.text(j, i, f'{val:.2f}\n({cm[i,j]})',
                    ha='center', va='center',
                    fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig('classifier_confusion.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ classifier_confusion.png")

def save_results(overall_acc, per_class, cm):
    lines = [
        "=" * 55,
        "BUZILISH TURI KLASSIFIKATSIYA NATIJALARI",
        "=" * 55,
        "",
        f"Umumiy aniqlik (Overall Accuracy): {overall_acc:.4f}  ({overall_acc*100:.2f}%)",
        "",
        "Sinf bo'yicha aniqlik:",
        "-" * 40,
    ]
    for name, acc in per_class.items():
        lines.append(f"  {name:<15}  {acc:.4f}  ({acc*100:.1f}%)")

    lines += [
        "",
        "Confusion matrix (absolut qiymatlar):",
        "-" * 40,
        "         " + "  ".join(f"{n[:6]:>6}" for n in CLASS_NAMES),
    ]
    for i, row in enumerate(cm):
        lines.append(f"  {CLASS_NAMES[i]:<8}" + "  ".join(f"{v:>6}" for v in row))

    txt = "\n".join(lines)
    print("\n" + txt)

    with open("classifier_results.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n✓ classifier_results.txt")

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    print("=" * 55)
    print("CNN Buzilish Turi Klassifikatoru")
    print(f"Dataset: {DATASET_FOLDER}")
    print(f"Device:  {DEVICE}")
    print("=" * 55)

    # 1. Dataset
    print("\n1. Dataset yuklanmoqda...")
    images = load_eurosat(DATASET_FOLDER, n=N_IMAGES,
                          seed=SEED, band_idx=BAND_IDX)

    # 2. Namunalar
    print("\n2. Namunalar generatsiya qilinmoqda...")
    samples = generate_samples(images, seed=SEED)

    # Sinf taqsimlanishini ko'rsatish
    print("\nSinf taqsimlanishi:")
    class_counts = defaultdict(int)
    for _, class_id, _ in samples:
        class_counts[CLASS_NAMES[class_id]] += 1
    for name, count in class_counts.items():
        print(f"  {name:<15} {count:>6} namuna")

    # 3. Split
    print("\n3. Dataset bo'linmoqda...")
    train_s, val_s, test_s = split_samples(
        samples, TRAIN_RATIO, VAL_RATIO, SEED)

    # 4. DataLoader
    train_ds = DistortionDataset(train_s, augment=True)
    val_ds   = DistortionDataset(val_s,   augment=False)
    test_ds  = DistortionDataset(test_s,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # 5. Model
    print("\n4. Model yaratilmoqda...")
    model = DistortionClassifier(
        in_channels=6, num_classes=len(CLASS_NAMES), dropout=DROPOUT
    ).to(DEVICE)
    print(f"  Parametrlar: {sum(p.numel() for p in model.parameters()):,}")

    # 6. O'qitish
    print("\n5. O'qitish boshlanmoqda...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, DEVICE)

    # 7. Model saqlash
    torch.save(model.state_dict(), "classifier_model.pth")
    print("\n✓ classifier_model.pth saqlandi")

    # 8. Test
    print("\n6. Test to'plami baholanmoqda...")
    overall_acc, per_class, cm, _, _ = evaluate_test(
        model, test_loader, DEVICE)

    # 9. Natijalar
    print("\n7. Natijalar saqlanmoqda...")
    plot_loss(train_losses, val_losses, train_accs, val_accs, EPOCHS_FREEZE)
    plot_confusion(cm, CLASS_NAMES)
    save_results(overall_acc, per_class, cm)

    print("\n" + "=" * 55)
    print("YAKUNIY NATIJA:")
    print(f"  Umumiy aniqlik: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print("=" * 55)
    print("\nChiqish fayllari:")
    print("  classifier_model.pth      — saqlangan model")
    print("  classifier_results.txt    — aniqlik natijalari")
    print("  classifier_confusion.png  — confusion matrix")
    print("  classifier_loss.png       — loss/accuracy grafiklar")


if __name__ == "__main__":
    main()