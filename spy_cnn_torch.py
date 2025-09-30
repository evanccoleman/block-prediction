
#!/usr/bin/env python3

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ------------------------------
# Dataset (PNG folders -> tensors)
# ------------------------------
class PngFolderDataset(Dataset):
    """
    Expects a directory with structure produced by png_builder.store_pngs:
      size_<N>/
        classes.txt           # one integer per line (block sizes)
        label_<BLOCKSIZE>/
          matrix_0.png
          matrix_1.png
          ...
    Images are grayscale PNGs where white=0, black=1 (sparsity pattern).
    We keep them in [0,1] floats with shape (1,H,W).
    """
    def __init__(self, root: str, target_size: Tuple[int,int]=None):
        root = Path(root)
        if (root / "classes.txt").exists():
            self.base = root
        elif root.name.startswith("size_") and (root / "classes.txt").exists():
            self.base = root
        else:
            # try one more level down if the user pointed at png_datasetXX
            sizes = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("size_")]
            if not sizes:
                raise FileNotFoundError(f"Could not find classes.txt under {root}. Expected png_datasetXX/size_XX/...")
            self.base = sizes[0]  # pick the first size folder by default
        self.target_size = target_size

        # read class list (block sizes in ascending order)
        with open(self.base / "classes.txt","r") as f:
            self.class_list = [int(line.strip()) for line in f if line.strip()]
        self.label_to_idx = {v:i for i,v in enumerate(self.class_list)}

        # enumerate images
        self.samples = []  # list of (path, label_idx)
        for d in sorted(self.base.iterdir()):
            if d.is_dir() and d.name.startswith("label_"):
                blocksize = int(d.name.split("_")[1])
                if blocksize not in self.label_to_idx:
                    # allow unseen class; append
                    self.class_list.append(blocksize)
                    self.label_to_idx[blocksize] = len(self.class_list)-1
                idx = self.label_to_idx[blocksize]
                for png in sorted(d.glob("*.png")):
                    self.samples.append((png, idx))

        if not self.samples:
            raise RuntimeError(f"No PNGs found under {self.base}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        im = Image.open(path).convert("L")
        if self.target_size is not None:
            im = im.resize(self.target_size, resample=Image.NEAREST)
        # convert to tensor in [0,1]; invert so nonzeros=1, zeros=0 if needed
        x = torch.from_numpy((np.array(im, dtype="float32")/255.0)).unsqueeze(0)  # (1,H,W)
        return x, y

# ------------------------------
# Model (light ResNet-ish CNN)
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, k1=3, k2=3, l2=0.0):
        super().__init__()
        pad1 = k1//2
        pad2 = k2//2
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=k1, padding=pad1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=k2, padding=pad2, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.proj  = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act   = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out

class SpyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.block1 = ResidualBlock(32, 32, 64, k1=5, k2=3)
        self.block2 = ResidualBlock(64, 64, 128, k1=3, k2=3)
        self.block3 = ResidualBlock(128, 64, 128, k1=3, k2=3)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

# ------------------------------
# Train / Eval
# ------------------------------
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc  += (logits.argmax(dim=1) == y).float().sum().item()
        n += x.size(0)
    return total_loss/n, total_acc/n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc  += (logits.argmax(dim=1) == y).float().sum().item()
        n += x.size(0)
    return total_loss/n, total_acc/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Path to png_dataset*/size_* folder or its parent")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--target_hw", type=int, default=None,
                    help="Resize images to (H,W); if omitted, keep original")
    ap.add_argument("--out", type=str, default="/mnt/data/spy_cnn_torch.ckpt")
    args = ap.parse_args()

    dataset = PngFolderDataset(args.data_dir,
                               target_size=(args.target_hw, args.target_hw) if args.target_hw else None)
    num_classes = len(dataset.class_list)
    # train/val split
    n_val = int(len(dataset)*args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpyCNN(num_classes=num_classes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"[{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": dataset.class_list,
                "hw": dataset.samples and Image.open(dataset.samples[0][0]).size[::-1],  # (H,W)
            }, args.out)
            print(f"Saved best checkpoint to {args.out} (val_acc={best_val:.3f})")

if __name__ == "__main__":
    main()
