
#!/usr/bin/env python3
# Multi-task SPY-image CNN (PyTorch) with optional tiling and regression head.
# - Heads: (a) per-row block-start probabilities (length H), (b) global classification over block sizes,
#          (c) global regression in [0,1] = block_size / n.
# - Dataset expects png_builder-like structure: .../size_<N>/{classes.txt,label_<bs>/*.png}
# - If per-row labels are not provided, we synthesize a uniform block-start vector from the class label.

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import os
import math
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------
# Helpers
# ------------------------------
def parse_size_from_path(p: Path) -> Optional[int]:
    for part in p.parts[::-1]:
        if part.startswith("size_"):
            try:
                return int(part.split("_")[1])
            except:
                pass
    return None

def make_uniform_row_starts(n: int, block_size: int) -> np.ndarray:
    """Row-start indicator vector of length n with 1 at rows 0, bs, 2bs, ..."""
    y = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        y[pos] = 1.0
        pos += block_size
    return y

def downsample_row_vector(v: np.ndarray, target_len: int) -> np.ndarray:
    """Average-pool a 1D vector to target length."""
    n = len(v)
    if target_len == n:
        return v.astype(np.float32)
    # map each target index to a range of source indices
    out = np.zeros(target_len, dtype=np.float32)
    for t in range(target_len):
        a = int(math.floor(t * n / target_len))
        b = int(math.ceil((t+1) * n / target_len))
        b = max(b, a+1)
        out[t] = v[a:b].mean()
    return out

def upsample_row_vector(v: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolate a 1D vector to target length."""
    n = len(v)
    if target_len == n:
        return v.astype(np.float32)
    xs = np.linspace(0, 1, n, endpoint=True)
    xt = np.linspace(0, 1, target_len, endpoint=True)
    return np.interp(xt, xs, v).astype(np.float32)

# ------------------------------
# Dataset
# ------------------------------
class PngFolderDatasetMT(Dataset):
    """
    Multitask dataset for SPY images.
    - Loads grayscale PNGs as (1,H,W) tensors.
    - Targets:
        * class_idx (global block size classification)
        * reg_frac  = block_size / n  (scalar in [0,1])
        * row_start_down (length target_hw)  -- synthesized if not provided
    Optional per-row labels: if a .npy with same basename + ".rowstarts.npy" exists under the PNG, we load it
    instead of synthesizing uniform starts.
    """
    def __init__(self, root: str, target_hw: int = 256, use_tiles: bool = False, tile_hw: int = 256,
                 tiles_per_image: int = 0, tile_stride: int = 128, rng_seed: int = 42):
        self.base = Path(root)
        # allow pointing to the dataset parent
        if not (self.base / "classes.txt").exists():
            # find a size_* folder
            cand = [p for p in self.base.iterdir() if p.is_dir() and p.name.startswith("size_")]
            if not cand:
                raise FileNotFoundError(f"Could not find classes.txt under {self.base}")
            self.base = cand[0]

        with open(self.base / "classes.txt","r") as f:
            self.class_list = [int(line.strip()) for line in f if line.strip()]
        self.label_to_idx = {v:i for i,v in enumerate(self.class_list)}

        self.samples = []  # (png_path, class_idx, n)
        n_from_path = parse_size_from_path(self.base)
        for d in sorted(self.base.iterdir()):
            if d.is_dir() and d.name.startswith("label_"):
                bs = int(d.name.split("_")[1])
                idx = self.label_to_idx.get(bs, None)
                if idx is None:
                    self.class_list.append(bs); idx = len(self.class_list)-1
                    self.label_to_idx[bs] = idx
                for png in sorted(d.glob("*.png")):
                    self.samples.append((png, idx, n_from_path))

        self.target_hw = int(target_hw)
        self.use_tiles = use_tiles
        self.tile_hw = int(tile_hw)
        self.tiles_per_image = int(tiles_per_image)
        self.tile_stride = int(tile_stride)
        self.rng = random.Random(rng_seed)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        im = Image.open(path).convert("L")
        return np.array(im, dtype=np.float32) / 255.0  # (H,W) in [0,1]

    def _maybe_rowstarts(self, path: Path, n: int, block_size: int) -> np.ndarray:
        cand = path.with_suffix("").as_posix() + ".rowstarts.npy"
        cand = Path(cand)
        if cand.exists():
            v = np.load(cand)
            if n is not None and len(v) != n:
                # trust the file but warn
                pass
            return v.astype(np.float32)
        # synthesize uniform starts
        if n is None:
            # fallback: assume image height equals n
            n = Image.open(path).size[1]
        return make_uniform_row_starts(n, block_size)

    def _tile_coords(self, H: int, W: int) -> List[Tuple[int,int,int,int]]:
        th = self.tile_hw; tw = self.tile_hw
        coords = []
        for y in range(0, max(1, H-th+1), self.tile_stride):
            for x in range(0, max(1, W-tw+1), self.tile_stride):
                coords.append((y, x, min(y+th, H), min(x+tw, W)))
        if (H, W) not in [(th, tw)]:
            coords.append((max(0, H-th), max(0, W-tw), H, W))  # ensure coverage of bottom-right
        return coords

    def __getitem__(self, i: int):
        path, class_idx, n = self.samples[i]
        bs = None
        # recover block size from label_<bs>
        try:
            bs = int(path.parent.name.split("_")[1])
        except:
            pass
        img = self._load_image(path)  # (H,W)
        H, W = img.shape
        # compute regression fraction
        reg_frac = 0.0
        if n is None:
            n = H
        if bs is not None and n is not None and n > 0:
            reg_frac = float(bs) / float(n)
        rowstarts_full = self._maybe_rowstarts(path, n, bs if bs is not None else max(1,n//8))
        # choose between tiling or resize full
        if self.use_tiles and self.tiles_per_image > 0:
            # pick a random tile
            coords = self._tile_coords(H, W)
            y0,x0,y1,x1 = self.rng.choice(coords)
            tile = img[y0:y1, x0:x1]
            # pad to tile_hw
            th, tw = self.tile_hw, self.tile_hw
            pad = np.zeros((th, tw), dtype=np.float32)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            x = torch.from_numpy(pad).unsqueeze(0)  # (1,th,tw)
            # row label: slice rows y0:y1 and downsample to tile_hw
            row_slice = rowstarts_full[y0:y1]
            row_down = downsample_row_vector(row_slice, th)
            y_row = torch.from_numpy(row_down)  # (th,)
            target_hw = th
        else:
            # resize full image to (target_hw, target_hw)
            tgt = self.target_hw
            im = Image.fromarray((img*255.0).astype("uint8"))
            im = im.resize((tgt, tgt), resample=Image.NEAREST)
            arr = np.array(im, dtype=np.float32)/255.0
            x = torch.from_numpy(arr).unsqueeze(0)  # (1,tgt,tgt)
            # downsample row labels from length n (or H) to target_hw
            base_len = len(rowstarts_full)
            if base_len != tgt:
                y_row = torch.from_numpy(downsample_row_vector(rowstarts_full, tgt))
            else:
                y_row = torch.from_numpy(rowstarts_full.copy())
            target_hw = tgt

        y_cls = torch.tensor(class_idx, dtype=torch.long)
        y_reg = torch.tensor([reg_frac], dtype=torch.float32)
        return x, y_cls, y_reg, y_row, target_hw

# ------------------------------
# Model (backbone + 3 heads)
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, k1=3, k2=3):
        super().__init__()
        pad1 = k1//2; pad2 = k2//2
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

class SpyCNNMultiTask(nn.Module):
    def __init__(self, num_classes: int, row_vec_len: int = 256):
        super().__init__()
        self.row_vec_len = row_vec_len
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.block1 = ResidualBlock(32, 32, 64, k1=5, k2=3)
        self.block2 = ResidualBlock(64, 64, 128, k1=3, k2=3)
        self.block3 = ResidualBlock(128, 64, 128, k1=3, k2=3)
        # classification & regression heads use global pooled features
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.head_cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.head_reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in [0,1]
        )
        # per-row head: collapse width, keep height -> vector, then upsample to row_vec_len
        self.row_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # (B,1,h',w')
        )
        self.row_act = nn.Identity()  # we'll use BCEWithLogitsLoss -> raw logits
    def forward(self, x):
        # x: (B,1,H,W)
        z = self.stem(x)
        z = F.max_pool2d(self.block1(z), 2)   # (B,64,H/2,W/2)
        z = F.max_pool2d(self.block2(z), 2)   # (B,128,H/4,W/4)
        z = self.block3(z)                    # (B,128,H/4,W/4)
        # global pooled features
        g = self.pool(z)                      # (B,128,1,1)
        logits_cls = self.head_cls(g)
        reg_frac   = self.head_reg(g)
        # per-row logits: conv -> average over width -> upsample to row_vec_len
        r = self.row_conv(z)                  # (B,1,h',w')
        r = r.mean(dim=3)                     # (B,1,h')
        r = r.squeeze(1)                      # (B,h')
        if r.size(1) != self.row_vec_len:
            r = F.interpolate(r.unsqueeze(1), size=(self.row_vec_len,), mode="linear", align_corners=False).squeeze(1)
        row_logits = self.row_act(r)          # (B,row_vec_len)
        return logits_cls, reg_frac, row_logits

# ------------------------------
# Train / Eval
# ------------------------------
def train_epoch(model, loader, opt, device, lw_row=1.0, lw_cls=0.5, lw_reg=0.5):
    model.train()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=0.02)
    total = {"loss":0.0, "row":0.0, "cls":0.0, "reg":0.0, "acc":0.0}; n = 0
    for x,y_cls,y_reg,y_row,row_len in loader:
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg = y_reg.to(device)
        y_row = y_row.to(device)
        logits_cls, reg_frac, row_logits = model(x)
        loss_row = bce(row_logits, y_row)
        loss_cls = ce(logits_cls, y_cls)
        loss_reg = huber(reg_frac.squeeze(1), y_reg.squeeze(1))
        loss = lw_row*loss_row + lw_cls*loss_cls + lw_reg*loss_reg
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (logits_cls.argmax(dim=1) == y_cls).float().mean().item()
        bs = x.size(0)
        total["loss"] += loss.item()*bs
        total["row"]  += loss_row.item()*bs
        total["cls"]  += loss_cls.item()*bs
        total["reg"]  += loss_reg.item()*bs
        total["acc"]  += acc*bs
        n += bs
    for k in total: total[k] /= max(1,n)
    return total

@torch.no_grad()
def eval_epoch(model, loader, device, lw_row=1.0, lw_cls=0.5, lw_reg=0.5):
    model.eval()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=0.02)
    total = {"loss":0.0, "row":0.0, "cls":0.0, "reg":0.0, "acc":0.0}; n = 0
    for x,y_cls,y_reg,y_row,row_len in loader:
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg = y_reg.to(device)
        y_row = y_row.to(device)
        logits_cls, reg_frac, row_logits = model(x)
        loss_row = bce(row_logits, y_row)
        loss_cls = ce(logits_cls, y_cls)
        loss_reg = huber(reg_frac.squeeze(1), y_reg.squeeze(1))
        loss = lw_row*loss_row + lw_cls*loss_cls + lw_reg*loss_reg
        with torch.no_grad():
            acc = (logits_cls.argmax(dim=1) == y_cls).float().mean().item()
        bs = x.size(0)
        total["loss"] += loss.item()*bs
        total["row"]  += loss_row.item()*bs
        total["cls"]  += loss_cls.item()*bs
        total["reg"]  += loss_reg.item()*bs
        total["acc"]  += acc*bs
        n += bs
    for k in total: total[k] /= max(1,n)
    return total

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--target_hw", type=int, default=256)
    ap.add_argument("--use_tiles", action="store_true")
    ap.add_argument("--tile_hw", type=int, default=256)
    ap.add_argument("--tile_stride", type=int, default=128)
    ap.add_argument("--tiles_per_image", type=int, default=0, help="If >0, dataset samples random tiles per image")
    ap.add_argument("--loss_weights", type=str, default="row=1.0,class=0.5,reg=0.5")
    ap.add_argument("--out", type=str, default="/mnt/data/spy_multitask_tiled.ckpt")
    args = ap.parse_args()

    lw = {"row":1.0,"class":0.5,"reg":0.5}
    for part in args.loss_weights.split(","):
        if not part: continue
        k,v = part.split("="); lw[k.strip()] = float(v)

    dataset = PngFolderDatasetMT(args.data_dir, target_hw=args.target_hw,
                                 use_tiles=args.use_tiles, tile_hw=args.tile_hw,
                                 tiles_per_image=args.tiles_per_image, tile_stride=args.tile_stride)

    num_classes = len(dataset.class_list)
    model = SpyCNNMultiTask(num_classes=num_classes, row_vec_len=args.target_hw)

    # split dataset
    n_val = int(len(dataset)*args.val_split); n_train = len(dataset)-n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1e9
    for epoch in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, device, lw_row=lw["row"], lw_cls=lw["class"], lw_reg=lw["reg"])
        va = eval_epoch(model, val_loader, device, lw_row=lw["row"], lw_cls=lw["class"], lw_reg=lw["reg"])
        print(f"[{epoch:03d}] tr: loss {tr['loss']:.4f} row {tr['row']:.4f} cls {tr['cls']:.4f} reg {tr['reg']:.4f} acc {tr['acc']:.3f} | "
              f"va: loss {va['loss']:.4f} row {va['row']:.4f} cls {va['cls']:.4f} reg {va['reg']:.4f} acc {va['acc']:.3f}")
        score = -va["loss"]  # maximize
        if score > best:
            best = score
            torch.save({
                "model_state": model.state_dict(),
                "classes": dataset.class_list,
                "row_vec_len": args.target_hw,
                "args": vars(args)
            }, args.out)
            print(f"Saved checkpoint -> {args.out}")

if __name__ == "__main__":
    main()
