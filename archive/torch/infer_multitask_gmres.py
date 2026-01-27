
#!/usr/bin/env python3
# Inference & GMRES utilities for the multi-task model, including tiling & stitching.

import argparse, math
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix, bmat
from scipy.sparse.linalg import gmres as sp_gmres, LinearOperator, splu
from scipy.io import mmread

# ---- Model (must match training) ----
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
            nn.Sigmoid()
        )
        self.row_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, x):
        z = self.stem(x)
        z = F.max_pool2d(self.block1(z), 2)
        z = F.max_pool2d(self.block2(z), 2)
        z = self.block3(z)
        g = self.pool(z)
        logits_cls = self.head_cls(g)
        reg_frac   = self.head_reg(g)
        r = self.row_conv(z).mean(dim=3).squeeze(1)  # (B,h')
        if r.size(1) != self.row_vec_len:
            r = F.interpolate(r.unsqueeze(1), size=(self.row_vec_len,), mode="linear", align_corners=False).squeeze(1)
        return logits_cls, reg_frac, r  # r are logits

def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    row_len = ckpt.get("row_vec_len", 256)
    model = SpyCNNMultiTask(num_classes=len(classes), row_vec_len=row_len)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, classes, row_len

# ---- Image IO & tiling ----
def load_png_gray(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.float32)/255.0  # (H,W)

def tiles(H:int, W:int, tile_hw:int, stride:int) -> List[Tuple[int,int,int,int]]:
    coords=[]
    th=tile_hw; tw=tile_hw
    for y in range(0, max(1,H-th+1), stride):
        for x in range(0, max(1,W-tw+1), stride):
            coords.append((y,x,min(y+th,H),min(x+tw,W)))
    if (H,W)!=(th,tw):
        coords.append((max(0,H-th),max(0,W-tw),H,W))
    return coords

@torch.no_grad()
def infer_full_or_tiled(model, img: np.ndarray, target_hw:int=256, use_tiles:bool=False, tile_hw:int=256, tile_stride:int=128):
    H,W = img.shape
    device = next(model.parameters()).device
    if not use_tiles:
        im = Image.fromarray((img*255).astype("uint8")).resize((target_hw,target_hw), resample=Image.NEAREST)
        x = torch.from_numpy(np.array(im,dtype=np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
        logits_cls, reg_frac, row_logits = model(x)
        return logits_cls.squeeze(0).cpu().numpy(), reg_frac.item(), row_logits.squeeze(0).cpu().numpy()
    # tiled
    acc_rows = np.zeros(H, dtype=np.float32)
    acc_counts = np.zeros(H, dtype=np.float32)
    logits_accum = None; logits_count = 0
    reg_sum = 0.0
    for (y0,x0,y1,x1) in tiles(H,W,tile_hw,tile_stride):
        tile = np.zeros((tile_hw,tile_hw),dtype=np.float32)
        tile[:(y1-y0),:(x1-x0)] = img[y0:y1,x0:x1]
        im = Image.fromarray((tile*255).astype("uint8"))
        x = torch.from_numpy(np.array(im,dtype=np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(device)
        logits_cls, reg_frac, row_logits = model(x)  # row_logits length = tile_hw
        row_vec = torch.sigmoid(row_logits).squeeze(0).cpu().numpy()  # (tile_hw,)
        acc_rows[y0:y1] += row_vec[:(y1-y0)]
        acc_counts[y0:y1] += 1.0
        lc = logits_cls.squeeze(0).cpu().numpy()
        logits_accum = lc if logits_accum is None else (logits_accum + lc)
        logits_count += 1
        reg_sum += reg_frac.item()
    acc_counts[acc_counts==0]=1.0
    row_full = acc_rows/acc_counts
    logits_mean = logits_accum / max(1,logits_count)
    reg_mean = reg_sum / max(1,logits_count)
    # Also produce a resized row vector at target_hw if desired by caller
    row_resized = row_full
    if H != target_hw:
        row_resized = np.interp(np.linspace(0,1,target_hw), np.linspace(0,1,H), row_full).astype(np.float32)
    return logits_mean, reg_mean, row_resized

# ---- Convert row-start probabilities to blocks ----
def threshold_row_starts(p: np.ndarray, min_block:int=1, hysteresis:float=0.2) -> List[int]:
    """Turn probabilities into start flags, then convert to block sizes.
       hysteresis: extra requirement to start a new block if previous prob was high; simple smoothing."""
    q = (p > 0.5).astype(np.int32)
    # ensure first is a start
    q[0]=1
    # remove too-dense starts by enforcing min_block
    starts = [0]
    last = 0
    for i in range(1,len(q)):
        if q[i]==1 and (i - last) >= min_block:
            starts.append(i); last = i
    # convert to sizes
    sizes = []
    for a,b in zip(starts, starts[1:]+[len(p)]):
        sizes.append(max(1,b-a))
    return sizes

def fuse_global_sizes(n:int, classes:List[int], logits_cls:np.ndarray, reg_frac:float) -> int:
    cls_idx = int(logits_cls.argmax())
    cls_bs  = int(classes[cls_idx])
    reg_bs  = max(1, int(round(reg_frac * n)))
    # simple fusion: average then snap to nearest class if close
    avg_bs = int(round(0.5*cls_bs + 0.5*reg_bs))
    # snap to closest in classes if within 10%
    nearest = min(classes, key=lambda c: abs(c-avg_bs))
    if abs(nearest-avg_bs) <= 0.1*avg_bs:
        return int(nearest)
    return max(1,avg_bs)

# ---- Block-Jacobi LinearOperator (no explicit inverses) ----
def block_jacobi_linear_operator(A: csr_matrix, block_sizes: List[int]):
    """If block_sizes is a single int, uses uniform blocks; else variable blocks along diagonal."""
    n = A.shape[0]
    if isinstance(block_sizes, int):
        seq = []
        s = block_sizes
        i=0
        while i<n:
            seq.append(min(s, n-i))
            i += s
        block_sizes = seq
    # build LU factors per block
    slices=[]; lus=[]
    start=0
    for s in block_sizes:
        end = min(n,start+s)
        sl = slice(start,end)
        slices.append(sl)
        lus.append(splu(A[sl,sl].tocsc()))
        start = end
    def mv(x):
        y = np.zeros_like(x)
        for sl,lu in zip(slices,lus):
            y[sl] = lu.solve(x[sl])
        return y
    from scipy.sparse.linalg import LinearOperator
    return LinearOperator(A.shape, matvec=mv, dtype=A.dtype)

def run_gmres(A: csr_matrix, b: np.ndarray, M=None, tol=1e-2, maxiter=None):
    x, info = sp_gmres(A, b, M=M, tol=tol, maxiter=maxiter)
    return x, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--png", type=str, required=True)
    ap.add_argument("--mm", type=str, default=None, help="MatrixMarket for GMRES")
    ap.add_argument("--target_hw", type=int, default=256)
    ap.add_argument("--use_tiles", action="store_true")
    ap.add_argument("--tile_hw", type=int, default=256)
    ap.add_argument("--tile_stride", type=int, default=128)
    ap.add_argument("--variable_blocks", action="store_true", help="Use per-row starts to create variable blocks")
    ap.add_argument("--min_block", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    model, classes, row_len = load_ckpt(args.ckpt)

    img = load_png_gray(args.png)
    logits_cls, reg_frac, row_logits = infer_full_or_tiled(model, img, target_hw=args.target_hw,
                                                           use_tiles=args.use_tiles, tile_hw=args.tile_hw, tile_stride=args.tile_stride)
    n = img.shape[0]  # assume SPY image height equals matrix n; if different, adjust here.
    # choose block size
    bs_uniform = fuse_global_sizes(n, classes, logits_cls, reg_frac)
    print(f"Pred global block size ~ {bs_uniform} (reg≈{reg_frac*n:.1f}, cls≈{classes[int(logits_cls.argmax())]})")

    # optional variable-size path
    row_probs = 1/(1+np.exp(-row_logits))  # sigmoid
    if args.variable_blocks:
        blocks = threshold_row_starts(row_probs, min_block=args.min_block)
        print(f"Derived variable blocks (first 10): {blocks[:10]} ... total blocks={len(blocks)}")
    else:
        blocks = bs_uniform

    if args.mm:
        A = mmread(args.mm).tocsr()
        b = np.ones(A.shape[0], dtype=np.float64)
        M = block_jacobi_linear_operator(A, blocks)
        x, info = run_gmres(A, b, M=M, tol=args.tol)
        print(f"GMRES info={info} (0=converged)")

if __name__ == "__main__":
    main()
