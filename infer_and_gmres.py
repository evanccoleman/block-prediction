
#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import csr_matrix, bmat, csc_matrix
from scipy.sparse.linalg import inv, gmres as sp_gmres

# --- Model should match spy_cnn_torch.SpyCNN ---
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, k1=3, k2=3):
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

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    model = SpyCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes

def load_png(path: str):
    im = Image.open(path).convert("L")
    x = torch.from_numpy((np.array(im, dtype="float32")/255.0)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return x

def pick_block(model, classes, png_path: str):
    x = load_png(png_path)
    with torch.no_grad():
        logits = model(x)
        idx = int(logits.argmax(dim=1))
    return classes[idx]

# --- Block-Jacobi utilities ---
def block_jacobi_preconditioner(A: csr_matrix, block_size: int):
    n = A.shape[0]
    inv_blocks = []
    for row_start in range(0, n, block_size):
        row_end = min(row_start + block_size, n)
        block = csc_matrix(A[row_start:row_end, row_start:row_end])
        inv_blocks.append(inv(block))
    num_blocks = len(inv_blocks)
    M = bmat([[inv_blocks[i] if i == j else None for j in range(num_blocks)]
              for i in range(num_blocks)], format="csr")
    return M

def run_gmres(A: csr_matrix, b: np.ndarray, M=None, tol=1e-2):
    x, info = sp_gmres(A, b, M=M, tol=tol)
    return x, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--png", type=str, required=True, help="PNG of SPY image of A")
    ap.add_argument("--mm", type=str, default=None, help="(optional) .mtx MatrixMarket file for GMRES run")
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    model, classes = load_checkpoint(args.ckpt)
    block_size = pick_block(model, classes, args.png)
    print(f"Predicted block size: {block_size}")

    if args.mm:
        from scipy.io import mmread
        A = mmread(args.mm).tocsr()
        b = np.ones(A.shape[0])
        M = block_jacobi_preconditioner(A, block_size)
        x, info = run_gmres(A, b, M=M, tol=args.tol)
        print(f"GMRES info code: {info} (0 means converged)")

if __name__ == "__main__":
    main()
