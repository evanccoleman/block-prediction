#!/usr/bin/env python
"""
Analyze how much information is lost when downsampling matrix spy images.

This helps answer: Can a fixed 128×128 image representation work for 
matrices of any size, or does critical block structure information get lost?

Key insight: If the "optimal block size" is determined by macro-scale
patterns (general blockiness, diagonal bands), then downsampling may
preserve enough information. But if it depends on fine-grained structure,
downsampling will fail.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import sparse
from scipy.ndimage import zoom
import warnings


def generate_block_matrix(n: int, block_size: int, noise: float = 0.01) -> sparse.csr_matrix:
    """Generate a sparse matrix with block-diagonal structure."""
    data = []
    rows = []
    cols = []
    
    num_blocks = n // block_size
    
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        
        # Dense block with some structure
        for i in range(start, end):
            for j in range(start, end):
                if np.random.random() < 0.8:  # 80% density within block
                    val = np.random.randn()
                    data.append(val)
                    rows.append(i)
                    cols.append(j)
        
        # Off-diagonal connections (sparse)
        if b < num_blocks - 1:
            next_start = (b + 1) * block_size
            for _ in range(block_size // 4):
                i = np.random.randint(start, end)
                j = np.random.randint(next_start, next_start + block_size)
                data.append(np.random.randn() * 0.1)
                rows.append(i)
                cols.append(j)
                data.append(np.random.randn() * 0.1)
                rows.append(j)
                cols.append(i)
    
    # Add noise
    noise_nnz = int(noise * n * n)
    for _ in range(noise_nnz):
        i, j = np.random.randint(0, n, 2)
        data.append(np.random.randn() * 0.01)
        rows.append(i)
        cols.append(j)
    
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return matrix


def matrix_to_image(matrix: sparse.spmatrix, size: int = 128) -> np.ndarray:
    """Convert sparse matrix to binary image of specified size."""
    n = matrix.shape[0]
    
    if n == size:
        # Direct conversion
        img = (matrix.toarray() != 0).astype(np.float32)
    elif n < size:
        # Upscale (rare case)
        binary = (matrix.toarray() != 0).astype(np.float32)
        img = zoom(binary, size / n, order=0)
    else:
        # Downscale - this is the interesting case
        # Use block averaging to preserve density information
        binary = (matrix.toarray() != 0).astype(np.float32)
        
        # Compute block size for averaging
        block_h = n / size
        block_w = n / size
        
        img = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                r_start = int(i * block_h)
                r_end = int((i + 1) * block_h)
                c_start = int(j * block_w)
                c_end = int((j + 1) * block_w)
                
                # Average density in this block
                block = binary[r_start:r_end, c_start:c_end]
                img[i, j] = block.mean()
        
    return img


def compute_block_visibility_score(original_n: int, block_size: int, 
                                    image_size: int = 128) -> dict:
    """
    Compute how well block structure is preserved after downsampling.
    
    Returns metrics about visibility of block structure.
    """
    # Generate matrix
    matrix = generate_block_matrix(original_n, block_size)
    
    # Convert to image
    img = matrix_to_image(matrix, image_size)
    
    # Metrics
    downsampling_ratio = original_n / image_size
    pixels_per_block = block_size / downsampling_ratio
    
    # Can we distinguish blocks?
    # A block needs at least 2-3 pixels to be visible
    blocks_visible = pixels_per_block >= 2
    
    # Compute "blockiness" score using autocorrelation
    # High autocorrelation at block_size/downsampling_ratio indicates visible blocks
    from scipy import signal
    
    # 1D autocorrelation along diagonal
    diag_signal = np.diag(img)
    autocorr = signal.correlate(diag_signal, diag_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Look for peak at expected block period
    expected_period = int(pixels_per_block)
    if expected_period > 1 and expected_period < len(autocorr):
        periodicity_score = autocorr[expected_period]
    else:
        periodicity_score = 0.0
    
    # Edge detection score (blocks should have sharp edges)
    from scipy.ndimage import sobel
    edges_h = sobel(img, axis=0)
    edges_v = sobel(img, axis=1)
    edge_magnitude = np.sqrt(edges_h**2 + edges_v**2)
    edge_score = edge_magnitude.mean()
    
    return {
        'original_n': original_n,
        'block_size': block_size,
        'image_size': image_size,
        'downsampling_ratio': downsampling_ratio,
        'pixels_per_block': pixels_per_block,
        'blocks_theoretically_visible': blocks_visible,
        'periodicity_score': periodicity_score,
        'edge_score': edge_score,
        'nnz': matrix.nnz,
        'density': matrix.nnz / (original_n ** 2),
    }


def visualize_downsampling_effect(output_dir: Path = None):
    """Create visualization of how block structure degrades with matrix size."""
    
    if output_dir is None:
        output_dir = Path('.')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Test different matrix sizes with fixed block structure
    matrix_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    block_size_fraction = 0.1  # 10% of matrix size = optimal block
    image_size = 128
    
    fig, axes = plt.subplots(2, len(matrix_sizes), figsize=(20, 6))
    
    results = []
    
    for idx, n in enumerate(matrix_sizes):
        block_size = max(8, int(n * block_size_fraction))
        
        print(f"Processing n={n}, block_size={block_size}...")
        
        # Generate and convert
        matrix = generate_block_matrix(n, block_size, noise=0.005)
        img = matrix_to_image(matrix, image_size)
        
        # Compute metrics
        metrics = compute_block_visibility_score(n, block_size, image_size)
        results.append(metrics)
        
        # Plot original (or portion of it)
        ax1 = axes[0, idx]
        if n <= 512:
            ax1.spy(matrix, markersize=0.5)
        else:
            # Show top-left corner
            corner_size = min(512, n)
            corner = matrix[:corner_size, :corner_size]
            ax1.spy(corner, markersize=0.5)
            ax1.set_title(f'n={n}\n(showing {corner_size}×{corner_size})')
        ax1.set_title(f'n={n}, block={block_size}')
        
        # Plot downsampled image
        ax2 = axes[1, idx]
        ax2.imshow(img, cmap='gray_r', interpolation='nearest')
        ax2.set_title(f'{metrics["pixels_per_block"]:.1f} px/block\n'
                     f'period={metrics["periodicity_score"]:.2f}')
        ax2.axis('off')
    
    axes[0, 0].set_ylabel('Original Matrix\n(or corner)', fontsize=12)
    axes[1, 0].set_ylabel(f'Downsampled to\n{image_size}×{image_size}', fontsize=12)
    
    plt.suptitle('Effect of Downsampling on Block Structure Visibility', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'downsampling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = [r['original_n'] for r in results]
    
    axes[0].semilogx(sizes, [r['pixels_per_block'] for r in results], 'bo-')
    axes[0].axhline(y=2, color='r', linestyle='--', label='Minimum for visibility')
    axes[0].set_xlabel('Matrix Size (n)')
    axes[0].set_ylabel('Pixels per Block')
    axes[0].set_title('Block Resolution vs Matrix Size')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].semilogx(sizes, [r['periodicity_score'] for r in results], 'go-')
    axes[1].set_xlabel('Matrix Size (n)')
    axes[1].set_ylabel('Periodicity Score')
    axes[1].set_title('Block Pattern Detectability')
    axes[1].grid(True)
    
    axes[2].semilogx(sizes, [r['edge_score'] for r in results], 'ro-')
    axes[2].set_xlabel('Matrix Size (n)')
    axes[2].set_ylabel('Edge Score')
    axes[2].set_title('Block Edge Sharpness')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNSAMPLING ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Matrix Size':<12} {'Block Size':<12} {'Downsample':<12} {'Px/Block':<10} {'Visible?':<10}")
    print("-"*60)
    
    for r in results:
        visible = "Yes" if r['pixels_per_block'] >= 2 else "NO"
        print(f"{r['original_n']:<12} {r['block_size']:<12} {r['downsampling_ratio']:<12.1f} "
              f"{r['pixels_per_block']:<10.2f} {visible:<10}")
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print(f"For a {image_size}×{image_size} image, block structure becomes invisible when:")
    print(f"  - Matrix size n > ~{image_size * 10} (assuming 10% block fraction)")
    print(f"  - This is because blocks need ≥2 pixels to be distinguishable")
    print(f"\nFor n = 10^6 with {image_size}×{image_size} image:")
    print(f"  - Downsampling ratio: {1e6/image_size:.0f}:1")
    print(f"  - A 10% block (100,000 pixels) becomes: {100000/(1e6/image_size):.1f} pixels")
    print(f"  - This MIGHT still be visible! (barely)")
    print(f"\nBut a 1% block (10,000 pixels) becomes: {10000/(1e6/image_size):.2f} pixels")
    print(f"  - This would NOT be visible.")
    print("="*80)
    
    return results


def analyze_tiling_approach(n: int = 1000, tile_size: int = 128, 
                           block_size: int = 64) -> dict:
    """
    Analyze the tiling approach from the original paper.
    
    Instead of downsampling, we extract tiles and predict on each.
    """
    print(f"\n{'='*60}")
    print(f"TILING ANALYSIS: n={n}, tile_size={tile_size}, block_size={block_size}")
    print(f"{'='*60}")
    
    # Generate matrix
    matrix = generate_block_matrix(n, block_size)
    
    # Extract tiles along diagonal
    num_tiles = (n - tile_size) // (tile_size // 2) + 1  # 50% overlap
    
    tiles = []
    tile_positions = []
    
    for i in range(num_tiles):
        start = i * (tile_size // 2)
        end = start + tile_size
        if end > n:
            break
        
        tile = matrix[start:end, start:end].toarray()
        tiles.append(tile)
        tile_positions.append((start, end))
    
    print(f"Number of tiles: {len(tiles)}")
    print(f"Tile positions: {tile_positions[:3]} ... {tile_positions[-1]}")
    
    # Analyze consistency of block structure across tiles
    block_densities = []
    for tile in tiles:
        # Estimate block structure in tile
        # Count transitions in sparsity pattern
        binary = (tile != 0).astype(float)
        row_densities = binary.mean(axis=1)
        
        # Look for periodicity
        from scipy import signal
        if len(row_densities) > 10:
            autocorr = signal.correlate(row_densities, row_densities, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            block_densities.append(autocorr[:tile_size//2].mean())
    
    consistency = np.std(block_densities) / (np.mean(block_densities) + 1e-10)
    
    print(f"\nTile consistency (lower is better): {consistency:.4f}")
    print(f"Mean block signal: {np.mean(block_densities):.4f}")
    print(f"Std block signal: {np.std(block_densities):.4f}")
    
    return {
        'n': n,
        'tile_size': tile_size,
        'num_tiles': len(tiles),
        'consistency': consistency,
        'mean_signal': np.mean(block_densities),
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze resolution effects')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run downsampling analysis
    print("Running downsampling analysis...")
    results = visualize_downsampling_effect(output_dir)
    
    # Run tiling analysis
    print("\n\nRunning tiling analysis...")
    for n in [500, 1000, 5000, 10000]:
        analyze_tiling_approach(n=n, tile_size=128, block_size=int(n * 0.1))
    
    print(f"\n\nPlots saved to: {output_dir}")
