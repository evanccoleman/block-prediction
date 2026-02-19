#!/usr/bin/env python3
"""
SuiteSparse Downloader (Login Node)

Downloads matrices from SuiteSparse to local storage. Run this on a login node
with network access, then use process_local_matrices.py on compute nodes.

Usage (on login node):
    python download_suitesparse.py --output ./matrices --max-n 500 --max-matrices 50
    
Then (on compute node):
    python process_local_matrices.py --input ./matrices --output ./suitesparse_eval
"""

import os
import sys
import json
import argparse
from pathlib import Path
import urllib.request
import tempfile
import tarfile
import gzip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SuiteSparse matrices (run on login node with network access)"
    )
    
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for downloaded matrices")
    parser.add_argument("--max-n", type=int, default=1000,
                        help="Maximum matrix dimension (default: 1000)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum matrix dimension (default: 10)")
    parser.add_argument("--max-matrices", type=int, default=None,
                        help="Maximum number of matrices to download")
    parser.add_argument("--spd-only", action="store_true", default=True,
                        help="Only download SPD matrices (default: True)")
    parser.add_argument("--include-non-spd", action="store_true",
                        help="Include non-SPD matrices")
    parser.add_argument("--list-only", action="store_true",
                        help="Just list matrices, don't download")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Only download from these groups (e.g., HB Boeing)")
    
    return parser.parse_args()


def download_index():
    """Download and parse the SuiteSparse index CSV."""
    print("Downloading SuiteSparse index...")
    url = "https://sparse.tamu.edu/files/ssstats.csv"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download index: {e}")
        print("\nIf you're on a compute node without network access,")
        print("run this script on a login node instead.")
        sys.exit(1)
    
    # Parse CSV - format is:
    # Line 0: count (e.g., "2904")
    # Line 1: date (e.g., "31-Oct-2023 18:12:37")
    # Line 2+: data rows with format:
    #   Group,Name,nrows,ncols,nnz,isReal,isBinary,isND,posdef,psym,nsym,kind,nnzdiag
    #   0     1    2     3     4   5      6        7    8      9    10   11   12
    
    lines = content.strip().split('\n')
    
    matrices = []
    matrix_id = 0
    
    for line in lines[2:]:  # Skip count and date lines
        parts = line.split(',')
        if len(parts) < 12:
            continue
        
        try:
            matrix_id += 1
            matrix = {
                'group': parts[0].strip(),
                'name': parts[1].strip(),
                'id': matrix_id,
                'nrows': int(parts[2].strip()),
                'ncols': int(parts[3].strip()),
                'nnz': int(parts[4].strip()),
                'is_real': parts[5].strip() == '1',
                'is_binary': parts[6].strip() == '1',
                'is_nd': parts[7].strip() == '1',
                'is_spd': parts[8].strip() == '1',  # posdef column
                'psym': float(parts[9].strip()) if parts[9].strip() else 0,
                'nsym': float(parts[10].strip()) if parts[10].strip() else 0,
                'kind': parts[11].strip() if len(parts) > 11 else '',
            }
            matrices.append(matrix)
        except (ValueError, IndexError) as e:
            continue  # Skip malformed lines
    
    print(f"Parsed {len(matrices)} matrices from index")
    return matrices


def filter_matrices(matrices, min_n, max_n, spd_only, groups=None):
    """Filter matrices by criteria."""
    filtered = []
    
    for m in matrices:
        # Square matrices only
        if m['nrows'] != m['ncols']:
            continue
        
        # Size filter
        if m['nrows'] < min_n or m['nrows'] > max_n:
            continue
        
        # Real-valued only
        if not m.get('is_real', True):
            continue
        
        # SPD filter
        if spd_only and not m.get('is_spd', False):
            continue
        
        # Group filter
        if groups and m['group'].lower() not in [g.lower() for g in groups]:
            continue
        
        filtered.append(m)
    
    # Sort by size for consistent ordering
    filtered.sort(key=lambda x: (x['nrows'], x['group'], x['name']))
    
    return filtered


def download_matrix(matrix, output_dir):
    """Download a single matrix from SuiteSparse."""
    group = matrix['group']
    name = matrix['name']
    
    # Output path
    out_path = Path(output_dir) / f"{group}_{name}.mtx"
    
    if out_path.exists():
        print(f"  Already exists: {out_path.name}")
        return True
    
    # SuiteSparse URL pattern (Matrix Market format)
    # Try .tar.gz first (most common), then .mtx.gz
    urls = [
        f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
        f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz",
    ]
    
    for url in urls:
        try:
            print(f"  Downloading from {url}...")
            
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            
            # Extract .mtx file from tarball
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                with tarfile.open(tmp_path, 'r:gz') as tar:
                    # Find the .mtx file
                    mtx_member = None
                    for member in tar.getmembers():
                        if member.name.endswith('.mtx') and not member.name.endswith('_b.mtx'):
                            mtx_member = member
                            break
                    
                    if mtx_member:
                        # Extract to output
                        f = tar.extractfile(mtx_member)
                        if f:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(out_path, 'wb') as out:
                                out.write(f.read())
                            print(f"  Saved: {out_path.name}")
                            return True
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"  Failed ({url}): {e}")
            continue
    
    print(f"  FAILED: Could not download {group}/{name}")
    return False


def main():
    args = parse_args()
    
    # Download and parse index
    all_matrices = download_index()
    
    # Filter
    spd_only = not args.include_non_spd
    matrices = filter_matrices(
        all_matrices, 
        args.min_n, 
        args.max_n, 
        spd_only=spd_only,
        groups=args.groups
    )
    
    # Show stats before filtering
    total_spd = sum(1 for m in all_matrices if m.get('is_spd', False))
    total_in_range = sum(1 for m in all_matrices 
                         if m['nrows'] == m['ncols'] 
                         and args.min_n <= m['nrows'] <= args.max_n)
    print(f"\nIndex statistics:")
    print(f"  Total matrices: {len(all_matrices)}")
    print(f"  Total SPD: {total_spd}")
    print(f"  Square matrices in size range: {total_in_range}")
    
    print(f"\nFound {len(matrices)} matrices matching criteria:")
    print(f"  Size range: {args.min_n} - {args.max_n}")
    print(f"  SPD only: {spd_only}")
    if args.groups:
        print(f"  Groups: {args.groups}")
    
    # If no SPD matrices found, suggest alternatives
    if len(matrices) == 0 and spd_only:
        print(f"\n  NOTE: No SPD matrices in this range. Try --include-non-spd")
    
    # Limit
    if args.max_matrices and len(matrices) > args.max_matrices:
        print(f"  Limiting to first {args.max_matrices}")
        matrices = matrices[:args.max_matrices]
    
    # List mode
    if args.list_only:
        print(f"\n{'ID':>6} | {'Group':<15} | {'Name':<25} | {'N':>6} | {'NNZ':>10} | SPD")
        print("-" * 80)
        for m in matrices:
            spd_str = "Yes" if m['is_spd'] else "No"
            print(f"{m['id']:>6} | {m['group']:<15} | {m['name']:<25} | {m['nrows']:>6} | {m['nnz']:>10} | {spd_str}")
        print(f"\nTotal: {len(matrices)} matrices")
        return
    
    # Download
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading to: {output_dir}")
    
    success = 0
    failed = 0
    
    for i, m in enumerate(matrices):
        print(f"[{i+1}/{len(matrices)}] {m['group']}/{m['name']} (n={m['nrows']}, nnz={m['nnz']})")
        
        if download_matrix(m, output_dir):
            success += 1
        else:
            failed += 1
    
    # Save manifest
    manifest = {
        'query': {
            'min_n': args.min_n,
            'max_n': args.max_n,
            'spd_only': spd_only,
            'groups': args.groups,
        },
        'matrices': matrices,
        'stats': {
            'total': len(matrices),
            'downloaded': success,
            'failed': failed,
        }
    }
    
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Downloaded: {success}")
    print(f"Failed: {failed}")
    print(f"Manifest saved to: {output_dir / 'manifest.json'}")
    print(f"\nNext step (on compute node):")
    print(f"  python process_local_matrices.py --input {output_dir} --output ./suitesparse_eval")


if __name__ == "__main__":
    main()
