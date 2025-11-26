import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def parse_cli():
    parser = argparse.ArgumentParser(description="Dataset Visualizer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--setup_coeff", type=float, default=5e-7, help="Override SETUP_COEFF for visualization")
    return parser.parse_args()


def calculate_interpolated_opt(perf_data, setup_coeff=1e-7, solve_coeff=1e-5):
    """
    Re-calculates the continuous interpolated winner offline using saved physics data.
    """
    x = []
    y = []

    for frac_str, stats in perf_data.items():
        if stats['status'] != 'converged': continue

        frac = float(frac_str)

        # Recalculate cost
        # Note: Using simplified proxy for Bandwidth (A.nnz + M.nnz)
        # Ensure this matches your generation logic or use what's saved
        setup_cost = (stats['block_size'] ** 3) * setup_coeff
        solve_cost = stats['iterations'] * ((stats['num_blocks'] * stats['preconditioner_nnz']) * solve_coeff)

        total_cost = setup_cost + solve_cost

        x.append(frac)
        y.append(total_cost)

    if len(x) < 3:
        return x[np.argmin(y)] if x else None

    try:
        z = np.polyfit(x, y, 2)
        a, b, c = z
        if a > 0:  # Convex
            min_x = -b / (2 * a)
            return max(0.05, min(0.40, min_x))
        else:
            return x[np.argmin(y)]
    except:
        return x[np.argmin(y)]


def load_metadata(data_dir, custom_setup_coeff):
    meta_dir = os.path.join(data_dir, "metadata")
    data = []

    if not os.path.exists(meta_dir):
        print(f"Error: {meta_dir} not found.")
        return pd.DataFrame()

    files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    print(f"Loading {len(files)} metadata files...")

    for f in files:
        try:
            with open(os.path.join(meta_dir, f), 'r') as json_file:
                entry = json.load(json_file)

                perf = entry.get('performance_data', {})
                labels = entry.get('labels', {})

                # 1. Get Regression Ground Truth (Physics)
                ground_truth = labels.get('regression_ground_truth')

                # 2. Get Regression Optimal (Solver)
                # Option A: Load from JSON (Fixed at generation time)
                # opt_interp = labels.get('regression_interpolated_optimal')

                # Option B: Recalculate (Allows tuning setup_coeff dynamically!)
                opt_interp = calculate_interpolated_opt(perf, setup_coeff=custom_setup_coeff)
                #opt_interp = float(entry['labels']['class_optimal_iterations'])

                if ground_truth is not None and opt_interp is not None:
                    row = {
                        "id": entry['matrix_id'],
                        "Truth": ground_truth,
                        "Optimal": opt_interp,
                        "Residual": opt_interp - ground_truth
                    }
                    data.append(row)
        except Exception as e:
            pass

    return pd.DataFrame(data)


def plot_regression_spread(df):
    plt.figure(figsize=(14, 6))

    # Subplot 1: Scatter (Truth vs Optimal)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Truth', y='Optimal', alpha=0.5, color='teal')
    plt.plot([0.05, 0.40], [0.05, 0.40], 'r--', alpha=0.8, label='Perfect Match')
    plt.title("Regression Correlation: Physics vs. Solver")
    plt.xlabel("Ground Truth (Generated Structure)")
    plt.ylabel("Solver Optimal (Interpolated)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Residual Distribution (The "Spread")
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='Residual', kde=True, color='orange', bins=30)
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.title("Oversizing Bias: (Optimal - Truth)")
    plt.xlabel("Difference (Positive = Solver prefers larger blocks)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_spread.png')
    print("Saved regression_spread.png")


if __name__ == "__main__":
    args = parse_cli()

    print(f"--- Visualizing Regression Spread ---")
    print(f"Using setup coefficient: {args.setup_coeff}")

    df = load_metadata(args.data_dir, args.setup_coeff)

    if not df.empty:
        print(f"Loaded {len(df)} samples with regression data.")
        plot_regression_spread(df)

        # Print stats
        mean_bias = df['Residual'].mean()
        print(f"\nMean Bias: {mean_bias:.4f} (Positive = Safe Oversizing)")
        print(f"Correlation: {df['Truth'].corr(df['Optimal']):.4f}")
    else:
        print("No regression data found.")