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
    # Add a flag to allow experimenting with coefficients without regeneration!
    parser.add_argument("--setup_coeff", type=float, default=5e-7, help="Override SETUP_COEFF for visualization")
    return parser.parse_args()


def calculate_theoretical_opt(perf_data, nnz, setup_coeff=1e-8, solve_coeff=1e-5):
    """
    Re-calculates the winner offline using saved physics data.
    """
    best_frac = None
    min_cost = float('inf')

    for frac_str, stats in perf_data.items():
        if stats['status'] != 'converged':
            continue

        frac = float(frac_str)
        block_size = stats['block_size']
        iterations = stats['iterations']

        # THE MODEL (Re-tunable!)
        setup_cost = (block_size ** 3) * setup_coeff
        solve_cost = iterations * (nnz * solve_coeff)
        total_cost = setup_cost + solve_cost

        if total_cost < min_cost:
            min_cost = total_cost
            best_frac = frac

    return best_frac


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
                nnz = entry['matrix_properties']['nnz']

                # Extract Ground Truth (Handle old files gracefully)
                ground_truth = entry.get('ground_truth', {}).get('generated_structure_fraction', None)

                # 1. Existing Labels
                opt_iters = float(entry['labels']['optimal_fraction_iterations'])

                # 2. Re-calculated Theoretical Label (OFFLINE TUNING)
                opt_theoretical = calculate_theoretical_opt(
                    perf, nnz, setup_coeff=custom_setup_coeff
                )

                if opt_theoretical is not None:
                    row = {
                        "id": entry['matrix_id'],
                        "Theoretical Time": opt_theoretical,
                        "Iterations": opt_iters,
                        "Generated Structure": ground_truth,
                        "size": entry['matrix_properties']['size'],
                        "nnz": nnz
                    }
                    data.append(row)
        except Exception as e:
            # print(f"Skipping broken file {f}: {e}")
            pass

    return pd.DataFrame(data)


def plot_balance(df):
    plt.figure(figsize=(12, 6))

    # We compare Iterations (Truth) vs Theoretical (Model)
    df_melt = df.melt(
        id_vars=['id'],
        value_vars=['Theoretical Time', 'Iterations'],
        var_name='Metric',
        value_name='Block Fraction'
    )

    sns.countplot(data=df_melt, x='Block Fraction', hue='Metric', palette='viridis')

    plt.title("Class Balance: Iterations vs. Tuned Model")
    plt.xlabel("Block Fraction Class")
    plt.ylabel("Number of Samples")
    plt.legend(title="Optimality Metric")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('dataset_balance_tuned.png')
    print("Saved dataset_balance_tuned.png")


def plot_correlation(df):
    """
    Plots Generated Structure (X) vs Solver Choice (Y).
    This tells us if the solver is actually 'seeing' the structure.
    """
    if df['Generated Structure'].isnull().all():
        print("Skipping correlation plot (No ground truth data found in JSONs).")
        return

    plt.figure(figsize=(8, 8))

    # Jitter to see density
    x_jitter = df['Generated Structure'] + np.random.uniform(-0.01, 0.01, len(df))
    y_jitter = df['Theoretical Time'] + np.random.uniform(-0.01, 0.01, len(df))

    plt.scatter(x_jitter, y_jitter, alpha=0.4, c='teal', s=15, edgecolors='none')

    # Perfect match line
    plt.plot([0.05, 0.40], [0.05, 0.40], 'r--', alpha=0.5, label='Ideal Match')

    plt.xlabel("True Generated Block Fraction")
    plt.ylabel("Theoretical Optimal Fraction")
    plt.title("Sanity Check: Does Structure Predict Optimality?")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('correlation_check.png')
    print("Saved correlation_check.png")


if __name__ == "__main__":
    args = parse_cli()

    print(f"--- Running Visualization ---")
    print(f"Using setup coefficient: {args.setup_coeff}")

    df = load_metadata(args.data_dir, args.setup_coeff)

    if not df.empty:
        print(f"Loaded {len(df)} samples.")
        plot_balance(df)
        plot_correlation(df)

        print("\nTo experiment with the cost model, run:")
        print("python visualize_dataset.py --data_dir YOUR_DIR --setup_coeff 5e-8")
    else:
        print("No data found.")
