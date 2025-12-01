import os
import subprocess

# --- CONFIGURATION ---
PYTHON_EXE = "python"  # or "python3"
SCRIPT_NAME = "unified_cnn.py"

# Folders containing your generated data (must have 'metadata' subfolder)
DATA_FOLDERS = [
    "png_dataset_128_script",
    "png_dataset_500_script",
    "png_dataset_1000_script",
    "png_dataset_2500_script"
]

# The Permutations
INPUT_TYPES = ["png", "dense"]
MODES = ["classification", "regression"]

# Output
RESULTS_DIR = "batch_results_v1"


def run_batch():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for folder in DATA_FOLDERS:
        for mode in MODES:
            for input_type in INPUT_TYPES:

                # Construct a descriptive run name
                # e.g. "dataset_500_A_class_png"
                clean_folder = os.path.basename(folder)
                if mode == "classification":
                    short_mode = "class"
                else:
                    short_mode = "reg"

                run_name = f"{clean_folder}_{short_mode}_{input_type}"

                print(f"--- STARTING RUN: {run_name} ---")

                cmd = [
                    PYTHON_EXE, SCRIPT_NAME,
                    "--data_dir", folder,
                    "--mode", mode,
                    "--input_type", input_type,
                    "--output_dir", RESULTS_DIR,
                    "--run_name", run_name,
                    "--epochs", "20",

                    # NEW ARGS TO FIX OOM
                    "--batch_size", "8"  # Reduced from 32
                    #"--target_size", "128"  # Downsample 500x500 -> 128x128
                ]

                # For regression, we default to 'solver' (interpolated),
                # but you could add a loop here if you wanted to test 'physics' too.
                if mode == "regression":
                    cmd += ["--reg_label", "solver"]

                try:
                    subprocess.run(cmd, check=True)
                    print(f"--- COMPLETED: {run_name} ---\n")
                except subprocess.CalledProcessError as e:
                    print(f"!!! FAILED: {run_name} with error {e} !!!\n")


if __name__ == "__main__":
    run_batch()

