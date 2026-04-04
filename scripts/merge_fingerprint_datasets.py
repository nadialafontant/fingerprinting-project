from pathlib import Path
import pandas as pd

RAW_DIR = Path("logs/raw_devices")
MERGED_DIR = Path("logs/merged")
MERGED_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = MERGED_DIR / "fingerprint_dataset_master.csv"

EXPECTED_COLUMNS = [
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "collection_mode",
    "sample_index",
    "true_label",
    "model_type",
    "parameters",
    "prediction",
    "execution_time_sec",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "total_energy_kwh",
    "total_emissions_kg",
    "cpu_usage_pct",
    "gpu_model",
    "ram_usage_pct",
    "cpu_clock_mhz",
    "memory_footprint_mb",
    "model_accuracy",
    "model_precision_weighted",
    "model_recall_weighted",
    "model_f1_weighted",
    "model_flops",
]

def main():
    files = sorted(RAW_DIR.glob("fingerprint_dataset_*.csv"))

    if not files:
        print("No dataset files found in logs/raw_devices/")
        return

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df = df[EXPECTED_COLUMNS]
            df["source_file"] = path.name
            dfs.append(df)
            print(f"Loaded {path.name}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped {path.name}: {e}")

    if not dfs:
        print("No valid data loaded.")
        return

    master = pd.concat(dfs, ignore_index=True)
    master.to_csv(MASTER_PATH, index=False)
    print(f"Saved merged dataset to {MASTER_PATH} with {len(master)} rows")

if __name__ == "__main__":
    main()