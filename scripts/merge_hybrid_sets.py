import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTERIM_DIR = ROOT / "data" / "interim"
RAW_DIR = ROOT / "data" / "raw"
FINAL_DIR = ROOT / "data" / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Reduced Apple/Windows files
MAC_REDUCED = INTERIM_DIR / "macbook_air_reduced.csv"
WIN10_REDUCED = INTERIM_DIR / "windows_10_reduced.csv"
WIN11_REDUCED = INTERIM_DIR / "windows_11_reduced.csv"

# Edge profiler folders
JETSON_DIR = RAW_DIR / "jetson_nano"
RPI_DIR = RAW_DIR / "raspberry_pi"

# Output
OUTPUT_FILE = FINAL_DIR / "classifier_reduced_device_dataset.csv"

# Final shared schema
HYBRID_COLUMNS = [
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "device_type",
    "collection_mode",
    "sample_index",
    "cpu_usage_pct",
    "ram_usage_pct",
    "cpu_clock_mhz",
    "memory_footprint_mb",
    "execution_time_sec",
    "cpu_temp_c",
    "gpu_model",
    "notes",
    "source_device",
    "os_family",
    "os_version",
    "collection_subtype",
]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def load_single_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = ensure_columns(df, HYBRID_COLUMNS)
    return df[HYBRID_COLUMNS]


def load_folder_csvs(folder: Path, source_device: str, os_family: str, os_version: str, collection_subtype: str) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["source_device"] = source_device
        df["os_family"] = os_family
        df["os_version"] = os_version
        df["collection_subtype"] = collection_subtype
        df = ensure_columns(df, HYBRID_COLUMNS)
        dfs.append(df[HYBRID_COLUMNS])

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame(columns=HYBRID_COLUMNS)


# Load reduced Apple/Windows
mac_df = load_single_csv(MAC_REDUCED)
win10_df = load_single_csv(WIN10_REDUCED)
win11_df = load_single_csv(WIN11_REDUCED)

# Load edge profiler datasets
jetson_df = load_folder_csvs(
    JETSON_DIR,
    source_device="Jetson_Nano",
    os_family="Linux",
    os_version="Jetson_Linux",
    collection_subtype="device_profiler_only",
)

rpi_df = load_folder_csvs(
    RPI_DIR,
    source_device="Raspberry_Pi",
    os_family="Linux",
    os_version="RaspberryPi_OS",
    collection_subtype="device_profiler_only",
)

# Merge everything
hybrid_df = pd.concat(
    [mac_df, win10_df, win11_df, jetson_df, rpi_df],
    ignore_index=True
)

# Save
hybrid_df.to_csv(OUTPUT_FILE, index=False)

print("Hybrid merge complete.\n")
print(f"MacBook reduced rows: {len(mac_df)}")
print(f"Windows 10 reduced rows: {len(win10_df)}")
print(f"Windows 11 reduced rows: {len(win11_df)}")
print(f"Jetson rows: {len(jetson_df)}")
print(f"Raspberry Pi rows: {len(rpi_df)}")
print(f"\nTotal hybrid rows: {len(hybrid_df)}")
print(f"Saved to: {OUTPUT_FILE}")

if not hybrid_df.empty:
    print("\nRows by source_device:")
    print(hybrid_df["source_device"].value_counts())

    print("\nRows by device_type:")
    print(hybrid_df["device_type"].value_counts(dropna=False))

    print("\nRows by collection_mode:")
    print(hybrid_df["collection_mode"].value_counts(dropna=False))