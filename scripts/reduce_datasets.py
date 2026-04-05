import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Input files
MAC_FILE = ROOT / "data" / "raw" / "macbook_air" / "fingerprint_dataset_10dbfbe4_automated.csv"

WINDOWS_10_FILES = [
    ROOT / "data" / "raw" / "windows_10" / "fingerprint_dataset_1240c490_automated.csv",
    ROOT / "data" / "raw" / "windows_10" / "fingerprint_dataset_1240c490_manual.csv",
    ROOT / "data" / "raw" / "windows_10" / "fingerprint_dataset_e9a20125_automated.csv",
    ROOT / "data" / "raw" / "windows_10" / "fingerprint_dataset_e9a20125_manual.csv",
]

WINDOWS_11_FILES = [
    ROOT / "data" / "raw" / "windows_11" / "fingerprint_dataset_306ab189_automated.csv",
    ROOT / "data" / "raw" / "windows_11" / "fingerprint_dataset_306ab189_manual.csv",
]

# Output files
INTERIM_DIR = ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

MAC_OUT = INTERIM_DIR / "macbook_air_reduced.csv"
WIN10_OUT = INTERIM_DIR / "windows_10_reduced.csv"
WIN11_OUT = INTERIM_DIR / "windows_11_reduced.csv"

# Shared columns for reduced hybrid dataset
REDUCED_COLUMNS = [
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
]

# How many rows you want per reduced device dataset
TARGET_ROWS_PER_DEVICE = 1000
RANDOM_SEED = 42


def load_and_tag(file_path: Path, source_device: str, os_family: str, os_version: str, collection_subtype: str):
    df = pd.read_csv(file_path)
    df["source_device"] = source_device
    df["os_family"] = os_family
    df["os_version"] = os_version
    df["collection_subtype"] = collection_subtype
    return df


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def reduce_df(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    df = ensure_columns(df, REDUCED_COLUMNS + ["source_device", "os_family", "os_version", "collection_subtype"])
    df = df[REDUCED_COLUMNS + ["source_device", "os_family", "os_version", "collection_subtype"]]

    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# Load MacBook Air
mac_df = load_and_tag(
    MAC_FILE,
    source_device="MacBook_Air",
    os_family="macOS",
    os_version="macOS",
    collection_subtype="automated",
)

# Load Windows 10
win10_parts = []
for file in WINDOWS_10_FILES:
    subtype = "manual" if "manual" in file.name.lower() else "automated"

    if "1240c490" in file.name:
        source_device = "Windows10_Device1"
    elif "e9a20125" in file.name:
        source_device = "Windows10_Device2"
    else:
        source_device = "Windows10_Unknown"

    win10_parts.append(
        load_and_tag(
            file,
            source_device=source_device,
            os_family="Windows",
            os_version="Windows10",
            collection_subtype=subtype,
        )
    )

win10_df = pd.concat(win10_parts, ignore_index=True)

# Load Windows 11
win11_parts = []
for file in WINDOWS_11_FILES:
    subtype = "manual" if "manual" in file.name.lower() else "automated"

    win11_parts.append(
        load_and_tag(
            file,
            source_device="Windows11_Device1",
            os_family="Windows",
            os_version="Windows11",
            collection_subtype=subtype,
        )
    )

win11_df = pd.concat(win11_parts, ignore_index=True)

# Reduce each dataset
mac_reduced = reduce_df(mac_df, TARGET_ROWS_PER_DEVICE)
win10_reduced = reduce_df(win10_df, TARGET_ROWS_PER_DEVICE)
win11_reduced = reduce_df(win11_df, TARGET_ROWS_PER_DEVICE)

# Save
mac_reduced.to_csv(MAC_OUT, index=False)
win10_reduced.to_csv(WIN10_OUT, index=False)
win11_reduced.to_csv(WIN11_OUT, index=False)

print("Reduction complete.\n")

print(f"Mac original rows: {len(mac_df)}")
print(f"Mac reduced rows: {len(mac_reduced)}")
print(f"Saved to: {MAC_OUT}\n")

print(f"Windows 10 original rows: {len(win10_df)}")
print(f"Windows 10 reduced rows: {len(win10_reduced)}")
print(f"Saved to: {WIN10_OUT}\n")

print(f"Windows 11 original rows: {len(win11_df)}")
print(f"Windows 11 reduced rows: {len(win11_reduced)}")
print(f"Saved to: {WIN11_OUT}\n")