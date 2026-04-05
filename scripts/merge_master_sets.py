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

# Output
OUTPUT_FILE = ROOT / "logs" / "merged" / "fingerprint_master_data.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_csv(file_path, source_device, os_version, collection_subtype):
    df = pd.read_csv(file_path)
    df["source_device"] = source_device
    df["os_family"] = "Windows" if "Windows" in source_device else "macOS"
    df["os_version"] = os_version
    df["collection_subtype"] = collection_subtype
    return df


dfs = []

# Mac
dfs.append(
    load_csv(
        MAC_FILE,
        source_device="MacBook_Air",
        os_version="macOS",
        collection_subtype="automated"
    )
)

# Windows 10
for file in WINDOWS_10_FILES:
    subtype = "manual" if "manual" in file.name.lower() else "automated"

    if "1240c490" in file.name:
        source_device = "Windows10_Device1"
    elif "e9a20125" in file.name:
        source_device = "Windows10_Device2"
    else:
        source_device = "Windows10_Unknown"

    dfs.append(
        load_csv(
            file,
            source_device=source_device,
            os_version="Windows10",
            collection_subtype=subtype
        )
    )

# Windows 11
for file in WINDOWS_11_FILES:
    subtype = "manual" if "manual" in file.name.lower() else "automated"

    dfs.append(
        load_csv(
            file,
            source_device="Windows11_Device1",
            os_version="Windows11",
            collection_subtype=subtype
        )
    )

# Merge all
merged_df = pd.concat(dfs, ignore_index=True)

# Save
merged_df.to_csv(OUTPUT_FILE, index=False)

print("Merge complete.")
print(f"Total rows: {len(merged_df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nRows by source device:")
print(merged_df["source_device"].value_counts())

print("\nRows by collection subtype:")
print(merged_df["collection_subtype"].value_counts())

print("\nRows by OS version:")
print(merged_df["os_version"].value_counts())