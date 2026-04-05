import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JETSON_DIR = ROOT / "data" / "raw" / "jetson_nano"
RPI_DIR = ROOT / "data" / "raw" / "raspberry_pi"

OUTPUT_FILE = ROOT / "data" / "interim" / "merged_edge_profiler.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_all_csvs(folder, source_device, os_version):
    files = list(folder.glob("*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df["source_device"] = source_device
        df["os_family"] = "Linux"
        df["os_version"] = os_version
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


jetson_df = load_all_csvs(JETSON_DIR, "Jetson_Nano", "Jetson_Linux")
rpi_df = load_all_csvs(RPI_DIR, "Raspberry_Pi", "RaspberryPi_OS")

merged_df = pd.concat([jetson_df, rpi_df], ignore_index=True)

merged_df.to_csv(OUTPUT_FILE, index=False)

print("Edge merge complete.")
print(f"Jetson rows: {len(jetson_df)}")
print(f"Raspberry Pi rows: {len(rpi_df)}")
print(f"Total rows: {len(merged_df)}")
print(f"Saved to: {OUTPUT_FILE}")

if not merged_df.empty:
    print("\nRows by source device:")
    print(merged_df["source_device"].value_counts())