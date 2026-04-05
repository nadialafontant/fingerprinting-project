import csv
import os
import platform
import socket
import time
import uuid
from pathlib import Path

import psutil
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"
RAW_DIR = LOG_DIR / "raw_devices"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_ID_FILE = CONFIG_DIR / "device_id.txt"


def get_device_id() -> str:
    if DEVICE_ID_FILE.exists():
        return DEVICE_ID_FILE.read_text().strip()

    new_id = str(uuid.uuid4())
    DEVICE_ID_FILE.write_text(new_id)
    return new_id


def get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "Unknown_Device"


def get_device_type() -> str:
    machine = platform.machine().lower()
    if "aarch64" in machine or "arm" in machine:
        return "Jetson_Nano_or_ARM_Edge_Device"
    return machine or "Unknown_Device_Type"


def get_gpu_model() -> str:
    # Jetson Nano integrated GPU is Maxwell-based Tegra X1
    # Keep this simple and robust for edge logging.
    try:
        model = "NVIDIA Tegra X1 Integrated GPU"
        return model
    except Exception:
        return "Unknown"


def get_cpu_clock_mhz():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 4)
    except Exception:
        return None


def get_cpu_temp_c():
    # Works on many Linux ARM boards if thermal zone exists.
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]

    for path in candidates:
        try:
            with open(path, "r") as f:
                raw = f.read().strip()
            value = float(raw)
            if value > 1000:
                value = value / 1000.0
            return round(value, 2)
        except Exception:
            continue

    return None


def get_output_path(device_short: str) -> Path:
    return RAW_DIR / f"fingerprint_dataset_{device_short}_jetson_profiler.csv"


def append_rows(rows, output_path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def get_existing_count(output_path: Path) -> int:
    if not output_path.exists():
        return 0

    try:
        with open(output_path, "r", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def build_row(
    device_uuid: str,
    device_short: str,
    hostname: str,
    device_type: str,
    cpu_usage_pct,
    ram_usage_pct,
    cpu_clock_mhz,
    memory_footprint_mb,
    execution_time_sec,
    cpu_temp_c,
    gpu_model,
    notes: str,
    sample_index: int,
):
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": device_uuid,
        "device_short_id": device_short,
        "pc_name": hostname,
        "device_type": device_type,
        "collection_mode": "jetson_device_profiler",
        "sample_index": sample_index,
        "cpu_usage_pct": cpu_usage_pct,
        "ram_usage_pct": ram_usage_pct,
        "cpu_clock_mhz": cpu_clock_mhz,
        "memory_footprint_mb": memory_footprint_mb,
        "execution_time_sec": execution_time_sec,
        "cpu_temp_c": cpu_temp_c,
        "gpu_model": gpu_model,
        "notes": notes,
    }


def profile_device(
    num_samples: int = 300,
    flush_every: int = 25,
    sample_sleep_sec: float = 0.5,
    notes: str = "Jetson device profiling and timing collector only",
):
    device_uuid = get_device_id()
    device_short = device_uuid.split("-")[0]
    hostname = get_hostname()
    device_type = get_device_type()
    gpu_model = get_gpu_model()
    output_path = get_output_path(device_short)

    existing_count = get_existing_count(output_path)
    if existing_count >= num_samples:
        print(f"Already complete: {existing_count}/{num_samples}")
        print(f"Output: {output_path}")
        return

    print(f"Device UUID: {device_uuid}")
    print(f"Device short ID: {device_short}")
    print(f"Hostname: {hostname}")
    print(f"Device type: {device_type}")
    print(f"Output path: {output_path}")
    print(f"Resuming from: {existing_count}/{num_samples}")

    rows = []

    progress = tqdm(
        range(existing_count, num_samples),
        desc="JetsonProfiler",
        unit="sample",
        dynamic_ncols=True,
    )

    for i in progress:
        start = time.time()

        # Small interval makes cpu_percent meaningful per sample
        cpu_usage_pct = psutil.cpu_percent(interval=0.2)
        ram_usage_pct = round(psutil.virtual_memory().percent, 4)
        cpu_clock_mhz = get_cpu_clock_mhz()
        memory_footprint_mb = get_memory_footprint_mb()
        cpu_temp_c = get_cpu_temp_c()

        # Optional pause so samples are spaced out a bit
        if sample_sleep_sec > 0:
            time.sleep(sample_sleep_sec)

        execution_time_sec = round(time.time() - start, 4)

        row = build_row(
            device_uuid=device_uuid,
            device_short=device_short,
            hostname=hostname,
            device_type=device_type,
            cpu_usage_pct=cpu_usage_pct,
            ram_usage_pct=ram_usage_pct,
            cpu_clock_mhz=cpu_clock_mhz,
            memory_footprint_mb=memory_footprint_mb,
            execution_time_sec=execution_time_sec,
            cpu_temp_c=cpu_temp_c,
            gpu_model=gpu_model,
            notes=notes,
            sample_index=i,
        )

        rows.append(row)

        progress.set_postfix({
            "cpu%": cpu_usage_pct,
            "ram%": ram_usage_pct,
            "tempC": cpu_temp_c,
            "time_s": execution_time_sec,
        })

        if (i + 1) % flush_every == 0:
            append_rows(rows, output_path)
            rows = []

    if rows:
        append_rows(rows, output_path)

    print("\nDone.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    profile_device(
        num_samples=300,
        flush_every=25,
        sample_sleep_sec=0.5,
        notes="Jetson Nano edge-device hardware profiling only",
    )