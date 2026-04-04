import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from model import SimpleCNN, SimpleDNN
from llm_utils import load_tiny_llm_model, predict_with_tiny_llm
from vlm_utils import load_tiny_vlm_model, predict_with_tiny_vlm
from PIL import Image
from torchvision import transforms
import time
import uuid
import os
import pandas as pd
import psutil
from codecarbon import EmissionsTracker


def get_device_id():
    id_file = "device_id.txt"
    if os.path.exists(id_file):
        with open(id_file, "r") as f:
            return f.read().strip()
    else:
        new_id = str(uuid.uuid4())
        with open(id_file, "w") as f:
            f.write(new_id)
        return new_id


DEVICE_UUID = get_device_id()


def get_hostname():
    return os.getenv("COMPUTERNAME", os.getenv("HOSTNAME", "Unknown_Device"))


def ensure_session_state():
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    if "run_counts" not in st.session_state:
        st.session_state.run_counts = {
            "CNN": 0,
            "DNN": 0,
            "Tiny LLM": 0,
            "Tiny VLM": 0
        }

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "last_log_entry" not in st.session_state:
        st.session_state.last_log_entry = None

    if "last_model_choice" not in st.session_state:
        st.session_state.last_model_choice = None

    if "last_confidence" not in st.session_state:
        st.session_state.last_confidence = None


def get_csv_counts():
    file_path = "fingerprint_master_data.csv"
    counts = {"CNN": 0, "DNN": 0, "Tiny LLM": 0, "Tiny VLM": 0}

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if "model_type" in df.columns:
                value_counts = df["model_type"].value_counts().to_dict()
                for model_name in counts:
                    counts[model_name] = int(value_counts.get(model_name, 0))
        except Exception:
            pass

    return counts


def log_fingerprint(model_name, num_params, exec_time, emissions_data, tracker, prediction):
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Only)"

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "pc_name": get_hostname(),
        "model_type": model_name,
        "parameters": num_params,
        "prediction": prediction,
        "execution_time_sec": round(exec_time, 4),

        "cpu_energy_kwh": getattr(tracker.final_emissions_data, "cpu_energy", 0),
        "gpu_energy_kwh": getattr(tracker.final_emissions_data, "gpu_energy", 0),
        "ram_energy_kwh": getattr(tracker.final_emissions_data, "ram_energy", 0),
        "total_energy_kwh": getattr(tracker.final_emissions_data, "energy_consumed", 0),
        "total_emissions_kg": emissions_data,

        "cpu_usage_pct": cpu_usage,
        "gpu_model": gpu_name,
        "ram_usage_pct": ram_usage,
        "cpu_clock_mhz": cpu_freq,
        "memory_footprint_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    }

    df = pd.DataFrame([metadata])
    file_path = "fingerprint_master_data.csv"
    df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)

    return metadata


ensure_session_state()

st.title("ML Fingerprinter")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar info
st.sidebar.header("Device Identity")
st.sidebar.text(f"ID: {DEVICE_UUID}")
st.sidebar.text(f"Device: {get_hostname()}")

csv_counts = get_csv_counts()

st.sidebar.header("Run Tracking")
st.sidebar.write("Saved dataset counts:")
st.sidebar.write(f"CNN: {csv_counts['CNN']} / 1000")
st.sidebar.write(f"DNN: {csv_counts['DNN']} / 1000")
st.sidebar.write(f"Tiny LLM: {csv_counts['Tiny LLM']} / 1000")
st.sidebar.write(f"Tiny VLM: {csv_counts['Tiny VLM']} / 1000")

st.sidebar.write("Current session counts:")
st.sidebar.write(f"CNN: {st.session_state.run_counts['CNN']}")
st.sidebar.write(f"DNN: {st.session_state.run_counts['DNN']}")
st.sidebar.write(f"Tiny LLM: {st.session_state.run_counts['Tiny LLM']}")
st.sidebar.write(f"Tiny VLM: {st.session_state.run_counts['Tiny VLM']}")

model_choice = st.selectbox(
    "Select Model to Fingerprint",
    ("CNN", "DNN", "Tiny LLM", "Tiny VLM")
)

# Load selected model
if model_choice == "CNN":
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

elif model_choice == "DNN":
    model = SimpleDNN().to(device)
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location=device))
    model.eval()

elif model_choice == "Tiny LLM":
    model = load_tiny_llm_model("mnist_tiny_llm.pth")

elif model_choice == "Tiny VLM":
    model, vlm_text_processor = load_tiny_vlm_model("mnist_tiny_vlm.pth")

st.write("Draw a single digit (0–9) below:")

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    key=f"canvas_{st.session_state.canvas_key}"
)

col1, col2 = st.columns(2)

with col1:
    identify_clicked = st.button("Identify")

with col2:
    if st.button("Draw Another Number"):
        st.session_state.canvas_key += 1
        st.session_state.last_prediction = None
        st.session_state.last_log_entry = None
        st.session_state.last_model_choice = None
        st.session_state.last_confidence = None
        st.rerun()

if canvas_result.image_data is not None and identify_clicked:
    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()

    start_time = time.time()

    if model_choice in ["CNN", "DNN"]:
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        preprocess = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()

        confidence = None

    elif model_choice == "Tiny LLM":
        pred, confidence = predict_with_tiny_llm(canvas_result.image_data, model)

    elif model_choice == "Tiny VLM":
        pred, confidence = predict_with_tiny_vlm(canvas_result.image_data, model, vlm_text_processor)

    exec_time = time.time() - start_time
    emissions_value = tracker.stop()

    num_params = sum(p.numel() for p in model.parameters())

    log_entry = log_fingerprint(
        model_name=model_choice,
        num_params=num_params,
        exec_time=exec_time,
        emissions_data=emissions_value,
        tracker=tracker,
        prediction=pred
    )

    st.session_state.run_counts[model_choice] += 1
    st.session_state.last_prediction = pred
    st.session_state.last_log_entry = log_entry
    st.session_state.last_model_choice = model_choice
    st.session_state.last_confidence = confidence

if st.session_state.last_prediction is not None:
    st.success(f"Identification Complete: {st.session_state.last_prediction}")

    if st.session_state.last_model_choice in ["Tiny LLM", "Tiny VLM"] and st.session_state.last_confidence is not None:
        st.info(f"Confidence: {st.session_state.last_confidence:.4f}")

    st.subheader("Hardware Fingerprint Metadata")
    st.json(st.session_state.last_log_entry)

    st.subheader("Progress")
    latest_csv_counts = get_csv_counts()
    st.write(
        f"{st.session_state.last_model_choice}: "
        f"{latest_csv_counts[st.session_state.last_model_choice]} / 1000 saved runs"
    )