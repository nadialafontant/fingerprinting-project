import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from model import SimpleCNN, SimpleDNN
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
    # Check if we have already assigned an ID to this PC
    if os.path.exists(id_file):
        with open(id_file, "r") as f:
            return f.read().strip()
    else:
        # Generate a new unique ID and save it
        new_id = str(uuid.uuid4())
        with open(id_file, "w") as f:
            f.write(new_id)
        return new_id

# Assign the ID when the app starts
DEVICE_UUID = get_device_id()

# Display Device Info in the sidebar
st.sidebar.header("Device Identity")
st.sidebar.text(f"ID: {DEVICE_UUID}")
st.sidebar.text(f"Device: {os.getenv('COMPUTERNAME', 'Unknown')}")

def log_fingerprint(model_name, num_params, exec_time, emissions_data, tracker, prediction):
    # --- Gather Hardware Metadata ---
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
    
    # GPU Metadata (if available)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Only)"
    
    # --- Create the Data Row ---
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "pc_name": os.getenv("COMPUTERNAME", "Unknown_PC"),
        "model_type": model_name,
        "parameters": num_params,
        "prediction": prediction,
        "execution_time_sec": round(exec_time, 4),
        
        # Power & Emissions (from CodeCarbon)
        "cpu_energy_kwh": getattr(tracker.final_emissions_data, 'cpu_energy', 0),
        "gpu_energy_kwh": getattr(tracker.final_emissions_data, 'gpu_energy', 0),
        "ram_energy_kwh": getattr(tracker.final_emissions_data, 'ram_energy', 0),
        "total_energy_kwh": getattr(tracker.final_emissions_data, 'energy_consumed', 0),
        "total_emissions_kg": emissions_data,
        
        # Real-time Hardware Stats
        "cpu_usage_pct": cpu_usage,
        "gpu_model": gpu_name,
        "ram_usage_pct": ram_usage,
        "cpu_clock_mhz": cpu_freq,
        "memory_footprint_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    }

    # --- Save with Pandas ---
    df = pd.DataFrame([metadata])
    file_path = "fingerprint_master_data.csv"
    
    # Append to file: if file doesn't exist, write header; else, don't.
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    return metadata

st.title("ML Fingerprinter")

# Select Model
model_choice = st.selectbox("Select Model to Fingerprint", ("CNN", "DNN"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
if model_choice == "CNN":
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
else:
    model = SimpleDNN().to(device)
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location=device))
model.eval()

# Drawing Canvas
canvas_result = st_canvas(stroke_width=20, stroke_color="#FFF", background_color="#000", height=280, width=280, key="canvas")

if canvas_result.image_data is not None and st.button("Identify"):
    # Preprocess
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    preprocess = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Predict & Log
    # Initialize tracker for this specific inference event
    # We use a temporary file for the internal tracker log to keep things clean
    # 'save_to_file=False' prevents it from creating a second messy CSV every time you draw.
    # We are manually logging to our own 'fingerprint_master_data.csv' anyway.
    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    exec_time = time.time() - start_time
    
    # Stop tracker and capture emissions
    emissions_value = tracker.stop()
    
    # --- CALL THE HELPER ---
    # We pass the results into your logging function to save to CSV
    log_entry = log_fingerprint(
        model_name=model_choice,
        num_params=sum(p.numel() for p in model.parameters()),
        exec_time=exec_time,
        emissions_data=emissions_value,
        tracker=tracker,
        prediction=pred
    )
    
    # Display results to the user
    st.success(f"Identification Complete: {pred}")
    st.subheader("Hardware Fingerprint Metadata")
    st.json(log_entry) # This now correctly shows the data saved to the CSV