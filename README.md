# ML Model Fingerprinting with CNN, DNN, Tiny LLM, and Tiny VLM

## Overview

This project explores **machine learning model fingerprinting through hardware, runtime, and system-level profiling**.

I compare four model families trained on the MNIST handwritten digit dataset:

- **CNN**
- **DNN**
- **Tiny LLM**
- **Tiny VLM**

My goal is to determine whether I can infer **what type of ML model is running on a machine** based only on measurable runtime and system behavior such as:

- execution time
- CPU usage
- RAM usage
- memory footprint
- clock speed
- device-specific hardware behavior
- energy/emissions data (when available)

This project supports **cross-device data collection** and builds multiple datasets for later classification experiments.

---

## Research Goal

This project is motivated by the idea of **model fingerprinting through side-channel style runtime signals**.

### Core idea

If different model architectures leave behind different hardware/runtime “signatures,” then I may be able to classify:

- **which model family is running**
- **which device is being used**
- or both

without directly inspecting the model itself.

### Example attacker-style framing

An attacker may not have access to the actual `.pth` model file, but they may still observe:

- CPU behavior
- timing behavior
- memory behavior
- system load
- energy usage

From that alone, they may still be able to predict:

- “This is probably a CNN”
- “This is likely a transformer-style tiny LLM”
- “This looks like a VLM-like workload”

This project builds the dataset needed to test that idea.

---

## Devices Used

This project currently includes data from:

- **MacBook Pro**
- **Windows 10 Device #1**
- **Windows 10 Device #2**
- **Windows 11**
- **Jetson Nano**
- **Raspberry Pi**

I use these devices in different dataset pipelines depending on what each device can support.

---

## Dataset Types in This Project

This repository supports **three separate dataset layers**:

### 1. Full Model Fingerprint Dataset
I use this for **full model inference fingerprinting**.

It includes:

- model type
- prediction
- parameters
- execution timing
- CPU / RAM usage
- memory footprint
- energy / emissions (when available)
- model benchmark metrics

Devices used:

- MacBook Pro
- Windows 10
- Windows 11

This is my **richest and most complete dataset**.

---

### 2. Edge Device Profiler Dataset
I use this for **device-only hardware/runtime profiling** on lower-resource devices.

It includes:

- CPU usage
- RAM usage
- CPU clock
- memory footprint
- CPU temperature
- execution timing
- device identity / type

Devices used:

- Jetson Nano
- Raspberry Pi

This dataset does **not** run all four ML models.  
Instead, it captures **device fingerprint behavior** for edge devices.

---

### 3. Hybrid Reduced Dataset
I use this for **cross-device classifier training** across all devices.

I create it by:

- taking the **big/full datasets** (Mac + Windows)
- reducing them down to a **shared subset of columns**
- combining them with the **edge profiler datasets** (Jetson + Raspberry Pi)

This gives me a **balanced, multi-device dataset** where all devices share the same feature schema.

This is likely the best dataset for my **final classifier experiments**.

---

## Project Workflow Summary

There are now **two main collection workflows**:

### Workflow A — Full Model Fingerprinting
For higher-capability machines:

- MacBook Pro
- Windows 10
- Windows 11

### Workflow B — Edge Device Profiling
For lower-resource devices:

- Jetson Nano
- Raspberry Pi

Then later:

### Workflow C — Merge + Reduce + Hybridize
I use this to combine and align everything into classifier-ready datasets.

---

# FULL MODEL FINGERPRINTING WORKFLOW

## Before Data Collection (Per Device)

I use this workflow on devices that can run the full inference pipeline, which currently includes:

- **MacBook Pro**
- **Windows 10**
- **Windows 11**

Before collecting data on a device, I make sure to:

1. **Pull the latest code**
   ```bash
   git pull --rebase

	2.	Activate the environment
Conda:

conda activate fingerprinting

or venv:

source .venv/bin/activate


	3.	Install dependencies

pip install -r requirements.txt


	4.	Verify that all trained model checkpoints exist
Expected files:

checkpoints/mnist_cnn.pth
checkpoints/mnist_dnn.pth
checkpoints/mnist_tiny_llm.pth
checkpoints/mnist_tiny_vlm.pth

If any are missing, I train them using:

python scripts/train.py
python scripts/train_tiny_llm.py
python scripts/train_tiny_vlm.py


	5.	Generate model metrics

python scripts/generate_model_metrics.py

This creates:

logs/model_metrics.json



This setup step ensures that every full-model device is using the same codebase, same checkpoints, and same metric references before I begin collecting fingerprint data.

⸻

Automated Dataset Collection

Once setup is complete, I generate a large automated fingerprint dataset using:

python scripts/generate_fingerprint_dataset.py

This script runs automated inference across all supported model types and logs runtime/system metadata for each run.

Typical output:
	•	1000 samples per model
	•	Total: 4000 rows per automated run

Saved to:

logs/raw_devices/fingerprint_dataset_<device_short>_automated.csv

This automated dataset is one of the main sources for my full model fingerprint dataset.

It captures structured inference runs without requiring manual input, which makes it ideal for building a larger training dataset.

⸻

Manual Dataset Collection (Optional but Recommended)

In addition to automated collection, I can also collect manual inference traces using the Streamlit interface:

python -m streamlit run app.py

This allows me to:
	•	draw digits by hand
	•	choose a model manually
	•	run inference interactively
	•	save additional fingerprinting metadata

Saved to:

logs/raw_devices/fingerprint_dataset_<device_short>_manual.csv

I treat this as a useful supplement to the automated dataset because it adds more natural variation and real user interaction patterns.

While optional, it is still recommended because it helps diversify the fingerprint data.

⸻

Files to Keep After Collection

After finishing collection on a full-model device, I keep the following files:

logs/raw_devices/fingerprint_dataset_<device_short>_automated.csv
logs/raw_devices/fingerprint_dataset_<device_short>_manual.csv
logs/model_metrics.json
logs/powermetrics_log.txt   (if available)
logs/emissions.csv          (if available)

Important notes:
	•	The automated CSV is required for my main model-classification dataset
	•	The manual CSV is optional but helpful
	•	The model metrics JSON should be preserved so I can align model performance with runtime fingerprints
	•	The powermetrics and emissions files may only exist on some devices, depending on platform support

Once I finish collecting data on one device, I move those files into the correct device folder under data/raw/ so they can later be merged into the full master dataset.

⸻

EDGE DEVICE PROFILING WORKFLOW

Why I Use a Separate Edge Pipeline

Jetson Nano and Raspberry Pi may not reliably support:
	•	all PyTorch dependencies
	•	all model checkpoints
	•	all inference workflows
	•	energy/emissions tooling

So instead, I use them to collect a hardware/runtime device fingerprint dataset.

This is still valuable because it gives my classifier cross-device variability and edge-device behavior.

⸻

Edge Device Requirements

I use the lightweight requirements file:

pip install -r requirements-edge.txt

This is designed for profiler-only collection.

⸻

Jetson Nano Device Profiling

I run:

python scripts/jetson_nano_device_profiler.py

This creates:

data/raw/jetson_nano/fingerprint_dataset_<device_short>_jetson_device_profiler.csv

Typical columns include:
	•	timestamp
	•	unique_device_id
	•	device_short_id
	•	pc_name
	•	device_type
	•	collection_mode
	•	sample_index
	•	cpu_usage_pct
	•	ram_usage_pct
	•	cpu_clock_mhz
	•	memory_footprint_mb
	•	execution_time_sec
	•	cpu_temp_c
	•	gpu_model
	•	notes

⸻

Raspberry Pi Device Profiling

I run:

python scripts/raspberry_pi_device_profiler.py

This creates:

data/raw/raspberry_pi/fingerprint_dataset_<device_short>_raspberry_pi_device_profiler.csv

Typical columns include:
	•	timestamp
	•	unique_device_id
	•	device_short_id
	•	pc_name
	•	device_type
	•	collection_mode
	•	sample_index
	•	cpu_usage_pct
	•	ram_usage_pct
	•	cpu_clock_mhz
	•	memory_footprint_mb
	•	execution_time_sec
	•	cpu_temp_c
	•	gpu_model
	•	notes

⸻

Recommended Edge Sample Counts

Recommended collection size:
	•	1000 samples per edge device

This is enough to give me a useful hardware fingerprint baseline without making the dataset too small.

⸻

DATA ORGANIZATION

Recommended Repository Structure

fingerprinting-project/
│
├── app.py
├── README.md
├── requirements.txt
├── requirements-edge.txt
├── .gitignore
│
├── checkpoints/
│   ├── mnist_cnn.pth
│   ├── mnist_dnn.pth
│   ├── mnist_tiny_llm.pth
│   └── mnist_tiny_vlm.pth
│
├── config/
│   └── device_id.txt
│
├── data/
│   ├── raw/
│   │   ├── macbook_air/
│   │   ├── windows_10/
│   │   ├── windows_11/
│   │   ├── jetson_nano/
│   │   └── raspberry_pi/
│   │
│   ├── interim/
│   │   ├── macbook_air_reduced.csv
│   │   ├── windows_10_reduced.csv
│   │   ├── windows_11_reduced.csv
│   │   ├── merged_edge_profiler.csv
│   │   └── hybrid_reduced_cross_device.csv
│   │
│   └── final/
│       ├── classifier_full_model_dataset.csv
│       └── classifier_reduced_device_dataset.csv
│
├── logs/
│   ├── raw_devices/
│   ├── merged/
│   │   └── fingerprint_master_data.csv
│   ├── emissions.csv
│   ├── powermetrics_log.txt
│   └── model_metrics.json
│
├── models/
│   ├── model.py
│   ├── tiny_llm_model.py
│   └── tiny_vlm_model.py
│
├── utils/
│   ├── llm_utils.py
│   └── vlm_utils.py
│
├── scripts/
│   ├── train.py
│   ├── train_tiny_llm.py
│   ├── train_tiny_vlm.py
│   ├── verification.py
│   ├── generate_model_metrics.py
│   ├── generate_fingerprint_dataset.py
│   ├── merge_master_sets.py
│   ├── merge_edge_sets.py
│   ├── reduce_datasets.py
│   ├── merge_hybrid_sets.py
│   ├── jetson_nano_device_profiler.py
│   ├── raspberry_pi_device_profiler.py
│   └── check_model_accuracy.py
│
└── __pycache__/


⸻

Full Model Dataset Merge

After collecting data from MacBook Pro, Windows 10, and Windows 11, I merge those datasets into a single full fingerprint dataset.

I use:

python scripts/merge_master_sets.py

This produces:

logs/merged/fingerprint_master_data.csv

This file contains the full model fingerprint data from the devices that support the main inference workflow.

I use this as the starting point for my main classifier dataset.

⸻

Edge Dataset Merge

After collecting profiler-only data from Jetson Nano and Raspberry Pi, I merge those into an edge-only dataset.

I use:

python scripts/merge_edge_sets.py

This produces:

data/interim/merged_edge_profiler.csv

This dataset is useful for:
	•	Jetson vs Raspberry Pi comparison
	•	edge-device fingerprint analysis
	•	baseline hardware behavior analysis

⸻

Reduced Dataset Creation

Because the MacBook and Windows datasets contain many more columns than Jetson and Raspberry Pi, I create reduced versions of the larger datasets using only the columns shared across all devices.

I use:

python scripts/reduce_datasets.py

This creates:

data/interim/macbook_air_reduced.csv
data/interim/windows_10_reduced.csv
data/interim/windows_11_reduced.csv

These files keep only the shared hardware/runtime columns needed for a cross-device hybrid dataset.

⸻

Hybrid Dataset Merge

Once the reduced Mac/Windows files and the edge profiler files are ready, I merge them into a final reduced cross-device dataset.

I use:

python scripts/merge_hybrid_sets.py

This creates:

data/final/classifier_reduced_device_dataset.csv

This is my cross-device aligned hybrid dataset, which includes:
	•	MacBook Pro
	•	Windows 10
	•	Windows 11
	•	Jetson Nano
	•	Raspberry Pi

using only shared features.

⸻

Final Datasets Produced

At the end of the full workflow, I produce two main final datasets:

1. Full Model Fingerprint Dataset

Used for:
	•	model classification
	•	architecture fingerprinting
	•	inference behavior analysis

Output:

logs/merged/fingerprint_master_data.csv

2. Reduced Cross-Device Hybrid Dataset

Used for:
	•	cross-device experiments
	•	hardware fingerprint comparison
	•	lightweight classifier experiments

Output:

data/final/classifier_reduced_device_dataset.csv


⸻

MODEL DETAILS

CNN

A convolutional neural network trained on MNIST.

This model represents a more traditional image-based deep learning architecture and provides a useful convolutional baseline for fingerprint comparison.

⸻

DNN

A fully connected neural network trained on MNIST.

This serves as a simpler baseline model and helps me compare whether lightweight fully connected architectures produce different runtime signatures from convolutional or transformer-style models.

⸻

Tiny LLM

A small transformer-style model that tokenizes image patches and predicts the digit class.

I use this as an LLM-like proxy architecture for fingerprinting experiments. It is not intended to be a production language model, but rather a lightweight architecture that behaves differently enough from CNNs and DNNs to test fingerprinting hypotheses.

⸻

Tiny VLM

A compact vision-language style model that aligns image embeddings with text prompts such as "digit 0" through "digit 9".

I use this as a multimodal proxy architecture to represent VLM-like behavior in a small, controlled setting.

⸻

MODEL PERFORMANCE METRICS

Classification Metrics

In addition to fingerprinting data, I also evaluate each model using standard classification metrics such as:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix

These help me compare predictive performance across model types.

⸻

Systems / Runtime Metrics

I also track systems-oriented metrics such as:
	•	Parameter Count
	•	FLOPs (Floating Point Operations)
	•	Execution Time
	•	Memory Footprint
	•	CPU Usage
	•	GPU Availability / Model
	•	Estimated Energy Consumption
	•	Estimated Emissions

These are especially important for fingerprinting because they capture the runtime and hardware behavior that my project is trying to model.

⸻

STREAMLIT APP

Running the App

To launch the interactive app, I run:

python -m streamlit run app.py


⸻

In-App Capabilities

The Streamlit app allows me to:
	•	select a model
	•	draw a digit
	•	run inference
	•	clear the canvas and test another digit
	•	log fingerprint metadata
	•	track dataset collection progress
	•	display model performance metrics for the selected architecture

This app is primarily used for manual fingerprint data collection and qualitative testing.

⸻

LOGGED FEATURES

Across the different collection pipelines, the project logs many runtime and fingerprinting features.

Depending on the device and workflow, logged fields may include:
	•	timestamp
	•	unique device identifier
	•	device short ID
	•	host / PC name
	•	device type
	•	collection mode
	•	sample index
	•	true label
	•	model type
	•	parameter count
	•	prediction result
	•	execution time
	•	CPU usage
	•	RAM usage
	•	memory footprint
	•	CPU clock speed
	•	GPU model
	•	CPU temperature
	•	CPU energy estimate
	•	GPU energy estimate
	•	RAM energy estimate
	•	total energy consumed
	•	total emissions

When available, rows may also include model benchmark metrics such as:
	•	model accuracy
	•	weighted precision
	•	weighted recall
	•	weighted F1 score
	•	FLOPs

⸻

PROJECT GOALS

The main goals of this project are to:
	•	train multiple model families on the same task
	•	run inference in a shared environment
	•	log hardware and execution metadata during inference
	•	build a structured fingerprint dataset
	•	analyze whether model families can be identified from their runtime profile
	•	compare behavior across different device types
	•	prepare the data for future classifier training

⸻

FUTURE IMPROVEMENTS

Possible next steps for this project include:
	•	training a classifier directly on fingerprint metadata
	•	evaluating how well model type can be predicted from runtime features
	•	building a visualization dashboard for fingerprint patterns
	•	improving energy and telemetry logging across more devices
	•	adding more model architectures
	•	expanding to larger datasets beyond MNIST
	•	studying adversarial or attacker-style inference scenarios more directly

⸻

NOTES

A few practical notes about this project:
	•	MNIST data is automatically downloaded if not already present
	•	Tiny LLM and Tiny VLM are lightweight proxy architectures for fingerprinting experiments
	•	CodeCarbon is used for emissions and energy estimation when supported
	•	Some telemetry fields are only available on certain platforms
	•	Jetson Nano and Raspberry Pi are intentionally treated as profiler-oriented edge devices
	•	The project is structured so that raw data, reduced data, and final datasets remain separated

⸻

AUTHOR

Nadia de Lafontant

Project focus: model fingerprinting, ML systems profiling, hardware-aware runtime analysis, and cross-device architecture identification.