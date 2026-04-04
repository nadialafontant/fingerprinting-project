# ML Model Fingerprinting with CNN, DNN, Tiny LLM, and Tiny VLM

## Overview

This project explores **machine learning model fingerprinting through power and hardware profiling**.  
The system compares several model families trained on the MNIST handwritten digit dataset:

- **CNN**
- **DNN**
- **Tiny LLM**
- **Tiny VLM**

A Streamlit app allows the user to draw a digit, run inference with a selected model, and record metadata such as:

- execution time
- memory usage
- CPU usage
- GPU availability
- estimated energy consumption
- emissions information

The goal is to build a dataset of model execution signatures and use those signatures to distinguish between model architectures based on their runtime and power-related behavior.

---

## Project Goals

- Train multiple model families on the same digit-recognition task
- Run inference in a shared interface
- Log hardware and execution metadata during inference
- Build a structured fingerprint dataset
- Analyze whether model families can be identified from their runtime profile

---

## Repository Structure

```text
fingerprinting-project/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── MNIST/
│       └── raw/
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
│   └── generate_model_metrics.py
│
├── checkpoints/
│   ├── mnist_cnn.pth
│   ├── mnist_dnn.pth
│   ├── mnist_tiny_llm.pth
│   └── mnist_tiny_vlm.pth
│
├── logs/
│   ├── emissions.csv
│   ├── fingerprint_master_data.csv
│   ├── powermetrics_log.txt
│   └── model_metrics.json
│
└── config/
    └── device_id.txt
```

---

## Models Included

### 1. CNN
A convolutional neural network trained on MNIST.

### 2. DNN
A fully connected neural network baseline for MNIST classification.

### 3. Tiny LLM
A small transformer-style model that tokenizes image patches and predicts the digit class.  
This is used to represent an LLM-like architecture in the fingerprinting study.

### 4. Tiny VLM
A compact vision-language model that aligns image embeddings with text prompts such as `"digit 0"` through `"digit 9"`.  
This is used to represent a multimodal architecture in the fingerprinting study.

---

## Model Performance Metrics

In addition to hardware and power-based fingerprinting data, the project also evaluates each model using standard classification and complexity metrics.

### Classification Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### Model Complexity / Systems Metrics

- **Parameter Count**
- **FLOPs (Floating Point Operations)**
- **Execution Time**
- **Memory Footprint**
- **CPU Usage**
- **GPU Availability / Model**
- **Estimated Energy Consumption**
- **Estimated Emissions**

These metrics make it possible to compare not only how well each model predicts digits, but also how expensive each model is to run.

---

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries used:

- `streamlit`
- `streamlit-drawable-canvas`
- `torch`
- `torchvision`
- `torchaudio`
- `pillow`
- `numpy`
- `pandas`
- `psutil`
- `codecarbon`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `thop`

---

## Environment Setup

### Option 1: Conda

Create and activate an environment:

```bash
conda create -n fingerprinting python=3.12 -y
conda activate fingerprinting
pip install -r requirements.txt
```

### Option 2: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training the Models

### Train CNN and DNN

```bash
python scripts/train.py
```

### Train Tiny LLM

```bash
python scripts/train_tiny_llm.py
```

### Train Tiny VLM

```bash
python scripts/train_tiny_vlm.py
```

After training, saved weights should appear in `checkpoints/`.

---

## Generating Model Metrics

After training the models, generate evaluation and complexity metrics with:

```bash
python scripts/generate_model_metrics.py
```

This script evaluates all trained models and saves the results to:

```text
logs/model_metrics.json
```

The generated metrics include:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- parameter count
- FLOPs

These metrics are then loaded by the Streamlit app and displayed alongside inference results.

---

## Running the App

Start the Streamlit interface with:

```bash
python -m streamlit run app.py
```

The app allows the user to:

- select a model
- draw a digit
- run inference
- clear the canvas and test another number
- log fingerprint metadata
- track progress toward a target dataset size
- display model performance metrics for the selected architecture

---

## Logged Metadata

Each inference run appends a row to:

```text
logs/fingerprint_master_data.csv
```

Logged runtime and fingerprinting features include:

- timestamp
- device identifier
- model type
- number of parameters
- prediction result
- execution time
- CPU usage
- RAM usage
- memory footprint
- CPU clock speed
- GPU name
- CPU energy estimate
- GPU energy estimate
- RAM energy estimate
- total energy consumed
- total emissions

When available, each inference row can also include model-level benchmark metrics such as:

- model accuracy
- weighted precision
- weighted recall
- weighted F1 score
- FLOPs

---

## In-App Metrics Display

The Streamlit app displays model performance metrics for the currently selected model, including:

- Accuracy
- Precision
- Recall
- F1 Score
- Parameter Count
- FLOPs
- Confusion Matrix

This allows the user to compare **predictive performance** and **systems behavior** in the same interface.

---

## Fingerprinting Dataset Collection

The project is designed to collect approximately **1000 runs per model**.  
The Streamlit sidebar tracks:

- saved dataset counts
- current session counts

This supports controlled collection of inference traces for later analysis.

---

## Notes

- MNIST data is automatically downloaded if not already present
- Tiny LLM and Tiny VLM are lightweight proxy architectures for fingerprinting experiments
- CodeCarbon is used for emissions and energy estimation during inference
- Caching is used in the Streamlit app to improve prediction speed
- Model performance metrics are generated separately and then loaded into the app for display and logging

---

## Future Improvements

- automated batch collection of fingerprint traces
- visualization dashboard for power signatures
- classifier trained on fingerprint metadata
- support for additional model architectures
- more robust GPU and system telemetry logging
- richer visualization of confusion matrices and benchmarking summaries

---

## Author

**Nadia de Lafontant**

Project focus: model fingerprinting, ML systems profiling, and architecture-aware runtime analysis.
