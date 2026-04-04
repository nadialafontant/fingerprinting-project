import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from thop import profile

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.model import SimpleCNN, SimpleDNN
from models.tiny_llm_model import TinyLLMClassifier, MNISTTokenDataset
from models.tiny_vlm_model import TinyVLM, VLMTextProcessor
from utils.llm_utils import predict_with_tiny_llm
from utils.vlm_utils import predict_with_tiny_vlm, load_tiny_vlm_model

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = LOG_DIR / "model_metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def safe_flops(model, sample_input):
    """
    Returns FLOPs as an int if possible, else None.
    """
    try:
        flops, params = profile(model, inputs=(sample_input,), verbose=False)
        return int(flops)
    except Exception as e:
        print(f"Could not compute FLOPs for {type(model).__name__}: {e}")
        return None


def evaluate_standard_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def evaluate_tiny_llm(model, helper_dataset, base_dataset):
    model.eval()
    y_true, y_pred = [], []

    for image_tensor, label in base_dataset:
        canvas_array = (image_tensor.squeeze(0).numpy() * 255).astype("uint8")
        pred, confidence = predict_with_tiny_llm(canvas_array, model, helper_dataset)
        y_true.append(int(label))
        y_pred.append(int(pred))

    return y_true, y_pred


def evaluate_tiny_vlm(model, cached_text_features, base_dataset):
    model.eval()
    y_true, y_pred = [], []

    for image_tensor, label in base_dataset:
        canvas_array = (image_tensor.squeeze(0).numpy() * 255).astype("uint8")
        pred, confidence = predict_with_tiny_vlm(canvas_array, model, cached_text_features)
        y_true.append(int(label))
        y_pred.append(int(pred))

    return y_true, y_pred


def build_metrics_dict(y_true, y_pred, parameters, flops):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "parameters": int(parameters),
        "flops": flops
    }


def main():
    print(f"Using device: {DEVICE}")

    # Shared MNIST test set
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset_standard = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=standard_transform
    )

    test_loader = DataLoader(test_dataset_standard, batch_size=128, shuffle=False)

    # Raw tensor version for LLM/VLM helpers
    base_dataset_raw = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    results = {}

    # ---------------- CNN ----------------
    print("Evaluating CNN...")
    cnn = SimpleCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_cnn.pth", map_location=DEVICE))
    cnn.eval()

    y_true, y_pred = evaluate_standard_model(cnn, test_loader)
    cnn_params = count_parameters(cnn)
    cnn_sample = torch.randn(1, 1, 28, 28).to(DEVICE)
    cnn_flops = safe_flops(cnn, cnn_sample)

    results["CNN"] = build_metrics_dict(y_true, y_pred, cnn_params, cnn_flops)

    # ---------------- DNN ----------------
    print("Evaluating DNN...")
    dnn = SimpleDNN().to(DEVICE)
    dnn.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_dnn.pth", map_location=DEVICE))
    dnn.eval()

    y_true, y_pred = evaluate_standard_model(dnn, test_loader)
    dnn_params = count_parameters(dnn)
    dnn_sample = torch.randn(1, 1, 28, 28).to(DEVICE)
    dnn_flops = safe_flops(dnn, dnn_sample)

    results["DNN"] = build_metrics_dict(y_true, y_pred, dnn_params, dnn_flops)

    # ---------------- Tiny LLM ----------------
    print("Evaluating Tiny LLM...")
    llm_checkpoint = torch.load(CHECKPOINT_DIR / "mnist_tiny_llm.pth", map_location=DEVICE)

    llm = TinyLLMClassifier(
        vocab_size=llm_checkpoint["vocab_size"],
        seq_len=llm_checkpoint["seq_len"],
        embed_dim=llm_checkpoint["embed_dim"],
        num_heads=llm_checkpoint["num_heads"],
        num_layers=llm_checkpoint["num_layers"],
        ff_dim=llm_checkpoint["ff_dim"],
        num_classes=llm_checkpoint["num_classes"],
        dropout=0.1
    ).to(DEVICE)

    llm.load_state_dict(llm_checkpoint["model_state_dict"])
    llm.eval()

    llm_helper_dataset = MNISTTokenDataset(train=False, root="./data")

    y_true, y_pred = evaluate_tiny_llm(llm, llm_helper_dataset, base_dataset_raw)
    llm_params = count_parameters(llm)

    # Use token input shape for FLOPs
    llm_seq_len = llm_checkpoint["seq_len"]
    llm_sample = torch.randint(0, llm_checkpoint["vocab_size"], (1, llm_seq_len)).to(DEVICE)
    llm_flops = safe_flops(llm, llm_sample)

    results["Tiny LLM"] = build_metrics_dict(y_true, y_pred, llm_params, llm_flops)

    # ---------------- Tiny VLM ----------------
    print("Evaluating Tiny VLM...")
    vlm, vlm_text_processor, cached_text_features = load_tiny_vlm_model(str(CHECKPOINT_DIR / "mnist_tiny_vlm.pth"))
    vlm.eval()

    y_true, y_pred = evaluate_tiny_vlm(vlm, cached_text_features, base_dataset_raw)
    vlm_params = count_parameters(vlm)

    # Tiny VLM likely expects image tensor input
    vlm_sample = torch.randn(1, 1, 28, 28).to(DEVICE)
    vlm_flops = safe_flops(vlm, vlm_sample)

    results["Tiny VLM"] = build_metrics_dict(y_true, y_pred, vlm_params, vlm_flops)

    # Save JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved model metrics to: {OUTPUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()