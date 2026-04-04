import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.model import SimpleCNN, SimpleDNN
from models.tiny_llm_model import TinyLLMClassifier, MNISTTokenDataset
from models.tiny_vlm_model import TinyVLM, VLMTextProcessor, MNISTVLMTrainDataset

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def safe_float(x):
    return float(x) if x is not None else None


def compute_classification_metrics(y_true, y_pred):
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision_weighted": safe_float(precision_w),
        "recall_weighted": safe_float(recall_w),
        "f1_weighted": safe_float(f1_w),
        "precision_macro": safe_float(precision_m),
        "recall_macro": safe_float(recall_m),
        "f1_macro": safe_float(f1_m),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_flops(model_name, model, extra=None):
    if not THOP_AVAILABLE:
        return "Install thop to compute FLOPs"

    model.eval()

    try:
        if model_name in ["CNN", "DNN"]:
            dummy = torch.randn(1, 1, 28, 28).to(DEVICE)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            return int(flops)

        elif model_name == "Tiny LLM":
            vocab_size = extra["vocab_size"]
            seq_len = extra["seq_len"]
            dummy_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)
            flops, _ = profile(model, inputs=(dummy_ids,), verbose=False)
            return int(flops)

        elif model_name == "Tiny VLM":
            vocab_size = extra["vocab_size"]
            dummy_img = torch.randn(1, 1, 28, 28).to(DEVICE)
            dummy_text = torch.randint(0, vocab_size, (1, 2), device=DEVICE)
            flops, _ = profile(model, inputs=(dummy_img, dummy_text), verbose=False)
            return int(flops)

    except Exception as e:
        return f"FLOPs unavailable: {str(e)}"

    return None


def evaluate_cnn_or_dnn(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    return compute_classification_metrics(y_true, y_pred)


def evaluate_tiny_llm(model):
    test_dataset = MNISTTokenDataset(train=False, root="./data")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(DEVICE)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["vocab_size"] = test_dataset.vocab_size
    metrics["seq_len"] = test_dataset.seq_len
    return metrics


def evaluate_tiny_vlm(model):
    text_processor = VLMTextProcessor()
    test_dataset = MNISTVLMTrainDataset(train=False, root="./data")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class_text_ids = text_processor.build_all_class_token_ids().to(DEVICE)

    y_true = []
    y_pred = []

    model.eval()
    with torch.inference_mode():
        text_features = model.encode_text(class_text_ids)

        for images, labels in test_loader:
            images = images.to(DEVICE)
            image_features = model.encode_image(images)
            logits = image_features @ text_features.T
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["vocab_size"] = text_processor.vocab_size
    return metrics


def load_cnn():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_cnn.pth", map_location=DEVICE))
    model.eval()
    return model


def load_dnn():
    model = SimpleDNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_dnn.pth", map_location=DEVICE))
    model.eval()
    return model


def load_llm():
    checkpoint = torch.load(CHECKPOINT_DIR / "mnist_tiny_llm.pth", map_location=DEVICE)
    model = TinyLLMClassifier(
        vocab_size=checkpoint["vocab_size"],
        seq_len=checkpoint["seq_len"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ff_dim=checkpoint["ff_dim"],
        num_classes=checkpoint["num_classes"],
        dropout=0.1
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_vlm():
    checkpoint = torch.load(CHECKPOINT_DIR / "mnist_tiny_vlm.pth", map_location=DEVICE)
    model = TinyVLM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        text_hidden_dim=checkpoint["text_hidden_dim"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    results = {}

    # CNN
    cnn = load_cnn()
    cnn_metrics = evaluate_cnn_or_dnn(cnn)
    cnn_metrics["parameters"] = count_parameters(cnn)
    cnn_metrics["flops"] = compute_flops("CNN", cnn)
    results["CNN"] = cnn_metrics

    # DNN
    dnn = load_dnn()
    dnn_metrics = evaluate_cnn_or_dnn(dnn)
    dnn_metrics["parameters"] = count_parameters(dnn)
    dnn_metrics["flops"] = compute_flops("DNN", dnn)
    results["DNN"] = dnn_metrics

    # Tiny LLM
    llm, llm_checkpoint = load_llm()
    llm_metrics = evaluate_tiny_llm(llm)
    llm_metrics["parameters"] = count_parameters(llm)
    llm_metrics["flops"] = compute_flops(
        "Tiny LLM",
        llm,
        extra={
            "vocab_size": llm_checkpoint["vocab_size"],
            "seq_len": llm_checkpoint["seq_len"]
        }
    )
    results["Tiny LLM"] = llm_metrics

    # Tiny VLM
    vlm, vlm_checkpoint = load_vlm()
    vlm_metrics = evaluate_tiny_vlm(vlm)
    vlm_metrics["parameters"] = count_parameters(vlm)
    vlm_metrics["flops"] = compute_flops(
        "Tiny VLM",
        vlm,
        extra={"vocab_size": vlm_checkpoint["vocab_size"]}
    )
    results["Tiny VLM"] = vlm_metrics

    output_path = LOG_DIR / "model_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved model metrics to {output_path}")


if __name__ == "__main__":
    main()