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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = ROOT / "checkpoints"


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"{name} RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


def evaluate_cnn():
    print("Evaluating CNN...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_cnn.pth", map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    print_metrics("CNN", y_true, y_pred)


def evaluate_dnn():
    print("Evaluating DNN...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleDNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "mnist_dnn.pth", map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    print_metrics("DNN", y_true, y_pred)


def evaluate_tiny_llm():
    print("Evaluating Tiny LLM...")
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

    test_dataset = MNISTTokenDataset(train=False, root="./data")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_true, y_pred = [], []

    with torch.inference_mode():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(DEVICE)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    print_metrics("Tiny LLM", y_true, y_pred)


def evaluate_tiny_vlm():
    print("Evaluating Tiny VLM...")
    checkpoint = torch.load(CHECKPOINT_DIR / "mnist_tiny_vlm.pth", map_location=DEVICE)

    model = TinyVLM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        text_hidden_dim=checkpoint["text_hidden_dim"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    text_processor = VLMTextProcessor()
    class_text_ids = text_processor.build_all_class_token_ids().to(DEVICE)
    test_dataset = MNISTVLMTrainDataset(train=False, root="./data")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_true, y_pred = [], []

    with torch.inference_mode():
        text_features = model.encode_text(class_text_ids)

        for images, labels in test_loader:
            images = images.to(DEVICE)
            image_features = model.encode_image(images)
            logits = image_features @ text_features.T
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    print_metrics("Tiny VLM", y_true, y_pred)


if __name__ == "__main__":
    evaluate_cnn()
    evaluate_dnn()
    evaluate_tiny_llm()
    evaluate_tiny_vlm()