import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tiny_llm_model import MNISTTokenDataset, TinyLLMClassifier


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    acc = correct / total
    return avg_loss, acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MNISTTokenDataset(train=True, root="./data")
    test_dataset = MNISTTokenDataset(train=False, root="./data")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = TinyLLMClassifier(
        vocab_size=train_dataset.vocab_size,
        seq_len=train_dataset.seq_len,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        num_classes=10,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": train_dataset.vocab_size,
            "seq_len": train_dataset.seq_len,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "ff_dim": 128,
            "num_classes": 10,
        },
        "mnist_tiny_llm.pth"
    )

    print("Saved model to mnist_tiny_llm.pth")


if __name__ == "__main__":
    train()