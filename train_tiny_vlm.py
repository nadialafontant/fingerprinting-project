import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tiny_vlm_model import MNISTVLMTrainDataset, TinyVLM, VLMTextProcessor


def evaluate(model, data_loader, text_processor, device):
    model.eval()
    correct = 0
    total = 0

    class_text_ids = text_processor.build_all_class_token_ids().to(device)

    with torch.no_grad():
        text_features = model.encode_text(class_text_ids)

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            logits = image_features @ text_features.T
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_processor = VLMTextProcessor()

    train_dataset = MNISTVLMTrainDataset(train=True, root="./data")
    test_dataset = MNISTVLMTrainDataset(train=False, root="./data")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = TinyVLM(
        vocab_size=text_processor.vocab_size,
        embed_dim=64,
        text_hidden_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            text_ids = torch.stack(
                [text_processor.build_label_token_ids(int(label.item())) for label in labels],
                dim=0
            ).to(device)

            optimizer.zero_grad()

            logits_image_to_text = model(images, text_ids)
            logits_text_to_image = logits_image_to_text.T

            targets = torch.arange(images.size(0), device=device)

            loss_i = criterion(logits_image_to_text, targets)
            loss_t = criterion(logits_text_to_image, targets)
            loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, text_processor, device)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": text_processor.vocab_size,
            "embed_dim": 64,
            "text_hidden_dim": 64,
            "seq_len": text_processor.seq_len,
            "token_to_id": text_processor.token_to_id,
        },
        "mnist_tiny_vlm.pth"
    )

    print("Saved model to mnist_tiny_vlm.pth")


if __name__ == "__main__":
    train()