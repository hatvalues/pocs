import torch
from torch import load as torch_load
from app.mnist_demo_helpers import dev, criterion, test_loader
from app.fixed_models.cnn_classifier import OneTwentyEightClassifier

model_path = (
    "models/onetwentyeight_20250508_091456.pth"  # adjust to your saved model path
)

model = OneTwentyEightClassifier().to(dev)
model.load_state_dict(torch_load(model_path, map_location=dev))
model.eval()

test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(dev), labels.to(dev)
        outputs = model(data)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * data.size(0)  # accumulate total loss
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_loss = test_loss / total
accuracy = correct / total * 100

print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")