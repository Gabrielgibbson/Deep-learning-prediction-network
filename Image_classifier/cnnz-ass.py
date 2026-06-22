import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import joblib


transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

train_data = datasets.ImageFolder(r"Image_classifier\natural_images\train", transform=transform)
test_data = datasets.ImageFolder(r"Image_classifier\natural_images\test", transform=transform)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

classes = train_data.classes
print(classes)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.fc = nn.Sequential(
        nn.Linear(32 * 14 * 14, 128),
        nn.ReLU(),
        nn.Linear(128, len(classes))
    )

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
  model.train()
  running_loss = 0

  for images, labels in train_loader:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  print(f"Epoch {epoch + 1} completed, loss = {running_loss:.4f}")
   

model.eval()
correct = 0
with torch.no_grad():
  for images, labels in test_loader:
    output = model(images)
    prediction = output.argmax(1)
    correct += (prediction == labels).sum().item()


torch.save(model.state_dict(), 'CNN_model_natural_images.pth')
metadata = {
  "classes": classes,
  "input_shape": (64, 64)
}

joblib.dump(metadata, "model_metadata_natural_images.joblib")
print("Model saved successfully")