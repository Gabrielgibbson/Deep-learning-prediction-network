import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import joblib


transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_data = datasets.ImageFolder(r"CNN\mydata\train", transform=transform)
test_data = datasets.ImageFolder(r"CNN\mydata\test", transform=transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

classes = train_data.classes
print(classes)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv = nn.Conv2d(3, 8, 3)
    self.pool = nn.MaxPool2d(2)
    self.fc = nn.Linear(8 * 15 * 15, len(classes))
    # output = input - kernel + 1

  def forward(self, x):
    x = torch.relu(self.conv(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
    
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
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


# img = Image.open(r"CNN\new_cat_again.jpg").convert("RGB")
# img = transform(img).unsqueeze(0)
# model.eval()
# with torch.no_grad():
#     prediction = model(img).argmax(1).item()

# print(classes[prediction])


torch.save(model.state_dict(), 'CNN_model.pth')
metadata = {
  "classes": classes,
  "input_shape": (32, 32)
}

joblib.dump(metadata, "model_metadata.joblib")
