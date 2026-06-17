import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

data = {
  "feature1": [1.2, 2.0, 3.1, 4.5, 0.5, 1.8, 2.9, 3.6, 5.1, 2.3],
  "feature2": [3.4, 1.0, 0.9, 2.1, 4.0, 2.2, 1.5, 0.8, 1.1, 3.0],
  "feature3": [2.1, 0.5, 4.2, 3.0, 1.2, 2.8, 3.1, 4.9, 0.2, 1.7],
  "target": [10.5, 5.2, 12.3, 15.1, 6.0, 9.8, 11.4, 14.2, 11.0, 8.9]
}

df = pd.DataFrame(data)
X = df[["feature1", "feature2", "feature3"]]
y = df[["target"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)


X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(3, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 1)
    )
  def forward(self, x):
      return self.network(x)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
   model.train()
   running_train_loss = 0.0
   for batch_x, batch_y in train_loader:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
   
      optimizer.zero_grad()
      y_prediction = model(batch_x)
      loss = criterion(y_prediction, batch_y)
      loss.backward()
      optimizer.step()
      running_train_loss += loss.item() * batch_x.size(0)
      epoch_train_loss = running_train_loss / len(train_loader.dataset)
      
      model.eval()
      with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_test_prediction = model(X_test)
        prediction_loss = criterion(y_test_prediction, y_test)

        if (epoch + 1) % 10 == 0 or epoch == 0:
           print(f"Epoch {epoch + 1 :02d}| Train msf: {epoch_train_loss:.4f}| prediction MSF: {prediction_loss.item():.4f}")
          

new_sample = [[2.5, 1.8, 3.0]]
new_sample_scaled = scalar.transform(new_sample)
new_tensor = torch.tensor(new_sample_scaled, dtype=torch.float32)
new_tensor = new_tensor.to(device)
model.eval()
with torch.no_grad():
  raw_prediction = model(new_tensor)
  final_prediction = raw_prediction.cpu().item()
  print(f"input features: {new_sample[0]}")
  print(f"predicted target values: {final_prediction:.4f}")

  criterion = nn.MSELoss()
  target = torch.tensor([[12.2]], dtype=torch.float32)
  loss = criterion(final_prediction, target)
  print(loss)