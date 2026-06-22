import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import joblib

df = pd.read_csv('clean_student_data.csv')

num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].astype('float32')

# df = df.astype(float)
df = df.drop(df[['Gender']], axis=1)

print(df.info())

# Hours_Studied,Attendance,Motivation_Level,Internet_Access,Tutoring_Sessions,Family_Income,Teacher_Quality,School_Type,Peer_Influence,Physical_Activity,Learning_Disabilities,Distance_from_Home,Gender,Exam_Score
X = df[['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Parental_Education_Level', 'Motivation_Level']]
y = df[['Exam_Score']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)


X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(9, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )
  def forward(self, x):
      return self.network(x)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 60
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

        # if (epoch + 1) % 10 == 0 or epoch == 0:
          #  print(f"Epoch {epoch + 1 :02d}| Train msf: {epoch_train_loss:.4f}| prediction MSF: {prediction_loss.item():.4f}")
        

torch.save(model.state_dict(), 'MLP_student_model.pth')
joblib.dump(scalar, "student_scalar.joblib")