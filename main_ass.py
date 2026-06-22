import torch
import torch.nn as nn
import joblib
from torchvision import transforms
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
from pydantic import BaseModel


class CNN(nn.Module):
  def __init__(self, num_classes):
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
  
app = FastAPI()
try:
  metadata = joblib.load("model_metadata_natural_images.joblib")
  classes = metadata["classes"]
  device = torch.device('cpu')
  model = CNN(num_classes=len(classes)).to(device)
  model.load_state_dict(torch.load("CNN_model_natural_images.pth"))
  model.eval()
except Exception as e:
  print(e)


transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
@app.post("/predict")
async def predict_image(file : UploadFile = File(...)):
  if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
    raise HTTPException(status_code=400, detail="Invalid image format")
  try:
    image_bytes = await file.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor_batch = transform(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
      logits = model(tensor_batch)
      probability = torch.softmax(logits, dim=1)
      confidence, prediction = torch.max(probability, 1)
      return {
        "status":"success",
        "file_name": file.filename,
        "prediction": classes[prediction.item()],
        "confidence percentage": round(confidence.item() * 100 , 2)
      }
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"error{e}")