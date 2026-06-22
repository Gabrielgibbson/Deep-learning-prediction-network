import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class NextWordRNN(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.embedding(x)
    out, h = self.rnn(x)
    out = self.fc(out)
    return out
  
app = FastAPI()
meta = joblib.load("Genmeta.joblib")
word_to_index = meta["words_to_idx"]
idx_to_words = meta["idx_to_words"]
vocab_size = len(word_to_index)
device = torch.device('cpu')

model = NextWordRNN(vocab_size, embed_dim=8, hidden_size=16)
model.load_state_dict(torch.load("next_word_rnn.pth"))
model.eval()

class WordInput(BaseModel):
  word: str

@app.post("/predict")
def predict_next_word(data: WordInput):
  new_word = data.word.lower().strip()
  if new_word not in word_to_index:
    raise HTTPException(status_code=400, detail=f"{new_word} not in sentence")
  index = word_to_index[new_word]
  input_tensor = torch.tensor([[index]])
  with torch.no_grad():
    out = model(input_tensor).squeeze(1)
    prediction_index = out.argmax(1).item()
  return {
    "input word": data.word,
    "predicted next word":  idx_to_words[prediction_index]
    }