import torch
import torch.nn as nn
import joblib

torch.manual_seed(42)
sentences = ["I love deep learning", "rnn models read sequentially", "Pytorch makes rnn easy", "python is the best teacher there ever is"]
words_to_idx = {}
idx_to_words = {}

index = 0

for sentence in sentences:
  for word in sentence.split():
    if word not in words_to_idx:
      word = word.lower()
      words_to_idx[word] = index
      idx_to_words[index] = word
      index += 1
vocab_size = len(words_to_idx)
# print(vocab_size)


training_pairs = []
for s in sentences:
  words = s.split()
  for i in range(len(words) - 1):
    input_index = words_to_idx[words[i].lower()]
    target_index = words_to_idx[words[i+1]]
    training_pairs.append([input_index, target_index])


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
  
model = NextWordRNN(vocab_size, embed_dim=8, hidden_size=16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epochs in range(151):
  total_loss = 0
  for input, target in training_pairs:
    input_tensor = torch.tensor([[input]])
    target_tensor = torch.tensor([target])
    
    optimizer.zero_grad()
    out = model(input_tensor).squeeze(1)
    loss = criterion(out, target_tensor)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  if epochs % 50 == 0:
    print(f"Epoch {epochs + 1}: loss {total_loss:.4f}")

def predict_next(word: str):
  word = word.lower().strip()
  if word not in words_to_idx:
    return "Unknown word"
  index = words_to_idx[word]
  input = torch.tensor([[index]])

  model.eval()
  with torch.no_grad():
    out = model(input).squeeze(1)
    predict_next_word = out.argmax(1).item()
  return idx_to_words[predict_next_word]
 
# print(predict_next("python"))

torch.save(model.state_dict(), "next_word_rnn.pth")
gen_meta = {"words_to_idx": words_to_idx, "idx_to_words": idx_to_words}
joblib.dump(gen_meta, "Genmeta.joblib")

  