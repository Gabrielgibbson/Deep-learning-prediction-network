import joblib
import torch.nn as nn
import torch


torch.manual_seed(42)

training_sentence = ['I love deep learning','RNN models read sequentially', 'PyTorch makes RNNs easy']

training_labels = torch.tensor([0,1,1])
word_to_index = {"<PAD>": 0}
index_to_word = {0: "<PAD>"}

current_index = 1
for sentence in training_sentence:
  words = sentence.split()
  for word in words:
    if word not in word_to_index:
      word_to_index[word] = current_index
      index_to_word[current_index] = word
      current_index += 1

vocabulary_size = len(word_to_index)

maximum_sentence_length = max(len(sentence.split()) for sentence in training_sentence)
encoded_sentences = []
for sentence in training_sentence:
   sentence_numbers = []
   for word in sentence.split():
     sentence_numbers.append(word_to_index[word])
   while len(sentence_numbers) < maximum_sentence_length:
     sentence_numbers.append(0)
   encoded_sentences.append(sentence_numbers)

training_data = torch.tensor(encoded_sentences)

class SentenceClassifier(nn.Module):
  def __init__(self, vocabulary_size, embedding_size, hidden_layer_size, number_of_classes):
    super().__init__()
    self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
    self.rnn_layer = nn.RNN(embedding_size, hidden_layer_size, batch_first=True)
    self.output_layer = nn.Linear(hidden_layer_size, number_of_classes)

  def forward(self, sentence_token):
    embedded_words = self.embedding_layer(sentence_token)
    hidden_state, final_hidden_state = self.rnn_layer(embedded_words)
    final_hidden_state = final_hidden_state.squeeze(0)
    prediction_scores = self.output_layer(final_hidden_state)
    return prediction_scores
  
model = SentenceClassifier(vocabulary_size=vocabulary_size, embedding_size=8, hidden_layer_size=16, number_of_classes=2)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(151):
  optimizer.zero_grad()
  prediction_scores = model(training_data)
  loss = loss_function(prediction_scores, training_labels)
  loss.backward()
  optimizer.step()
  if epoch % 50 == 0:
    print(f"Epoch {epoch}: loss {loss.item():.4f}")

def classify_sentence(sentence):
  sentence_numbers = []
  for word in sentence.lower().split():
    sentence_numbers.append(word_to_index.get(word, 0))
  while len(sentence_numbers) < maximum_sentence_length:
    sentence_numbers.append(0)
  sentence_tensors = torch.tensor(sentence_numbers).unsqueeze(0)

  model.eval()
  with torch.no_grad():
    prediction_scores = model(sentence_tensors)
    prediction_class = prediction_scores.argmax(1).item()
    if prediction_class == 0:
      return 'deep learning topic'
    else:
      return 'rnn/pytorch topic'
    
print(classify_sentence('RNN models reads sequentially'))

torch.save(model.state_dict(), 'rnn_classifier.pth')
metadata = {
  "word_to_index": word_to_index,
  "index_to_word": index_to_word,
  "Max_sentence_length": maximum_sentence_length
}
joblib.dump(metadata, 'classifier_metadata.joblib')