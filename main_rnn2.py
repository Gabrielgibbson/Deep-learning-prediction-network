import torch
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, HTTPException
from RNN.rnn2 import SentenceClassifier


app = FastAPI()
metadata = joblib.load('classifier_metadata.joblib')

word_to_index = metadata['word_to_index']
index_to_word = metadata['index_to_word']
maximum_sentence = metadata['Max_sentence_length']
vocabulary_size = len(word_to_index)

device = torch.device('cpu')
model = SentenceClassifier(vocabulary_size=vocabulary_size, embedding_size=8, hidden_layer_size=16, number_of_classes=2)

model.load_state_dict(torch.load('rnn_classifier.pth'))
model.eval()

class SentenceInput(BaseModel):
  sentence: str

def convert_sentence_to_numbers(sentence):
  sentence_numbers = []
  for word in sentence.lower().split():
    sentence_numbers.append(word_to_index.get(word, 0))
  while len(sentence_numbers) < maximum_sentence:
    sentence_numbers.append(0)
  return sentence_numbers

@app.post('/')
def classify_sentence(data: SentenceInput):
  sentence_number = convert_sentence_to_numbers(data.sentence)
  input_tensor = torch.tensor(sentence_number).unsqueeze(0)
  with torch.no_grad():
    prediction_scores = model(input_tensor)
    prediction_class = prediction_scores.argmax(1).item()
  if prediction_class == 0:
    sentence_label = 'deep learning sentence'
  else:
    sentence_label = 'rnn/pytorch learning sentence'
  return {
    'input_sentence': data.sentence,
    'predicted)_topic': sentence_label
  }
    