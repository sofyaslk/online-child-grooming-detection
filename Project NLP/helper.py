import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
save_directory = './saved_models'

#function that changes the encoded_label back to category
def predict_sentiment(num):
	if num==0:
		return 'bullying'
	elif num==1:
		return 'hate speech'
	elif num==2:
		return 'homophobia'
	elif num==3:
		return 'none'
	elif num==4:
		return 'racism'
	else:
		return 'sexism'
	
def classify(input_text):
  """
  This method takes in the input text and returns the predicted class and the probability score of each class.
  """
  loaded_tokenizer =  DistilBertTokenizer.from_pretrained(save_directory)
  loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)
  classes = ['bullying', 'hate_speech', 'homophobia', 'none', 'racism', 'sexism']
  if(input_text):
      # inference on input text
      tokens=loaded_tokenizer(input_text, padding=True,truncation=True, return_tensors='tf')
      logits=loaded_model.predict(dict(tokens), verbose=1).logits
      prob=tf.nn.softmax(logits, axis=1).numpy()
      predictions=np.argmax(prob, axis=1)
      
      prob_val = [round(p,4) for p in prob[0]]
      prob_values = dict(map(lambda i,j : (i,j),classes,prob_val))
      
  return predict_sentiment(predictions), prob_values