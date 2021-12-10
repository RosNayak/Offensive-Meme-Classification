#import the libraries.
import matplotlib.pyplot as plt
import cv2
import math
import os
import numpy as np
from PIL import Image
import re 
import string
import torchtext
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
 
# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()

def tokenize_texts(csv_file):
  sentences = list(csv_file['sentence'])
  tokens = []
  for sentence in sentences:
    tokens.append(word_tokenize(sentence))
  return tokens

def to_sequence(tokens, vocab):
  return [vocab[token] for token in tokens]

class PreprocessText:
  #convert the chars to lower case in the sentence.
  def to_lower(self, sentence):
    return sentence[0].lower().strip()

  #remove punctuations from the sentence.
  def remove_punct(self, sentence):
    return sentence[0].translate(str.maketrans('', '', string.punctuation))

  #expand the contractions in the english words. don't -> do not.
  def decontracted(self, sentence): 
    # specific 
    sentence = re.sub(r"won\'t", "will not", sentence[0]) 
    sentence = re.sub(r"can\'t", "can not", sentence) 
    # general 
    sentence = re.sub(r"n\'t", " not", sentence) 
    sentence = re.sub(r"\'re", " are", sentence) 
    sentence = re.sub(r"\'s", " is", sentence) 
    sentence = re.sub(r"\'d", " would", sentence) 
    sentence = re.sub(r"\'ll", " will", sentence) 
    sentence = re.sub(r"\'t", " not", sentence) 
    sentence = re.sub(r"\'ve", " have", sentence) 
    sentence = re.sub(r"\'m", " am", sentence) 
    return sentence

  def lemmatization(self, sentence, wnl):
    words = sentence[0].split(' ')
    lem_words = []
    for word in words:
      lem_words.append(wnl.lemmatize(word))
    return ' '.join(lem_words)