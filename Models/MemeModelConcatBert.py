import cv2
import math
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import timm

class MemeModelConcatBert(nn.Module):

  def __init__(self, image_model, out_features, inp_channels, embedding_dim, drop_prob=0.2, pretrained=True):
    super().__init__()
    #image pretrained model.
    self.image_model = timm.create_model(image_model, pretrained=pretrained, in_chans=inp_channels)
    n_features = self.image_model.head.in_features
    self.image_model.head = nn.Linear(n_features, 128)

    #language model layers.
    self.lstm_1 = nn.LSTM(embedding_dim, 128, 2, dropout=drop_prob, batch_first=True, bidirectional=True)
    self.lstm_2 = nn.LSTM(256, 64, 2, dropout=drop_prob, batch_first=True, bidirectional=True)

    #feed forward layers.
    self.fc = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Dropout(drop_prob),
      nn.Linear(64, out_features)
    )
    self.dropout = nn.Dropout(drop_prob)

  def forward(self, image, text):
    img_feats = self.image_model(image)
    img_feats = self.dropout(img_feats)
    lstm_out, _ = self.lstm_1(text)
    lstm_out, _ = self.lstm_2(lstm_out)
    text_feats = torch.sum(lstm_out, axis=1)/lstm_out.shape[1] 
    feats = torch.cat([img_feats, text_feats], dim=1)

    output = self.fc(feats)

    return output

class EmbeddingModel(nn.Module):
  def __init__(self, embedding_model, embedding_dim):
    super().__init__()
    self.bert = AutoModel.from_pretrained(embedding_model)

  def forward(self,**xb):
    return self.bert(**xb)[0]