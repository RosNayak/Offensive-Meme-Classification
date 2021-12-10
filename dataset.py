import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class MemeTrainSet(Dataset):

  def __init__(self, image_paths, image_captions, labels, transform=None):
    self.image_paths = image_paths
    self.image_captions = image_captions
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.image_captions)

  def __getitem__(self, index):
    img = cv2.imread(self.image_paths[index], 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if self.transform is not None:
      img = self.transform(image=img)['image']

    #text.
    caption = self.image_captions[index]
    caption = np.asarray(caption)

    label = torch.tensor(1 if self.labels[index] == 'offensive' else 0).int()

    return (img, caption, label)