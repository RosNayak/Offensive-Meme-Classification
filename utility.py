import torchtext
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import precision_recall_fscore_support
from Dataset.MemeDataset import MemeTrainSet
from Dataset.MemeDatasetBert import MemeTrainSetBert
from transformers import (AutoModel, AutoTokenizer, 
                          AutoModelForSequenceClassification)


#Device selection.
def select_device():
  if torch.cuda.is_available():
      return torch.device('cuda')
  device = torch.device('cpu')

#metric function.
def calc_precision_recall_f1score(preds, targets):
  targets = targets.detach().cpu().numpy()
  preds = preds.detach().cpu().numpy()
  prf = precision_recall_fscore_support(targets, preds, average='weighted')
  return round(prf[0], 3), round(prf[1], 3), round(prf[2], 3)

#binarize the predictions.
def bin_tonp_tocpu(outputs, targets):
  outputs = torch.sigmoid(outputs)
  targets = targets.detach().cpu().numpy()
  outputs = outputs.detach().cpu().numpy()
  for index in range(len(outputs)):
    if outputs[index] >= 0.5:
      outputs[index] = 1
    else:
      outputs[index] = 0
  return torch.tensor(outputs).int(), torch.tensor(targets).int()

#apply sigmoid on array elementwise.
def op_sigmoid(predictions):
  op = []
  for x in predictions:
    z = 1/(1 + np.exp(-x[0]))
    op.append(z)
  return op

#converts the probablities to their respective classes 0 or 1.
def cvt_binary(probs):
  preds = []
  for prob in probs:
    if prob >= 0.5:
      preds.append(1)
    else:
      preds.append(0)
  return preds

#dataset functions.
def get_dataset(df, sequences, images, state='training'):
    ids = list(df['image_name'])
    image_paths = [os.path.join(images, idx) for idx in ids]
    target = df['label'].values

    if state == 'training':
        transform = train_transform_object(224)
    elif state == 'validation' or state == 'testing':
        transform = valid_transform_object(224)
    else:
        transform = None

    return MemeTrainSet(image_paths, sequences, target, transform)

def get_dataset_bert(df, tokens, images, state='training'):
    ids = list(df['image_name'])
    image_paths = [os.path.join(images, idx) for idx in ids]
    target = df['label'].values

    if state == 'training':
        transform = train_transform_object(224)
    elif state == 'validation' or state == 'testing':
        transform = valid_transform_object(224)
    else:
        transform = None

    return MemeTrainSetBert(image_paths, tokens, target, transform)

#collate function for data loaders.
def collate_fn(batch, pad_index):
  images = torch.stack([ex[0] for ex in batch])
  labels = torch.stack([ex[2] for ex in batch])
  captions = [torch.tensor(ex[1]) for ex in batch]
  captions = nn.utils.rnn.pad_sequence(captions, padding_value=pad_index, batch_first=True)
  return images, captions, labels

def collate_fn_bert(batch, tokenizer, embed_model):
  images = torch.stack([ex[0] for ex in batch])
  labels = torch.stack([ex[2] for ex in batch])

  max_len, captions = -1, []
  for ex in batch:
    captions.append(ex[1])
    max_len = max(max_len, len(ex[1]))

  sequences = tokenizer(captions, return_tensors='pt', 
                                max_length=max_len, is_split_into_words=True,
                                padding='max_length',truncation=True)
  captions = embed_model(**sequences)
  return images, captions, labels

#Data augmentation.
def train_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM,DIM),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Flip(p=0.5),
            albumentations.Blur(blur_limit=3, always_apply=False, p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ]
    )

def valid_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM,DIM),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )