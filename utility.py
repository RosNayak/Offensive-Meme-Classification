import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import precision_recall_fscore_support

#Data augmentation.
def train_transform_object(DIM = 384):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM,DIM),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1), p=0.5
            ),
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