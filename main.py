from tqdm.auto import tqdm
import functools

EPOCHS = 10
device = select_device()

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

def collate_fn(batch, pad_index):
  images = torch.stack([ex[0] for ex in batch])
  labels = torch.stack([ex[2] for ex in batch])
  captions = [torch.tensor(ex[1]) for ex in batch]
  captions = nn.utils.rnn.pad_sequence(captions, padding_value=pad_index, batch_first=True)
  return images, captions, labels

train = pd.read_csv("/content/drive/MyDrive/Image Processing project/Meme Classification/Dataset/Split Dataset/Training_meme_dataset.csv")
val = pd.read_csv("/content/drive/MyDrive/Image Processing project/Meme Classification/Dataset/Split Dataset/Validation_meme_dataset.csv")

images = "/content/drive/MyDrive/Image Processing project/Meme Classification/Dataset/Processed Images" #path to the images folder.

train_dataset = get_dataset(train, train_sequences, images)
val_dataset = get_dataset(val, val_sequences, images, state='validation')

collate_fn = functools.partial(collate_fn, pad_index=pad_index)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

model_params = {
    'image_model' : 'swin_small_patch4_window7_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'vocab_size' : vocab_length,
    'embedding_dim' : 300,
    'pad_index' : pad_index,
    'drop_prob' : 0.3,
    'pretrained' : True
}
model = MemeModelConcat(**model_params)
model = model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-6, amsgrad=False)

train_losses, val_losses = [], []
model_folder = os.path.join("/content/drive/MyDrive/Image Processing project/Meme Classification", model_params['image_model'] + '_Bilstm')
for epoch in range(EPOCHS):
    train_loss = train_fn(train_loader, model, loss_fn, optimizer, epoch, device)
    val_loss, valid_targets, predictions = validation_fn(val_loader, model, loss_fn, epoch, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    outputs, targets = bin_tonp_tocpu(predictions, valid_targets)
    wt_precision, wt_recall, wt_f1 = calc_precision_recall_f1score(outputs, targets)

    if not os.path.exists(model_folder):
      os.mkdir(model_folder)
    
    model_name = f"{model_params['image_model'] + '_BiLSTM'}_epoch_{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(model_folder, model_name))
    print(f'The saved model is: {model_name}')