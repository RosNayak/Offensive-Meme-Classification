from tqdm.auto import tqdm
import functools
from utility import *
from driving_functions import *

EPOCHS = 10
embedding_dim = 768
embedding_model = 'distilbert-base-uncased'
device = select_device()

train = pd.read_csv("Dataset/Training_meme_dataset.csv")
val = pd.read_csv("Dataset/Validation_meme_dataset.csv")

images = "Dataset/Processed Images" #path to the images folder.

train_dataset = get_dataset_bert(train, train_tokens, images)
val_dataset = get_dataset_bert(val, val_tokens, images, state='validation')

embed_model = EmbeddingModel(embedding_model, embedding_dim)
embed_model.eval()
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
collate_fn_bert = functools.partial(collate_fn_bert, tokenizer=tokenizer, embed_model=embed_model)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_bert)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_bert)

model_params = {
    'image_model' : 'swin_small_patch4_window7_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'embedding_dim' : embedding_dim,
    'drop_prob' : 0.2,
    'pretrained' : True
}
model = MemeModelConcatBertAttention(**model_params)
model = model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)

train_losses, val_losses = [], []
model_folder = os.path.join("/content/drive/MyDrive/Image Processing project/Meme Classification", model_params['image_model'] + '_BiLSTM_DistilBert_Attention')
for epoch in range(EPOCHS):
    train_loss = train_fn(train_loader, model, loss_fn, optimizer, epoch, device)
    val_loss, valid_targets, predictions = validation_fn(val_loader, model, loss_fn, epoch, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    outputs, targets = bin_tonp_tocpu(predictions, valid_targets)
    wt_precision, wt_recall, wt_f1 = calc_precision_recall_f1score(outputs, targets)

    if not os.path.exists(model_folder):
      os.mkdir(model_folder)
    
    model_name = f"{model_params['image_model'] + '_BiLSTM_DistilBert'}_epoch_{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(model_folder, model_name))
    print(f'The saved model is: {model_name}')