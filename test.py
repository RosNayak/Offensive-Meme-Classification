from tqdm.auto import tqdm
import functools
from utility import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import timm
from driving_functions import test_fn

embedding_dim = 768
embedding_model = 'distilbert-base-uncased'

#model parameters.
model_params1 = {
    'image_model' : 'swin_small_patch4_window7_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'vocab_size' : vocab_length,
    'embedding_dim' : 300,
    'pad_index' : pad_index,
    'drop_prob' : 0.3,
    'pretrained' : False
}
model_params2 = {
    'image_model' : 'vit_base_patch16_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'vocab_size' : vocab_length,
    'embedding_dim' : 300,
    'pad_index' : pad_index,
    'drop_prob' : 0.3,
    'pretrained' : False
}
model_params3 = {
    'image_model' : 'swin_small_patch4_window7_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'embedding_dim' : embedding_dim,
    'drop_prob' : 0.3,
    'pretrained' : False
}
model_params4 = {
    'image_model' : 'vit_base_patch16_224',
    'out_features' : 1,
    'inp_channels' : 3,
    'embedding_dim' : embedding_dim,
    'drop_prob' : 0.3,
    'pretrained' : False
}

device = select_device()

#model paths.
model_paths = [
               "Models/swin_small_patch4_window7_224_BiLSTM_epoch_5.pth",
               "Models/vit_base_patch16_224_BiLSTM_epoch_9.pth",
               "Models/swin_small_patch4_window7_224_BiLSTM_DistilBert_epoch_7.pth",
               "Models/vit_base_patch16_224_BiLSTM_DistilBert_epoch_9.pth"
]

#models.
models = [
        MemeModelConcat(**model_params1),
        MemeModelConcat(**model_params2),
        MemeModelConcatBert(**model_params3),
        MemeModelConcatBert(**model_params4)  
]

#load the model parameters.
models[0].load_state_dict(torch.load(model_paths[0]))
models[1].load_state_dict(torch.load(model_paths[1]))
models[2].load_state_dict(torch.load(model_paths[2]))
models[3].load_state_dict(torch.load(model_paths[3]))

#model1
test_dataset = get_dataset(test, test_sequences, images, state='testing')
collate_fn = functools.partial(collate_fn, pad_index=pad_index)
test_loader = DataLoader(test_dataset, batch_size=149, shuffle=False, collate_fn=collate_fn)
test_targets1, predictions1 = test_fn(test_loader, models[0], device)

#model2
test_dataset = get_dataset(test, test_sequences, images, state='testing')
collate_fn = functools.partial(collate_fn, pad_index=pad_index)
test_loader = DataLoader(test_dataset, batch_size=149, shuffle=False, collate_fn=collate_fn)
test_targets2, predictions2 = test_fn(test_loader, models[1], device)

#model3
test_dataset = get_dataset_bert(test, test_tokens, images, state='testing')
embed_model = EmbeddingModel(embedding_model, embedding_dim)
embed_model.eval()
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
collate_fn = functools.partial(collate_fn_bert, tokenizer=tokenizer, embed_model=embed_model)
test_loader = DataLoader(test_dataset, batch_size=149, shuffle=False, collate_fn=collate_fn)
test_targets3, predictions3 = test_fn(test_loader, models[2], device)

#model4
test_dataset = get_dataset_bert(test, test_tokens, images, state='testing')
embed_model = EmbeddingModel(embedding_model, embedding_dim)
embed_model.eval()
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
collate_fn = functools.partial(collate_fn_bert, tokenizer=tokenizer, embed_model=embed_model)
test_loader = DataLoader(test_dataset, batch_size=149, shuffle=False, collate_fn=collate_fn)
test_targets4, predictions4 = test_fn(test_loader, models[3], device)

#apply sigmoid to calculate probablities.
outputs1 = op_sigmoid(predictions1)
outputs2 = op_sigmoid(predictions2)
outputs3 = op_sigmoid(predictions3)
outputs4 = op_sigmoid(predictions4)

#compute soft voting
final_probs = []
for o1, o2, o3, o4 in zip(outputs1, outputs2, outputs3, outputs4):
  final_probs.append((o1 + o2 + o3 + o4)/4)

#convert probablities to their respective classes.
final_preds = cvt_binary(final_probs)
outputs1 = cvt_binary(outputs1)
outputs2 = cvt_binary(outputs2)
outputs3 = cvt_binary(outputs3)
outputs4 = cvt_binary(outputs4)

#save the outputs.
test['model1'] = outputs1
test['model2'] = outputs2
test['model3'] = outputs3
test['model4'] = outputs4
test['final_probs'] = final_probs
test.to_csv('final_predictions.csv', index=False)