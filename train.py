def train_fn(train_loader, model, loss_fn, optimizer, epoch, device, scheduler=None):
    model.train()
    stream = tqdm(train_loader)
    loss_val = 0
    final_loss = 0
    out_st, tar_st = None, None

    for i, (image, text, target) in enumerate(stream, start=1):
        image = image.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float().view(-1, 1)

        output = model(image, text)

        loss = loss_fn(output, target)
        loss_val += loss.item()
        final_loss = loss_val/i
        outputs, targets = bin_tonp_tocpu(output, target)
        if out_st is None:
          out_st, tar_st = outputs.squeeze(1), targets.squeeze(1)
        else:
          out_st, tar_st = torch.cat([out_st, outputs.squeeze(1)]), torch.cat([tar_st, targets.squeeze(1)])
        wt_precision, wt_recall, wt_f1 = calc_precision_recall_f1score(out_st, tar_st)
        stream.set_description(f"Epoch {epoch:02}. Train. Loss {final_loss}. Precision {wt_precision}. Recall {wt_recall}. F1Score {wt_f1}")
          
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return final_loss

def validation_fn(validation_loader, model, loss_fn, epoch, device):
    model.eval()
    stream = tqdm(validation_loader)
    final_targets = []
    final_outputs = []
    loss_val = 0
    final_loss = 0
    out_st, tar_st = None, None
    
    with torch.no_grad():
        for i, (image, text, target) in enumerate(stream, start=1):
            image = image.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float().view(-1, 1)
    
            output = model(image, text)

            loss = loss_fn(output, target)
            loss_val += loss.item()
            final_loss = loss_val/i
            outputs, targets = bin_tonp_tocpu(output, target)
            if out_st is None:
              out_st, tar_st = outputs.squeeze(1), targets.squeeze(1)
            else:
              out_st, tar_st = torch.cat([out_st, outputs.squeeze(1)]), torch.cat([tar_st, targets.squeeze(1)])
            wt_precision, wt_recall, wt_f1 = calc_precision_recall_f1score(out_st, tar_st)
            stream.set_description(f"Epoch: {epoch:02}. Valid. Loss {final_loss}. Precision {wt_precision}. Recall {wt_recall}. F1Score {wt_f1}")
            
            target = (target.detach().cpu().numpy()).tolist()
            output = (output.detach().cpu().numpy()).tolist()
            
            final_targets.extend(target)
            final_outputs.extend(output)
    
    return final_loss, torch.tensor(final_targets), torch.tensor(final_outputs)