# -*- coding = utf-8 -*-
# Time ：2022/3/29 11:47
# @ Author: Sccc
# @ File:engine.py


from tqdm import tqdm
import torch
import config

def train_fn(model, data_loader, optimizer):
    model.train()
    final_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss/len(data_loader)

def eval_fn(model, data_loader, optimizer):
    model.eval()
    final_loss = 0
    final_preds = []
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        batch_preds, loss = model(**data)

        final_loss += loss.item()
        final_preds.append(batch_preds)
    return final_preds, final_loss/len(data_loader)