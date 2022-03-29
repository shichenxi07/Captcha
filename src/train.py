# -*- coding = utf-8 -*-
# Time ：2022/3/28 18:23
# @ Author: Sccc
# @ File:train.py


import os
import glob
import torch
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn import metrics


import config
import dataset
from model import CaptchaModel
import engine




def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR,"*.png"))
    # "/../.../abcde.png"
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    # abcde ->[a,b,c,d,e]
    targets = [[c for c in x] for x in targets_orig]
    target_flat = [c for clist in targets for c in clist]

    # labelencoder
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(target_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    )= model_selection.train_test_split(
        image_files,targets_enc,targets_orig,test_size=0.1,random_state=42
    )

    train_dataset = dataset.Classification(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCHSIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    test_dataset = dataset.Classification(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCHSIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8,patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)




    # print(targets_enc)
    # print(len(lbl_enc.classes_))
    # print(np.unique(target_flat))

if __name__ =="__main__":
    run_training()


