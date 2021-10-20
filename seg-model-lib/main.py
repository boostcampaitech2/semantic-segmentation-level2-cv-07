import torch
import numpy as np
import random
from torch import nn

from dataset import *
from train import Train

from util.JsonConfig import JsonConfig
import argparse
import wandb
from wandb_function import wandbInit


def init_seed(random_seed = 2021):
    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main(cfg):
    init_seed(random_seed=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    category_names = ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                      'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
    
    train_loader, val_loader, test_loader = loadDataLoader(
        dataset_path=cfg.dataset_path, 
        batch_size=8
    )
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(params = cfg.model.parameters(), lr = cfg.lr, weight_decay=cfg.weight_decay)
    
    if 'wandb' in cfg:
        wandbInit(
            project=cfg.wandb.project, 
            config=cfg, 
            run_name=cfg.wandb.run_name
        )
    T = Train(
        num_epochs=cfg.epoch, 
        classes=category_names, 
        model=cfg.model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        device=device
    )
    T.train(
        optimizer=optimizer, 
        saved_dir=cfg.saved_dir,
        filename=cfg.filename,
        saveWandb='wandb' in cfg
    )
    
    if 'wandb' in cfg:
        wandb.finish()
    

if __name__ == '__main__':
    cfg = JsonConfig(file_path='./config/__base__.json')
    main(cfg)
