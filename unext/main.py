import torch
import numpy as np
import random
from torch import nn

from dataset import *
from train import Train

from util.JsonConfig import JsonConfig
import argparse
import pathlib
import wandb
from util.wandb_function import wandbInit



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
    
    train_loader, val_loader = loadDataLoader(
        dataset_path=cfg.dataset_path, 
        batch_size=cfg.batch,
        num_workers=cfg.num_workers
    )
    
    model = cfg.model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = cfg.lr, weight_decay=cfg.weight_decay)
    
    if 'wandb' in cfg:
        wandbInit(
            project=cfg.wandb.project, 
            config=cfg, 
            run_name=cfg.wandb.run_name
        )
    T = Train(
        num_epochs=cfg.epoch, 
        classes=category_names, 
        model=model, 
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
    parser = argparse.ArgumentParser(description='config file path')
    parser.add_argument(
        '--config', 
        type=pathlib.Path,
        default='./config/__base__.json'
    )
    args = parser.parse_args()
    
    cfg = JsonConfig(file_path=args.config)
    main(cfg)
