import os
import mmcv
from mmcv import Config
# from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from confusion_matrix import getConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import argparse
import pathlib

import pandas as pd
import numpy as np
import torch
import json
import sys
sys.path.append('../../')
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

def __main__(config_path, checkpoint_path, output_img):
    cfg = Config.fromfile(config_path)

    if 'pretrained' in cfg.model:
        del cfg.model.pretrained

    dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_segmentor(cfg.model)

    CLASSES = ("Backgroud", "General trash", "Paper", "Paper pack",
                        "Metal", "Glass", "Plastic", "Styrofoam",
                        "Plastic bag", "Battery", "Clothing")

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model = MMDataParallel(model.cuda(), device_ids=[0])

    # output = single_gpu_test(model, data_loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confusion_matrix = getConfusionMatrix(model, data_loader, device)
    print(confusion_matrix)

    # Viz
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=CLASSES)
    disp.plot(ax=ax, values_format='')
    plt.savefig(output_img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config file path')
    parser.add_argument(
        '--config', 
        type=pathlib.Path,
        default='../../configs/_base_/my_seg/fcn_hr48_512x512_160k_ade20k.py'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str,
        default='../../work_dirs/fcn_hr48_512x512_160k_ade20k/epo_30_adamW.pth'
    )
    parser.add_argument(
        '--output', 
        type=pathlib.Path,
        default='./saved/hrnet.png'
    )
    args = parser.parse_args()
    
    __main__(
        config_path=args.config, 
        checkpoint_path=args.checkpoint, 
        output_img=args.output
    )