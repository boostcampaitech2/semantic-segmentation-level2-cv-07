import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from dataset import *

from util.JsonConfig import JsonConfig
import argparse
import pathlib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from itertools import chain
    
def valid(model, val_loader, device):
    print("Start validation")
    model.eval()
    cm = np.zeros((11, 11), dtype=int)

    with torch.no_grad():
        n_class = 11
        labels = []
        predictions = []
        
        for (images, masks, _) in tqdm(val_loader):

            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            

            # device 할당
            model = model.to(device)
            outputs = model(images)

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            for (mask_img, output_img) in zip(masks, outputs):
                for (mask, output) in zip(chain.from_iterable(mask_img), chain.from_iterable(output_img)):
                    if mask!=output:
                        cm[mask][output] += 1
    return cm

def main(cfg, model_path, output_img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    category_names = ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                      'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
    
    _, val_loader = loadDataLoader(
        dataset_path=cfg.dataset_path, 
        batch_size=cfg.batch,
        num_workers=cfg.num_workers
    )
    
    model=cfg.model
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    print("End loading weights")
    
    cm = valid(model, val_loader, device)
    print("End calculating confusion matrix")

    # Viz
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=category_names)
    disp.plot(ax=ax, values_format='')
    plt.savefig(output_img)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config file path')
    parser.add_argument(
        '--config', 
        type=pathlib.Path,
        default='./config/__base__.json'
    )
    parser.add_argument(
        '--model', 
        type=pathlib.Path,
        default='./saved/DeepLabV3Plus-xception/DeepLabV3Plus_76_0.576.pth'
    )
    parser.add_argument(
        '--output', 
        type=pathlib.Path,
        default='./DeeplabPlus-xception.png'
    )
    args = parser.parse_args()
    
    cfg = JsonConfig(file_path=args.config)
    main(
        cfg, 
        model_path=args.model, 
        output_img=args.output
    )
