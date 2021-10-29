import numpy as np
import torch
from itertools import chain
from tqdm import tqdm
import sys
sys.path.append('../../')
from mmseg.apis import single_gpu_test
import cv2

def getConfusionMatrix(model, val_loader, device):
    print("Start validation")
    cm = np.zeros((11, 11), dtype=int)

    output = single_gpu_test(model, val_loader)
    print()
    for image_id, predict in tqdm(enumerate(output)):
        filepath = f'/opt/ml/segmentation/input/mmseg/annotations/validation/{image_id:04d}.png'
        label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        for (mask, output) in zip(chain.from_iterable(label), chain.from_iterable(predict)):
            if mask!=output:
                cm[mask][output] += 1
            
    return cm