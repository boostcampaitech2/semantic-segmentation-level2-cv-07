import torch
import albumentations as A
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.JsonConfig import JsonConfig
from dataset import *

import argparse
import pathlib

def getModel(model, model_path, device):
    # best model 저장된 경로
    # model_path = './saved/fcn8s/FCN_8_epoch_20'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    model.eval()
    return model

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def saveCSV(model, test_loader, device, filename):
    # sample_submisson.csv 열기
    submission = pd.DataFrame()

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({
                                        "image_id" : file_name, 
                                        "PredictionString" : ' '.join(str(e) for e in string.tolist())
                                       }, ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(filename, index=False)
    
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
        default='./saved/Unet-resnet50/Unet-resnet50_10_0.439'
    )
    parser.add_argument(
        '--output', 
        type=pathlib.Path,
        default="./submission/submission.csv"
    )
    args = parser.parse_args()
    
    cfg = JsonConfig(file_path=args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = getModel(
        model = cfg.model,
        model_path = args.model,
        device = device
    )
    test_loader = loadDataLoader(
        dataset_path=cfg.dataset_path, 
        batch_size=cfg.batch,
        train=False
    )
    saveCSV(
        model=model, 
        test_loader=test_loader, 
        device=device,
        filename=args.output
    )
    