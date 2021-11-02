import os
import mmcv

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np
import json


# cfg = Config.fromfile('hrnet.py')
cfg = Config.fromfile('swin_ocr.py')


epoch = 'latest'
# epoch = 'epoch_37'
cfg.seed=2021
cfg.gpu_ids = [0]
cfg.work_dir = '/opt/ml/semantic-segmentation-level2-cv-07/mmsegmentation/work_dirs/swin_ocr_multi_scale_103020'
# /work_dirs/swin_ocr_multi_scale_103020/latest.pth
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

dataset = build_dataset(cfg.data.test)
# cfg.data.samples_per_gpu=1
# cfg.data.workers_per_gpu=1
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
model.CLASSES = dataset.CLASSES

model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, data_loader)

# sample_submisson.csv 열기
submission = pd.read_csv('./work_dirs/sample_submission.csv', index_col=None)
json_dir = os.path.join("../input/data/test.json")
with open(json_dir, "r", encoding="utf8") as outfile:
    datas = json.load(outfile)

input_size = 512
output_size = 256
bin_size = input_size // output_size

# PredictionString 대입
for image_id, predict in enumerate(output):
    image_id = datas["images"][image_id]
    file_name = image_id["file_name"]
    
    temp_mask = []
    predict = predict.reshape(1, 512, 512)
    mask = predict.reshape((1, output_size, bin_size, output_size, bin_size)).max(4).max(2) # resize to 256*256
    temp_mask.append(mask)
    oms = np.array(temp_mask)
    oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)

    string = oms.flatten()

    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=False)