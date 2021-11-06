#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np
import json


# In[2]:


# config file 들고오기
cfg = Config.fromfile('./configs/_base_/my_seg/upernet_swin_base_wind12_512.py')

epoch = 'epoch_28'

cfg.model.decode_head.num_classes=11
cfg.model.auxiliary_head.num_classes=11

# dataset config 수정
# cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 1

cfg.work_dir = './work_dirs/upper_swin_base'

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')


# In[3]:


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


# In[4]:


model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


# In[5]:


output = single_gpu_test(model, data_loader)


# In[22]:


import cv2


# In[23]:

# test.json 열기
json_dir = os.path.join("/opt/ml/segmentation/input/data/test.json")
with open(json_dir, "r", encoding="utf8") as outfile:
    datas = json.load(outfile)
		
# test용 마스크를 생성
for image_id, predict in enumerate(output):
    image_id = datas["images"][image_id]
    file_dir = f"/opt/ml/segmentation/input/mmseg/annotations/test/{image_id['id']:04}.png"
    masks = predict.reshape(1, 512, 512)[0]
    cv2.imwrite(file_dir, masks)


# In[ ]:




