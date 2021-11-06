#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mmcv import Config

from mmdet.datasets import (build_dataloader, build_dataset,)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test

from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel

import pandas as pd
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as mutils


# In[2]:


import cv2
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt


# In[3]:


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
# swin_tiny
# cfg = Config.fromfile('./configs/_base_/my_seg/swin.py')
# cfg.work_dir = './work_dirs/swin_base'

# swin_base
# cfg = Config.fromfile('./configs/_base_/my_seg/swin_base.py')
# cfg.work_dir = './work_dirs/real_swin_base'

# swin_large
cfg = Config.fromfile('./configs/_base_/my_seg/swin_large.py')
cfg.work_dir = './work_dirs/real_swin_large'

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 4
cfg.seed = 2021
cfg.gpu_ids = [0]

epoch = 'epoch_16'

cfg.seed = 2021
cfg.gpu_ids = [0]

# image_size 변환 - default : (1024, 1024), 적용시 (2048, 2048)
# cfg.train_pipeline[2].img_scale=(2048,2048)
# cfg.test_pipeline[1].img_scale=(2048,2048)
# cfg.val_pipeline[2].img_scale=(2048,2048)

# neck 변환 - default : FPN, 적용시 PAFPN
# cfg.model.neck=dict(type = 'PAFPN',
#                     in_channels=[96, 192, 384, 768],
#                     out_channels=256,
#                     num_outs=6)

# anchor size, 비율 변환 - default : scales = [8], ratios=[0.5, 1.0, 2.0], strides=[2, 4, 8, 16, 32, 64])
# cfg.model.rpn_head.anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.33, 0.5, 1.0, 2.0, 3.0],
#             strides=[2, 4, 8, 16, 32, 64])

# rpn_proposal 변환 - default : 2000 적용시 : 1000
# cfg.model.train_cfg.rpn.rpn_proposal=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0)

# loss변환 - default : cross entropy 적용시 focal loss
# cfg.model.rpn_head.loss_cls = dict(type='FocalLoss', loss_weight=1.0)
# cfg.model.roi_head.bbox_head[0].loss_cls = dict(
#     type='FocalLoss', loss_weight=1.0)
# cfg.model.roi_head.bbox_head[1].loss_cls = dict(
#     type='FocalLoss', loss_weight=1.0)
# cfg.model.roi_head.bbox_head[2].loss_cls = dict(
#     type='FocalLoss', loss_weight=1.0)

cfg.model.roi_head.bbox_head[0].num_classes = 10
cfg.model.roi_head.bbox_head[1].num_classes = 10
cfg.model.roi_head.bbox_head[2].num_classes = 10
cfg.model.roi_head.mask_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None


# In[4]:


dataset= build_dataset(cfg.data.test)


# In[5]:


# build dataset & dataloader

data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# checkpoint path
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


# In[6]:


output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산


# In[7]:


# show output format
metric = dataset.format_results(output)
print(metric)


# In[8]:


# show output shape
data = np.array(output)
print("array :\n",np.array(data).shape)


# In[9]:


# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()

class_num = 10
size = 256

for i, out in enumerate(output):
    preds_array = np.zeros((size,size), dtype=np.int)
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    confidence = 0
    for j in range(class_num):
        for idx, k in enumerate(out[1][j]) :
            mask_array = cv2.resize(mutils.decode(k)*(j+1), (size,size), interpolation = cv2.INTER_NEAREST).astype(int)
            # 마스크를 덮어쓰기 안함
            # mask_pred = np.where(preds_array == 0, 1, 0)
            # preds_array += mask_pred*mask_array
            
            # 마스크를 덮어쓰기 함
            # mask_pred = np.where(mask_array == 0, 1, 0)
            # preds_array *= mask_pred
            # preds_array += mask_array
            
            # confidence score을 기준으로 덮어쓰기
            if confidence < out[0][j][idx][4] :
                mask_pred = np.where(mask_array == 0, 1, 0)
                preds_array *= mask_pred
                preds_array += mask_array
                confidence = out[0][j][idx][4]
            else :
                mask_pred = np.where(preds_array == 0, 1, 0)
                preds_array += mask_pred*mask_array
            
    file_names.append(image_info['file_name'])
    prediction_strings.append(' '.join(str(e) for e in preds_array.flatten().tolist()))


# In[10]:


submission = pd.DataFrame()
submission['image_id'] = file_names
submission['PredictionString'] = prediction_strings
submission.head()


# In[11]:


submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)


# In[12]:


# tried show img but i found good tool
# img = '/opt/ml/segmentation/input/data/batch_01_vt/0300.jpg'

# model = init_detector(cfg,checkpoint_path)
# result = inference_detector(model,img)

# show_result_pyplot(model, img, result, score_thr=0.6)# show the image with result
# model.show_result(img, result)# save image with result # , out_file='0617.jpg'


# In[13]:


# batch_id=0
# with torch.no_grad():
#     for index, outs in enumerate(output):
#         oms = torch.argmax(outs, dim=1).detach().cpu().numpy()

# fig, ax = plt.subplots(nrows=num_examples, ncols=2, figsize=(10, 4*num_examples), constrained_layout=True)
# for row_num in range(num_examples):
#     # Original Image
#     ax[row_num][0].imshow(temp_images[row_num].permute([1,2,0]))
#     ax[row_num][0].set_title(f"Orignal Image : {image_infos[row_num]['file_name']}")
#     # Pred Mask
#     ax[row_num][1].imshow(label_to_color_image(oms[row_num]))
#     ax[row_num][1].set_title(f"Pred Mask : {image_infos[row_num]['file_name']}")
#     ax[row_num][1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    
# plt.show()

#     fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
#     ax1.imshow(preds_array)
#     ax1.grid(False)
#     plt.show()   

