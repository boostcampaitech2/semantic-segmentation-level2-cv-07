#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !apt-get update
# !apt-get install -y libsm6 libxext6 libxrender-dev
# !pip install opencv-python


# In[2]:


#!pip install openmim
#!mim install mmdet


# In[3]:


# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


# In[4]:


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/_base_/my_seg/swin_base.py')

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 4

cfg.seed = 2021
cfg.gpu_ids = [0]

cfg.work_dir = './work_dirs/real_swin_base'

cfg.model.roi_head.bbox_head[0].num_classes = 10
cfg.model.roi_head.bbox_head[1].num_classes = 10
cfg.model.roi_head.bbox_head[2].num_classes = 10
cfg.model.roi_head.mask_head.num_classes = 10

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

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=-1, interval=1)
cfg.runner = dict(type='EpochBasedRunner', max_epochs=30)
cfg.lr_config =  dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5,
)

# wandb log설정
cfg.log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='EpochWandbLogger',
        init_kwargs=dict(
            project='Swin-seg',
            name='swin_base'))
])
cfg.workflow = workflow = [('train', 1),('val',1)]


# In[5]:


# build_dataset
datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)] # 


# In[6]:


datasets[0]


# In[7]:


# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()


# In[8]:


# 모델 학습
train_detector(model, datasets, cfg, distributed=False, validate=True)

