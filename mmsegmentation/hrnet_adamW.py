#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !apt-get update
# !apt-get install -y libsm6 libxext6 libxrender-dev
# !pip install opencv-python


# In[1]:


#!pip install openmim
#!mim install mmdet


# In[15]:


# 모듈 import

from mmcv import Config
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


# In[16]:


# config file 들고오기
# cfg = Config.fromfile('./configs/_base_/my_seg/upernet_swin_base_wind12_512.py')
cfg = Config.fromfile('./configs/_base_/my_seg/fcn_hr48_512x512_160k_ade20k.py')

cfg.data.samples_per_gpu = 12
cfg.data.workers_per_gpu = 4

cfg.seed = 2021
cfg.gpu_ids = [0]

cfg.work_dir = './work_dirs/fcn_hr48'

cfg.model.decode_head.num_classes=11
# for swin
# cfg.model.auxiliary_head.num_classes=11

# image_size 변환 - default : (1024, 1024), 적용시 (2048, 2048)
# cfg.train_pipeline[2].img_scale=(2048,2048)
# cfg.test_pipeline[1].img_scale=(2048,2048)
# cfg.val_pipeline[2].img_scale=(2048,2048)

# cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=25, interval=1)
cfg.runner = dict(type='EpochBasedRunner', max_epochs=50)
cfg.lr_config =  dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5,
)

cfg.optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# wandb log설정
cfg.log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='EpochWandbLogger',
        init_kwargs=dict(
            project='segmentation',
            name='hr_net_adamW_0.0001'))
])
cfg.workflow = [('train', 1),('val',1)]
cfg.evaluation = dict(interval=1, by_epoch=True, metric='mIoU')
# cfg.norm_cfg = dict(type='BN', requires_grad=True)


# In[17]:


# build_dataset
datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)] # 


# In[18]:


datasets[0]


# In[19]:


# 모델 build 및 pretrained network 불러오기
model = build_segmentor(cfg.model)
model.init_weights()


# In[ ]:


# 모델 학습
train_segmentor(model, datasets, cfg, distributed=False, validate=True)

