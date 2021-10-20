import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/_base_/my_train/cascade_rcnn_swin-t-p4-w7_fpn_40epo_coco.py')

cfg.data.samples_per_gpu = 10
cfg.data.workers_per_gpu = 8

epoch = 'latest'

cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/small_various_anchor_swi_casca_cosann_PAFPN_ima_2048_low_propos'

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

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.model.train_cfg = None

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
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

output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()

class_num = 10
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
    file_names.append(image_info['file_name'])
    prediction_strings.append(prediction_string)
    

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
submission.head()