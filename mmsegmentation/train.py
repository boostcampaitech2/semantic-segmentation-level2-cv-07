from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config

cfg = Config.fromfile('deeplabv4_unet_s5-d16.py')

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

cfg.seed=2021
cfg.gpu_ids = [0]


# cfg.work_dir = './work_dirs/swin_decetors'
cfg.work_dir = './work_dirs/test'
# Build the detector
# model = build_segmentor(
    # cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'))
model = build_segmentor(cfg.model)


# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict())