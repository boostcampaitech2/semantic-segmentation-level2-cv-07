from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config

# cfg = Config.fromfile('/opt/ml/semantic-segmentation-level2-cv-07/mmsegmentation/deeplabv4_unet_s5-d16.py')
# cfg = Config.fromfile('deeplabv4_unet_s5-d16.py')
cfg = Config.fromfile('hrnet.py')

datasets = [build_dataset(cfg.data.train)]

cfg.seed=2021
cfg.gpu_ids = [0]


cfg.work_dir = './work_dirs/hrnet'
model = build_segmentor(cfg.model)

# print(datasets[0])


# Add an attribute for visualization convenience
# model.CLASSES = datasets[0].CLASSES
model.CLASSES = ("Backgroud", "General trash", "Paper", "Paper pack",
                    "Metal", "Glass", "Plastic", "Styrofoam",
                    "Plastic bag", "Battery", "Clothing")
# Create work_dir
train_segmentor(model, datasets, cfg, distributed=False, validate=False, 
                meta=dict())