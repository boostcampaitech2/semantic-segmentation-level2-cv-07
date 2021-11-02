from mmseg.models.builder import build_loss
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config
from torchsummary import summary
from datetime import datetime, timedelta



# cfg = Config.fromfile('/opt/ml/semantic-segmentation-level2-cv-07/mmsegmentation/deeplabv4_unet_s5-d16.py')
# cfg = Config.fromfile('deeplabv4_unet_s5-d16.py')
# cfg = Config.fromfile('hrnet.py')
# config_name='ocrnet_hr48'
config_name='swin_ocr'
sub_title='multi_scale'
cfg = Config.fromfile(f'{config_name}.py')

datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]

cfg.seed=2021
cfg.gpu_ids = [0]
now = (datetime.now() + timedelta(hours=9)).strftime('%m%d%H')
work_dir_folder = f'{config_name}_{sub_title}_{now}'
cfg.work_dir = f'./work_dirs/{work_dir_folder}'
#wandb project name setting
cfg.log_config.hooks[1].init_kwargs.project='segmentation'
cfg.log_config.hooks[1].init_kwargs.name=work_dir_folder

model = build_segmentor(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
model.init_weights()
train_segmentor(model, datasets, cfg, distributed=False, validate=True)
