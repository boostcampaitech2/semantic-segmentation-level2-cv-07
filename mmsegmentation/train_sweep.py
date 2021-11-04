from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmcv import Config
from datetime import datetime, timedelta
import wandb


# cfg = Config.fromfile('/opt/ml/semantic-segmentation-level2-cv-07/mmsegmentation/deeplabv4_unet_s5-d16.py')
# cfg = Config.fromfile('deeplabv4_unet_s5-d16.py')
# cfg = Config.fromfile('hrnet.py')
# config_name='ocrnet_hr48'
def train():
    wandb.init() 
    params = wandb.config
    config_name='swin_ocr'
    
    drop_rate_swin = params['drop_rate_swin']/10
    drop_rate_fcn = params['drop_rate_fcn']/10
    drop_rate_ocr = params['drop_rate_ocr']/10
    sub_title=f'drop_{drop_rate_swin}_{drop_rate_fcn}_{drop_rate_ocr}'
    print(sub_title)
    cfg = Config.fromfile(f'{config_name}.py')

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]

    cfg.seed=2021
    cfg.gpu_ids = [0]
    now = (datetime.now() + timedelta(hours=9)).strftime('%m%d%H')
    work_dir_folder = f'{config_name}_{sub_title}_{now}'
    cfg.work_dir = f'./work_dirs/{work_dir_folder}'
    
    cfg.model.backbone.drop_rate = drop_rate_swin
    cfg.model.decode_head[0].dropout_ratio = drop_rate_fcn
    cfg.model.decode_head[1].dropout_ratio = drop_rate_ocr

    cfg.log_config.hooks[1].init_kwargs.project='sweep_seg'
    cfg.log_config.hooks[1].init_kwargs.name=work_dir_folder

    model = build_segmentor(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.init_weights()
    train_segmentor(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    count = 100
    sweep_config = {
        "name" : "mmseg drop ratio Hyper Param Tunning",
        "method" : "bayes",
        "parameters" : {
            "drop_rate_swin":{
                "distribution": "int_uniform",
                "min": 0,
                "max": 9
            },
            "drop_rate_fcn":{
                "distribution": "int_uniform",
                "min": 0,
                "max": 9
            },
            "drop_rate_ocr":{
                "distribution": "int_uniform",
                "min": 0,
                "max": 9
            },
            # "loss":{
            #     "values" : ['cross_entropy_loss', 'focal_loss']
            # },
            # "batch_size":{
            #     "values" : [16,32, 64]
            # },
            # "learning_rate":{
            #     "values" : [1e-4, 2e-4, 3e-4]
            # },
            
        },
        "metric":{
            "name": "val/mIoU",
            "goal": "maximize"
        },
    }

    sweep_id = wandb.sweep(sweep_config, 
                       project="sweep_seg")

    wandb.agent(sweep_id, function=train, count=count)