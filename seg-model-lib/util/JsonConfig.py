import segmentation_models_pytorch as smp
from easydict import EasyDict
import json

def JsonConfig(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))
            
    models = {
        'Unet': smp.Unet, 'UnetPlusPlus': smp.UnetPlusPlus, 'Linknet': smp.Linknet, 'FPN': smp.FPN, 
        'PSPNet': smp.PSPNet, 'DeepLabV3': smp.DeepLabV3, 'DeepLabV3Plus': smp.DeepLabV3Plus, 'PAN': smp.PAN
    }
    model = smp.create_model(
        arch=cfg.model.name,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
    )
    cfg.model = model
            
    return cfg
                
                
if __name__ == '__main__':
    cfg = JsonConfig(file_path='../config/__base__.json')
    print(cfg)