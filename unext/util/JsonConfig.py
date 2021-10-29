import segmentation_models_pytorch as smp
from easydict import EasyDict
import json
from model import *

def JsonConfig(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))
            
    models = {
        'UneXt50': UneXt50,
        'UneXt50Enc3': UneXt50Enc3,
        'UneXt50SWSL': UneXt50SWSL,
        'UneXt101': UneXt101,
        'UneXt50Bicubic': UneXt50Bicubic
    }
    cfg.model = models[cfg.model_name]
            
    return cfg
                
                
if __name__ == '__main__':
    cfg = JsonConfig(file_path='../config/__base__.json')
    print(cfg)