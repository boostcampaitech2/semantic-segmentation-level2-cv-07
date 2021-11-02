from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOCustom(CustomDataset):
    CLASSES = ["Backgroud", "General trash", "Paper", "Paper pack",
                    "Metal", "Glass", "Plastic", "Styrofoam",
                    "Plastic bag", "Battery", "Clothing"]
    PALETTE = [[0, 0, 0],
                [192, 0, 128],
                [0, 128, 192],
                [0, 128, 64],
                [128, 0, 0],
                [64, 0, 128],
                [64, 0, 192],
                [192, 128, 64],
                [192, 192, 128],
                [64, 64, 128],
                [128, 0, 192]]

    reduce_zero_label=False

    def __init__(self, **kwargs):
        super(COCOCustom, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
            # img_suffix='.jpg', seg_map_suffix='.jpg', **kwargs)
