# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

```latex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

## Results and Models

| Architecture | Backbone  | Style   | Conv         | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:---------:|:-------:|:------------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch |   -          | 1x      | 3.4      | 28.8           | 37.5   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/faster_rcnn_r50_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204_143530.log.json) |
| Mask   R-CNN | R-50      | pytorch |   -          | 1x      | 3.6      | 24.1           | 38.1   | 34.7    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_1x_coco/mask_rcnn_r50_fpn_fp16_1x_coco_20200205-59faf7e4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_1x_coco/mask_rcnn_r50_fpn_fp16_1x_coco_20200205_130539.log.json) |
| Mask   R-CNN | R-50      | pytorch | dconv(c3-c5) | 1x      | 3.0      |                | 41.9   | 37.5    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco_20210520_180247-c06429d2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco_20210520_180247.log.json)  |
| Mask   R-CNN | R-50      | pytorch | mdconv(c3-c5)| 1x      | 3.1      |                | 42.0   | 37.6    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434-cf8fefa5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434.log.json)  |
| Retinanet    | R-50      | pytorch |   -          | 1x      | 2.8      | 31.6           | 36.4   |         |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702_020127.log.json) |
