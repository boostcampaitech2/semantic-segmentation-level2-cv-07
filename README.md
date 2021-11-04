# Level2 P-stage Semantic Segmentation

### 💡 **Team: 컴퓨터구조**

## Project Overview

- **Predict Trash Objects**
- Input: 512 x 512 Image
  - Train: 2617 images
  - Validation: 655 images
- Output: Category classification for each pixel
    - Class(11): Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

## Archive contents

```
image-classification-level1-02/
├── input/data/
│   ├── batch_01_vt/
│   ├── batch_02_vt/
│   ├── batch_03/
│   ├── train.json
│   ├── val.json
│   ├── train_all.json
│   └── test.json
├── mmsegmentation/
│   ├── configs/
│   ├── mmdetection library folders and files ...
│   ├── train.py
│   └── inference.py
└── util/
```

- ```input/data/``` : download from [https://stages.ai/](https://stages.ai/)

## get start

### train & inference
```
cd mmsegmentation

python train.py
python inference.py
```

### visualize
```
cd util

jupyter notebook visualize_CSVs.ipynb
```
```python
# set result csv files
csv_names = ['Unet_resnet50.csv', 'deeplabv3_resnet50_31epoch.csv']
```
<img src="https://user-images.githubusercontent.com/85881032/139802682-d89814ac-d5f5-4d70-babd-72d1fa92f3ca.png" width="500"/>

### Requirements

- Ubuntu 18.04.5
- Python 3.8.5
- pytorch 1.7.1
- torchvision 0.8.2

Install packages :  `pip install -r requirements.txt` 

#### Hardware

- CPU: 8 x Intel(R) Xeon(R) Gold 5220 CPU
- GPU: V100
- RAM: 88GB


## Contributors

| **Name** @github                                              | 
| ------------------------------------------------------------  | 
| **고재욱** [@고재욱](https://github.com/pkpete)               |
| **김성민** [@ksm0517](https://github.com/ksm0517)             |
| **박지민** [@박지민](https://github.com/ddeokbboki-good)      | 
| **박진형** [@ppjh8263](https://github.com/ppjh8263)           |
| **심세령** [@Seryoung Shim](https://github.com/seryoungshim17)| 
| **윤하정** [@Yoon Hajung](https://github.com/YHaJung)         | 

## Data Citation ```네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0```
