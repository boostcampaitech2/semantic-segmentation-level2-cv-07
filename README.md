# Level2 P-stage Semantic Segmentation

### π‘ **Team: μ»΄ν¨ν°κ΅¬μ‘°**

## Project Overview

- **Trash semantic segmentation**
- Input: 512 x 512 Image
  - Train: 2617 images
  - Validation: 655 images
- Output: Category classification for each pixel
    - Class(11): Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

## Archive contents

```
image-classification-level1-02/
βββ input/data/
β   βββ batch_01_vt/
β   βββ batch_02_vt/
β   βββ batch_03/
β   βββ train.json
β   βββ val.json
β   βββ train_all.json
β   βββ test.json
βββ mmsegmentation/
β   βββ configs/
β   βββ mmdetection library folders and files ...
β   βββ train.py
β   βββ inference.py
βββ util/
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

converting visualize_CSVs py to ipynb

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
| **κ³ μ¬μ±** [@κ³ μ¬μ±](https://github.com/pkpete)               |
| **κΉμ±λ―Ό** [@ksm0517](https://github.com/ksm0517)             |
| **λ°μ§λ―Ό** [@λ°μ§λ―Ό](https://github.com/ddeokbboki-good)      | 
| **λ°μ§ν** [@ppjh8263](https://github.com/ppjh8263)           |
| **μ¬μΈλ Ή** [@Seryoung Shim](https://github.com/seryoungshim17)| 
| **μ€νμ ** [@Yoon Hajung](https://github.com/YHaJung)         | 

## Data Citation <br/> ```λ€μ΄λ² μ»€λ₯νΈμ¬λ¨ - μ¬νμ© μ°λ κΈ° λ°μ΄ν°μ / CC BY 2.0```
