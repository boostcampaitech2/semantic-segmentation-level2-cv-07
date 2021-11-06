#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


# 베이스가 될 csv와 특정 라벨을 삽입할 때 쓸 csv를 지정
base_result = './result/uper_swin_SETR_0.721.csv'
implant_result = './result/uper_swin_384_0.724.csv'

# 삽입하고 싶은 라벨
# (0은 사실 10이랑 겹쳐서 쓰면 안되는 경우인데 background implant는 잘 없을 것 같아서 굳이 고려 안했습니다.)
implant_label = '9'

# 파일을 저장할 위치
work_dir = './result'


# In[3]:


# csv를 읽음
base_df = pd.read_csv(base_result)
implant_df = pd.read_csv(implant_result)


# In[4]:


# 이미지 순서에 맞추어서 base_df를 불러옴
for img_number in range(len(base_df)) :
    prediction = ''
    
    # image에 있는 라벨들을 쪼개서 비교하고 삽입할 준비를 함
    base_splited = base_df['PredictionString'][img_number].split()
    implant_splited = implant_df['PredictionString'][img_number].split()
    
    # 8인 경우만 변환해서 prediction을 1줄 만듬
    for base_pixel, implant_pixel in zip(base_splited, implant_splited) :
        if base_pixel == implant_label :
            prediction += (implant_pixel + ' ')
        else :
            prediction += (base_pixel + ' ')
            
    # 만들어진 prediction을 base_df에 넣어줌
    base_df['PredictionString'][img_number] = prediction


# In[5]:


# csv만들기
base_df.to_csv(os.path.join(work_dir, f'implant_{implant_label}.csv'), index=False)

