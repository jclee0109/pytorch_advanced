#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
from torchvision import models, transforms

# 파이토치 버전 확인
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# In[3]:


# VGG-16 모델의 인스턴스 생성
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval() # 추론 모드 (평가 모드)로 설정
print(net)


# In[5]:


class BaseTransform():
    """
    화상 크기 변경 및 색상 표준화
    Attributes
    -------------------
    resize : int
        크기 변경 전의 화상 크기
    mean : (R, G, B)
        각 색상 채널의 평균값
    std : (R, G, B)
        각 색상 채널의 표준편차
    """
    
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize), # 화상 크기 변경
            transforms.CenterCrop(resize), # 화상 중앙을 resize*resize로 자름
            transforms.ToTensor(), # 토치 텐서로 변환
            transforms.Normalize(mean, std) # 색상 정보 표준화
        ])
    
    def __call__(self, img):
        return self.base_transform(img)
    


# In[6]:


# 화상 전처리 확인

# 1. 이미지 읽기
image_file_path = "./data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path) # [높이][너비][색RGB] , pillow라서 : from PIL import Image

# 2. 원본 이미지 확인
plt.imshow(img)
plt.show()

# 3. 화상 전처리 및 처리된 화상의 표시
resize = 224
mean = (0.485, 0.456, 0.406) # ImageNet의 평균값 (ILSVRC 2012 데이터셋의 훈련 데이터로 구해지는 값)
std = (0.229, 0.224, 0.225) # ImageNet의 표준편차 (ILSVRC 2012 데이터셋의 훈련 데이터로 구해지는 값)

transform = BaseTransform(resize, mean, std)
image_transformed = transform(img) # __call__함수 호출

# (색, 높이, 너비) -> (높이, 너비, 색)로 변경
image_transformed = image_transformed.numpy().transpose((1, 2, 0))
image_transformed = np.clip(image_transformed, 0, 1) # 0보다 작은 값은 0으로, 1보다 큰 값은 1로

plt.imshow(image_transformed)
plt.show()


# In[8]:


ILSVRC_class_index = json.load(open("./data/imagenet_class_index.json", "r"))
ILSVRC_class_index


# In[9]:


# 출력 결과에서 라벨을 예측하는 후처리 클래스
class ILSVRCPredictor():
    """
    ILSVRC 데이터 모델의 출력에서 라벨을 구한다.
    
    Attributes
    ----------
    class index : dictionary
        클래스 index와 라벨명을 대응시킨 사전형 변수

    """
    
    def __init__(self, class_index):
        self.class_index = class_index
        
    def predict_max(self, out):
        """
        최대 확률의 ILSVRC 라벨명을 가져온다.
        
        Parameters
        -----------
        out : torch.Size([1, 1000])
            Net에서 출력
            
        Returns
        --------
        predict_label_name : str
            가장 예측 확률이 높은 라벨명
        """
        
        maxid = np.argmax(out.detach().numpy()) # 최대값의 인덱스
        # 출력 값을 network에서 분리하는 것 : out.detach()
        predict_label_name = self.class_index[str(maxid)][1]


# In[10]:


# ILSVRC 에서 라벨정보를 읽어 사전형 변수 생성
ILSVRC_class_index = json.load(open("./data/imagenet_class_index.json", "r"))

# ILSRVRCPredictor 인스턴스 생성
predictor = ILSVRCPredictor(ILSVRC_class_index)

# 입력 Image 읽기
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

resize = 224
mean = (0.485, 0.456, 0.406) # ImageNet의 평균값 (ILSVRC 2012 데이터셋의 훈련 데이터로 구해지는 값)
std = (0.229, 0.224, 0.225) # ImageNet의 표준편차 (ILSVRC 2012 데이터셋의 훈련 데이터로 구해지는 값)

transform = BaseTransform(resize, mean, std)
image_transformed = transform(img) # torch.Size([3, 224, 224])

inputs = image_transformed.unsqueeze_(0) # torch.Size([1, 3, 224, 224])


"""
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval() # 추론 모드 (평가 모드)로 설정
print(net)
"""
# 모델에 입력하고 모델 출력을 라벨로 변환
out = net(inputs) # torch.Size([1. 1000])
result = predictor.predict_max(out)

print("추론 결과: ", result)


# In[ ]:




