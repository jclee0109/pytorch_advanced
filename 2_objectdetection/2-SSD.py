#!/usr/bin/env python
# coding: utf-8

# In[1]:


# パッケージのimport
import os.path as osp
import random
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2
# pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

get_ipython().run_line_magic('matplotlib', 'inline')

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# In[ ]:


def make_datapath_list(rootpath):
    """
    데이터 경로를 저장한 리스트 작성
    
    
    Parameters
    ----------
    rootpath : str
        데이터 경로
    
    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        데이터 경로를 저장한 리스트   
    """
    
    # 이미지 파일과 어노테이션 파일의 경로 템플릿 작성
    imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    # 훈련 및 검증 파일 ID(파일 이름) 취득
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    # 훈련 데이터의 이미지 파일과 어노테이션 파일의 경로 리스트 작성
    train_img_list = []
    train_anno_list = []
    
    for line in open(train_id_names):
        file_id = line.strip() # 공백과 줄 바꿈 제거
        img_path = (imgpath_template % file_id) # 이미지 경로
        anno_path = (annopath_template % file_id) # 어노테이션 경로
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    # 검증 데이터의 이미지 파일과 어노테이션 파일의 경로 리스트 작성
    val_img_list = []
    val_anno_list = []
    
    for line in open(val_id_names):
        file_id = line.strip() # 공백과 줄 바꿈 제거
        img_path = (imgpath_template % file_id) # 이미지 경로
        anno_path = (annopath_template % file_id) # 어노테이션 경로
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list


# In[ ]:


rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

print(train_img_list[0])


# # 2.2.5 xml 형식의 어노테이션 데이터 리스트를 변환하기

# In[ ]:


# XML 형식의 어노테이션을 리스트 형식으로 변환하는 클래스

class Anno_xml2list(object):
    """
    한 이미지의  XML 형식 어노테잇ㄴ 데이터를 이미지 크기로 규격화하여 리스트 형식으로 변환
    bbox 크기 맞추는 것이 필요한 거야!!
    
    Attributes
    ----------
    
    classes : list
        VOC 데이터셋의 클래스 리스트
    """
    
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height) -> list:
        """
        bbox 의 크기를 이미지의 크기에 맞춰야 하므로, width, height를 받아서 사용
        
        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ...]
            물체의 어노테이션 데이터를 리스트, 이미지에 존재하는 물체 수만큼의 요소를 가진다.
        """
        
        # 이미지 내 모든 물체의 어노테이션을 이 리스트에 저장
        ret = []
        
        # xml 파일 로드
        xml = ET.parse(xml_path).getroot()
        
        # 이미지 내 물체(object) 수만큼 반복
        for obj in xml.iter('objecdt'):
            # 어노테이션에서 탐지가 difficult로 설정된 것은 제외
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
                
            # 한 물체의 어노테이션을 저장하는 리스트
            bndbox = []
            name = obj.find('name').text.lower().strip() # 물체 이름
            bbox = obj.find('bndbox') # 바운딩 박스 정보
            
            # 어노테이션의 xmin, ymin, xmax, ymax를 취득하고 0~1로 규격화
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for pt in (pts):
                # VOC는 원점이 (1, 1)이므로 1을 빼서 (0,0)으로 만들어준다.
                cur_pixel = int(bbox.find(pt).text) - 1
                
                # 폭, 높이로 규격화
                if pt == 'xmin' or pt =='xmax': # x 방향의 경우 폭으로 나눈다.
                    cur_pixel /= width
                else: # y 방향의 경우 높이로 나눈다.
                    cur_pixel /= height
                    
                bndbox.append(cur_pixel)
            
            # 어노테이션 클래스명 index를 취득하여 추가
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            
            # res에 [xmin, ymin, xmax, ymax, label_ind]를 추가
            ret += [bndbox]
            
        return np.array(ret) # [[xmin, ymin, xmax, ymax, label_ind], ...]
        
        
        


# In[ ]:


# 동작 확인
voc_classes = ['aeroaplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)

# 화상 로드용으로 OpenCV 사용
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path) # [높이][너비][색BGR]
height, width, channels = img.shape # 화상 크기 취득

# 어노테이션을 리스트로 표시
transform_anno(val_anno_list[ind], width, height)


# # 2.2.6 화상과 어노테이션의 전처리를 실시하는 DataTransform 클래스 작성
# 학습 시와 추론 시 다르게 작동
# 
# - 이미지 크기가 데이터마다 다를 수 있어서, 이걸 규격화해주는 거야.
# - 이미지 크기를 바꿔주면 bbox의 크기도 바뀌어야하기 때문에 이걸 직접 만들어줘야함 torch에 없음

# In[ ]:


# utils 폴더에 있는 ㅇata_augumentation.py 에서 import
# 입력 영상의 전처리 클래스
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform():
    """
    이미지와 어노테이션의 전처리 클래스. 훈련과 추론에서 다르게 작동한다.
    이미지 크기를 300 x 300 으로 한다.
    학습 시 데이터 확장을 수행한다.
    
    Attributes
    ----------
    input_size : int
        리사이즈 대상 이미지의 크기
    color_mean : (B, G, R)
        각 색상 채널의 평균값
    """
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train' : Compose([
                ConvertFromInts(), # int형을 float형으로 변환
                ToAbsoluteCoords(), # 어노테이션 데이터를 절대 좌표값으로 변환
                PhotometricDistort(), # 색상을 변환
                Expand(color_mean), # 이미지의 캐넙스 확대
                RandomSampleCrop(), # 특정 부분 무작위로 추출
                RandomMirror(), # 좌우 반전
                ToPercentCoords(), # 어노테이션 데이터를 0~1로 정규화
                Resize(input_size), # 입력 크기에 맞춰서 리사이즈
                SubtractMeans(color_mean) # 색상의 평균을 빼서 정규화
            ]),
            'val' : Compose([
                ConvertFromInts(), # int형을 float형으로 변환
                Resize(input_size), # 입력 크기에 맞춰서 리사이즈
                SubtractMeans(color_mean) # 색상의 평균을 빼서 정규화
            ])
        }

def __call__(self, img, phase, boxes, labels):
    return self.data_transform[phase](img, boxes, labels)


# In[ ]:


# 동작 확인

# 1. 이미지 읽기
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path) # [높이][폭][색BGR]
height, width, shape = img.shape

# 2. 어노테이션을 리스트로
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 원본 표시
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 4. 전처리 클래스 작성
color_mean = (104, 117, 123) # (BGR) 색상의 평균값
input_size = 300 # 이미지 input 사이즈를 300x300으로
transform = DataTransform(input_size, color_mean)

# 5. Train 이미지 표시
phase = 'train'
img_transformed, boxes, labels = transform(img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(img_transformed)
plt.show()

# 6. Val 화상 표시
phase = 'val'
img_transformed, boxes, labels = transform(img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(img_transformed)
plt.show()


# In[ ]:


class VOCDataset(data.Dataset):
    """
    VOC2012의 Dataset을 
    
    Attributes
    ----------
    img_list : 리스트
        화상 경로를 저장한 리스트
    anno_list : 리스트
        어노테이션 경로를 저장한 리스트
    phase : 'train' or 'test'
        학습 또는 훈련 설정
    transform : object
        전처리 클래스의 인스턴스
    transform_anno : object
        xml 어노테이션을 리스트로 변환하는 인스턴스
    """
    
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase # train or val
        self.transform = transform # 이미지 변형
        self.transform_anno = transform_anno # 어노테이션 데이터를 xml에서 리스트로 변경
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index: int):
        """
        전처리한 이미지의 텐서 형식 데이터와 어노테이션 취득
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt
    
    def pull_item(self, index):
        """
        전처리한 화상의 텐서 형식 데이터, 어노테이션, 화상의 높이, 폭 취득
        """
        # 1. 이미지 읽기
        image_file_path = self.img_list[index]
        img = cv2.imgread(image_file_path) # [높이][폭][색BGR]
        height, width, channels = img.shape
        
        # 2. xml 형식의 어노테이션 정보를 리스트에 저장
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)
        
        # 3. 전처리 실시
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])
        
        # 색상 채널의 순서가 BGR이므로 RGB로 순서 변경
        # (높이, 폭, 색상 채널)의 순서를 (색상 채널, 높이, 폭)으로 변경
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        
        # BBox 와 라벨을 세트로 한 np.array를 작성. 변수 이름 gt는 ground truth의 약자
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, gt, height, width
        
    


# In[ ]:


# 동작 확인
color_mean = (104, 117, 123) # (BGR) 색의 평균값
input_size = 300 # 이미지 input 사이즈를 300x300으로 한다.

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

# 데이터 출력 예
val_dataset.__getitem__(1)


# 
