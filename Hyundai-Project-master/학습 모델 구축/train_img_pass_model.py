#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##alphanumeric order
def sort(lst): 
    lst = [str(i) for i in lst] 
    lst.sort() 
    lst = [int(i) if i.isdigit() else i for i in lst ] 
    return lst 

import glob

## training data에 label 붙여주기
import numpy as np

num_list = []
for i in range(756):
  num_list.append(i)

## wheel_imgs_num dictionary : {카테고리 번호 : 카테고리 속하는 이미지 갯수}
wheel_imgs_num = {}

for idx in num_list:
  wheel_img_path = glob.glob('/home/tjsalszla123/Augmentation/Training/%s/*.jpeg' % idx) ## %s 부분은 '카테고리 번호명인 폴더'를 의미하니, 참고하여 경로를 설정해주시면 됩니다.  
  wheel_imgs_num[idx] = len(wheel_img_path)

## 빈 array를 만들고, alphanumeric order로 레이블 만들기 : wheel_imgs_num.keys()를 sort함수 적용하면 alphanumeric order로 순회합니다.
## 빈 array에 np.concatenate로 numpy 배열 붙여나가는 방식으로 labeling 했습니다.
label_trained = np.array([], dtype ='int32')
for idx in sort(wheel_imgs_num.keys()):
  x = np.array([idx]*wheel_imgs_num[idx], dtype ='int32') ## [카테고리 번호]를 카테고리 속하는 이미지 갯수만큼 배열 만들기
  label_trained = np.concatenate((label_trained,x)) ## 배열 붙이기

del wheel_imgs_num
del wheel_img_path
del x

#pip install image_classifiers==0.2.2

from classification_models.resnet import ResNet50, preprocess_input
from skimage.transform import resize
from PIL import Image
model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)

def result(img,model):
    x = resize(img,(224,224))*255
    x = preprocess_input(x)
    x = np.expand_dims(x,0)
    y = model.predict(x)
    y = np.mean(y, axis=(1,2))
    return y

train_img_array = np.zeros((len(label_trained),2048))
pathes = glob.glob("/home/tjsalszla123/Augmentation/Training/*")
sorted_pathes = sort(pathes)
del pathes
for folder_path in sorted_pathes:
    img_pathes = glob.glob(folder_path+"/*.jpeg")
    for i in range(len(img_pathes)):
        img = Image.open(img_pathes[i])
        img_numpy = np.array(img)
        img_vector = result(img_numpy,model)
        train_img_array[i] = img_vector
        del img
        del img_numpy
        del img_vector
    print(folder_path+"is done.")
    
np.save('/home/tjsalszla123/model/bottleneck_features_train.npy', train_img_array)

