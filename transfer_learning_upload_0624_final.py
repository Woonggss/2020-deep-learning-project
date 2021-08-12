#!/usr/bin/env python
# coding: utf-8

# # 1. training data와 validation data에 label 붙여주기
# 학습을 시킬 때, ImageGenerator 옵션에서 'shuffle = False'인 경우 지정 경로에서 alphanumeric order로 데이터가 입력됩니다. 그래서 먼저 alphanumeric으로 숫자를 정렬해주고, label을 붙여줍니다.

# In[1]:


##alphanumeric order
def sort(lst): 
    lst = [str(i) for i in lst] 
    lst.sort() 
    lst = [int(i) if i.isdigit() else i for i in lst ] 
    return lst 


# In[3]:


## training data에 label 붙여주기
import numpy as np
import glob

num_list = []
for i in range(91):
  num_list.append(i)

## wheel_imgs_num dictionary : {카테고리 번호 : 카테고리 속하는 이미지 갯수}
wheel_imgs_num = {}

for idx in num_list:
  wheel_img_path = glob.glob('/home/qwe3142/프로젝트/Data Augmentation/%s/*.jpeg' % idx) ## %s 부분은 '카테고리 번호명인 폴더'를 의미하니, 참고하여 경로를 설정해주시면 됩니다.  
  wheel_imgs_num[idx] = len(wheel_img_path)

## 빈 array를 만들고, alphanumeric order로 레이블 만들기 : wheel_imgs_num.keys()를 sort함수 적용하면 alphanumeric order로 순회합니다.
## 빈 array에 np.concatenate로 numpy 배열 붙여나가는 방식으로 labeling 했습니다.
label_trained = np.array([], dtype ='int32')
for idx in sort(wheel_imgs_num.keys()):
  x = np.array([idx]*wheel_imgs_num[idx], dtype ='int32') ## [카테고리 번호]를 카테고리 속하는 이미지 갯수만큼 배열 만들기
  label_trained = np.concatenate((label_trained,x)) ## 배열 붙이기
label_trained

np.save('경로/label_trained.npy', label_trained)

nb_train_samples = len(label_trained) ## 총 training data의 갯수는 최종 만들어진 training label의 갯수와 같습니다.


# In[4]:


## validation data에 label 붙여주기

## wheel_imgs_num dictionary : {카테고리 번호 : 카테고리 속하는 이미지 갯수}

wheel_imgs_num = {}

## 위에서 만든 num_list
for idx in num_list:
  wheel_img_path = glob.glob('/home/qwe3142/프로젝트/augmented_validation/%s/*.jpeg' % idx) ## %s 부분은 '카테고리 번호 명'인 폴더를 의미하니, 참고하여 경로를 설정해주시면 됩니다. 
  wheel_imgs_num[idx] = len(wheel_img_path)

## 빈 리스트를 만들고, alphanumeric order로 레이블 만들기 : wheel_imgs_num.keys()를 sort함수 적용하면 alphanumeric order로 순회합니다.
## 빈 array에 np.concatenate로 numpy 배열 붙여나가는 방식으로 labeling 했습니다.
label_validate = np.array([], dtype ='int32')
for idx in sort(wheel_imgs_num.keys()):
  x = np.array([idx]*wheel_imgs_num[idx], dtype ='int32') ## [카테고리 번호]를 카테고리 속하는 이미지 갯수만큼 배열 만들기
  label_validate = np.concatenate((label_validate,x)) ## 배열 붙이기
label_validate

np.save('경로/label_validate.npy', label_validate)


nb_validation_samples = len(label_validate) ## 총 validation data의 갯수는 최종 만들어진 validate label의 갯수와 같습니다.


# #2. Transfer Learning 1 : ResNet50 계층 그대로 가져다 쓰기
# 전이학습에는 계층을 그대로 가져다 쓰는 방법과, 계층 중에서 일부를 더 학습시키는 방법(fine-tuning)이 있습니다. 첫 번째 방법은 분류기를 제외한 사전 학습된 모델을 가져온 다음 그 위에 사용자에게 맞게 fully-connected 계층을 얹어서 모델을 구축합니다. 기존 예제코드에서 돌려볼 수 있는 부분은 돌려보면서 바로 사용할 수 있게끔 만들었습니다.
# 
# 참고 : https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/ <- 케라스 공식 블로그, 작은 데이터셋으로 강력한 이미지 분류 모델 설계하기
# 
# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069 <- 참고한 원본 코드
# 
# https://keras.io/ko/getting-started/functional-api-guide/ <- 케라스 api 한국어 버전

# In[ ]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import EarlyStopping, LambdaCallback
 


# dimensions of our images.
img_width, img_height = 224, 224 ##프로젝트 활용 이미지의 크기

top_model_weights_path = '경로/model_features/bottleneck_fc_model.h5' ## 뒤의 bottleneck_fc_model.h5는 지우시면 안됩니다.
train_data_dir = '경로' #~~/training data/182(카테고리 숫자)/*.jpg 경로가 이렇게 생겼으면 traning data 까지만 복사해서 넣어주시면 됩니다.(맨 뒤에 '/'붙일 필요X)
validation_data_dir = '경로'  #~~/validation data/182(카테고리 숫자)/*.jpg 경로가 이렇게 생겼으면 validation data 까지만 복사해서 넣어주시면 됩니다.(맨 뒤에 '/'붙일 필요X)
nb_train_samples = len(label_trained) ## 총 training data의 갯수는 최종 만들어진 training label의 갯수와 같습니다.
nb_validation_samples = len(label_validate) ## 총 validation data의 갯수는 최종 만들어진 validation label의 갯수와 같습니다.
epochs = 50 ## 학습 에폭 수
batch_size = 128 ## 학습 당 batch_size

## bottleneck_features : classifier 직전의 vector를 의미합니다. (ResNet 50을 통과시킨) feature vector와 같은 의미입니다.
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the ResNet50 Network , imagenet의 가중치를 그대로 가져오고, include_top = False로 두어 classifier 부분은 부르지 않습니다.
    model = applications.ResNet50(include_top=False, weights='imagenet')
    # Shuffle을 false로 두어야 카테고리가 뒤섞이지 않고 순서대로 들어갑니다. 케라스 API에 따르면, alphanumeric 순서를 따른다고 합니다.
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator)
    bottleneck_features_train = np.mean(bottleneck_features_train, axis=(1,2))
    # np.save / np.load : 경로 지정한 곳으로 저장, 불러오기 가능, .npy확장자로 save하면 뽑힌 feature vector들이 저장됩니다.
    np.save('경로/model_features/bottleneck_features_train.npy', 
            bottleneck_features_train) ## 뒤의 bottleneck_features_train.npy는 지우시면 안됩니다.

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator)
    bottleneck_features_validation = np.mean(bottleneck_features_validation, axis=(1,2))
    np.save('경로/model_features/bottleneck_features_validation.npy',
            bottleneck_features_validation) ## 뒤의 bottleneck_features_validation.npy는 지우시면 안됩니다.

    # 위에서 저장한 feature vector들을 불러와서, 순서대로 label을 붙여주고 우리 목적에 맞는 classfier를 만들어서 학습시킵니다. 이 때 label은 위에서 만든 label을 활용합니다.
def train_top_model():
    train_data = np.load('경로/model_features/bottleneck_features_train.npy') ##bottleneck_features_train.npy 저장 설정했던 경로
    train_labels = label_trained

    validation_data = np.load('경로/model_features/bottleneck_features_validation.npy') ##bottleneck_features_validation.npy 저장 설정했던 경로
    validation_labels = label_validate
    
    # 새로 쌓은 모델의 계층. 예제 코드에서 가져왔습니다.

    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(91, activation='softmax')) #dense는 분류기 갯수만큼

    # 어떻게 학습시킬지 complie 메소드로 정해줄 수 있습니다.

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # print_weights와 early_stopping은 학습 과정에 추가할 수 있는 옵션입니다.
    # early stopping은 학습에 진전이 없을 경우 학습을 종료하는 옵션이며, print_weights는(LambdaCallback) 학습이 잘 되고 있는지 특정 계층을 모니터링 할 수 있는 옵션입니다.
    # 여기서는 2번째 layer를 기준으로 가중치가 업데이트 되는지 확인할 수 있습니다.
    print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[1].get_weights()))
    early_stopping = EarlyStopping(patience=3, mode='auto', monitor='val_loss')
    
    # fit 함수로 학습을 시작합니다.
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[early_stopping, print_weights])
    
    #학습이 끝난 뒤, 모델을 저장합니다. 지정해 준 경로에 저장됩니다.
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()


# #3. Transfer Learning 2 : ResNet50 계층 일부분을 추가로 학습시켜서 활용하기(Fine-tuning)
# 기존에 학습되어있는 fully-connected 계층을 가져와서(위에서 학습시킨 모델 가중치) ResNet50 위에 얹습니다. 이후 ResNet50 계층 구조를 참고하여 학습시키지 않을 부분은 동결시키고(non-trainable) 학습시킬 부분만 살려서 학습을 진행합니다.(중간에 non-trainable 계층 갯수는 생각해봐야 할 부분) 참고 페이지에서도 말하듯 fine tuning의 핵심은 미세 조정이기 때문에 learning rate를 낮게 설정하고, 또한 사례에서는 모델의 마지막 계층만을 추가로 학습시켰습니다.
# 
# 참고 : https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/ <- 케라스 공식 블로그, 작은 데이터셋으로 강력한 이미지 분류(Transfer Learning 1과 동일)
# 
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975 <- 원본 코드
# 
# https://eremo2002.tistory.com/76 <- ResNet50 구조

# #### 6월 23일 Trnasfer_Learning 2 수정
# 
# 1. new_model 추가 : 기존의 model(ResNet50)에다 바로 top_train_model을 얹으려니 에러가 발생했습니다. model은 Sequential()이 있는 형태에서만 add가 가능하다고 하여, new_model을 만들고 거기에 ResNet50 모델과 top_train_model을 이어서 얹는 방식으로 구성했습니다.
# 
# 2. load_weights 에러 해결 : keras에 있는 버그라고 하는데, top_model의 첫 Dense 계층에 input_dim을 설정해줌으로써 해결했습니다.
# 
# 3. GlobalAveragePooling2D : 기존 transfer learning 1에서 np.mean으로 average pooling을 해줬으니, 여기서도 pooling 계층을 추가해주어야 합니다.
# 
# 4. model.complie : ImageDataGenerator에서 class_mode=categorical로 주면, categorical_crossentropy로 loss함수를 지정해줘야 합니다.
# 
# 5. new_model : new_model_layers[0]까지 들어가야 ResNet을 열어볼 수 있어서 반영해주었습니다.

# In[ ]:





# In[12]:


import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, LambdaCallback


label_trained = np.load('/home/qwe3142/프로젝트2/model_features/label_trained.npy')
label_validate = np.load('/home/qwe3142/프로젝트2/model_features/label_validate.npy')

# path to the model weights files.
## top_model_weights : 얹기 전에 미리 학습을 시켜야 하는데, 위에서 학습한 모델 가중치 불러와서 활용하면 됩니다.
top_model_weights_path = '/home/qwe3142/프로젝트2/model_features/bottleneck_fc_model.h5' ## 먼저 학습이 된 top model의 weight를 불러와야 합니다. 위의 top_model_weights_path를 가져오시면 됩니다.
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '/home/qwe3142/프로젝트2/Data Augmentation' # ~~/training data/182(카테고리 숫자)/*.jpg 경로가 이렇게 생겼으면 traning data 까지만 복사해서 넣어주시면 됩니다.(뒤에 '/'붙일 필요 없음)
validation_data_dir = '/home/qwe3142/프로젝트2/augmented_validation' # ~~/validation data/182(카테고리 숫자)/*.jpg 경로가 이렇게 생겼으면 validation data 까지만 복사해서 넣어주시면 됩니다.(뒤에 '/'붙일 필요 없음)
nb_train_samples = len(label_trained) ## 총 training data의 갯수는 최종 만들어진 training label의 갯수와 같습니다.
nb_validation_samples = len(label_validate) ## 총 validation data의 갯수는 최종 만들어진 validation label의 갯수와 같습니다.
epochs = 50
batch_size = 128

# build the ResNet50 network(사용자에 맞게)
model = applications.ResNet50(weights='imagenet', include_top=False)

print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu', input_dim = 2048))
top_model.add(Dropout(0.5))
top_model.add(Dense(91, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base

new_model = Sequential()
new_model.add(model)
new_model.add(GlobalAveragePooling2D())
new_model.add(top_model)


## ResNet50 기준으로 conv5(마지막 convolution layer)만 살리고 나머지는 동결, conv5는 layer 기준 143번부터 시작하므로, 그 전까지(0~142)의 계층은 모두 얼려줍니다.
for layer in new_model.layers[0].layers[:143]:
    layer.trainable=False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
new_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Generator로 데이터 준비합니다. 이 때는 상위 폴더의 이름(카테고리 명)이 바로 label로 붙도록, class_mode를 categorical로 지정해줍니다.
train_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# print_weights와 early_stopping은 학습 과정에 추가할 수 있는 옵션입니다.
# early stopping은 학습에 진전이 없을 경우 학습을 종료하는 옵션이며, print_weights는(LambdaCallback) 학습이 잘 되고 있는지 특정 계층을 모니터링 할 수 있는 옵션입니다.
# 여기서는 171번째 layer를 기준으로 가중치가 업데이트 되는지 확인할 수 있습니다.(convolution 2D 계층, ResNet50의 마지막 Convolution 계층) 
print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(new_model.layers[0].layers[171].get_weights()))
early_stopping = EarlyStopping(patience=3, mode='auto', monitor='val_loss')


# fine-tune the model, 여기서는 generator로 데이터가 들어오므로 fit_generator를 활용해줍니다.
new_model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, print_weights])

# 학습이 끝난 뒤 모델을 저장합니다. 위와 마찬가지로 경로만 바꿔주고 뒤에 h5 파일은 그대로 두셔야 합니다.
new_model.save_weights('/home/qwe3142/프로젝트2/model_features/fine_tuned_model.h5')


# In[ ]:




