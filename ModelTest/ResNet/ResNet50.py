import tensorflow as tf
from tensorflow import keras
import ywcho.utils as ywutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2


# 모델 로드
model = keras.applications.resnet.ResNet50()
# ywutil.keras_model_info_save(model,__file__)

# classification test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scale #####################################
x_train = x_train[:600]                     # (60000, 28, 28)
x_test = x_test[:100]                       # (10000, 28, 28)
y_train = y_train[:600]                     # (60000, 28, 28)
y_test = y_test[:100]                       # (10000, 28, 28)

"""
x_train[:] = cv2.resize(x_train, dsize=(224,224),interpolation=cv2.INTER_LINEAR)

x_train, x_valid = x_train[:48000,:,:], x_train[48000:,:,:]     # (224,224,60000) -> (224,224,48000), (224,224,12000)
y_train, y_valid = y_train[:48000], y_train[48000:]             # (60000) -> (48000), (12000)

x_train = cv2.cvtColor(x_train,cv2.COLOR_GRAY2RGB)
x_valid = cv2.cvtColor(x_valid,cv2.COLOR_GRAY2RGB)
"""


# 용우형 조언
# 1. resize를 tf함수로 수행함으로써 np <> tf 변환에 속도가 오래 걸림
# 2. np.resize 추천
# transpose 4 resize
x_train = np.transpose(x_train,(1,2,0))     # (28, 28, 60000)
x_test = np.transpose(x_test,(1,2,0))       # (28, 28, 10000)

# resize
x_train = tf.image.resize(x_train, [224,224]).numpy()
x_test = tf.image.resize(x_test, [224,224]).numpy()

# re transpose 4 train model
x_train = np.transpose(x_train,(2,0,1))                         # (60000, 28, 28)
x_test = np.transpose(x_test,(2,0,1))                           # (10000, 28, 28)

x_train = np.expand_dims(x_train,axis=-1)
x_train = np.concatenate((x_train,x_train,x_train), axis=-1)
x_test = np.expand_dims(x_test,axis=-1)
x_test = np.concatenate((x_test,x_test,x_test), axis=-1)


x_valid = x_train[480:,:,:,:]
x_train = x_train[:480,:,:,:]
y_valid = y_train[480:]
y_train = y_train[:480]

n_classes = np.unique(y_train).size

# ResNetModel Load
base_model = keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs = output)

# 훈련 초기에 가중치 동결
for layer in base_model.layers:
    layer.trainable=False

# 모델 컴파일 & 훈련
print("Compile Start")
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print("Fit Start")
history = model.fit(x_train, y_train, epochs= 5, validation_data=(x_valid, y_valid), input=None)

# 최상위 층 훈련 후 가중치 동결 해제
for layer in base_model.layers:
    layer.trainable=True

# 모델 컴파일 & 훈련
print("Fit Start 2")
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(x_train,y_train, epochs= 5, validation_data=(x_valid,y_valid))
