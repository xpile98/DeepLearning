# ResNet-34
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same",use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same",use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:             # stride 가 1보다 큰 경우에 특성맵 크기를 맞춰주기 위해 필요함.
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64,7, strides=2, input_shape=[224, 224, 3], padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters=filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

# 케라스가 제공하는 사전훈련된 모델 사용하기
# 샘플 이미지 로드
import numpy as np
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])

# 사전 훈련된 ResNet-50 모델 로드 및 이미지 resize
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")

# 이미지넷 데이터셋에서 사전훈련된 가중치 다운로드
model = keras.applications.resnet50.ResNet50(weights=None)

# ResNet-50 입력에 맞추어 이미지 224 x 224픽셀로 변환
images_resized = tf.image.resize(images, [224,224])

# 전처리 과정 추가
inputs = keras.applications.resnet.preprocess_input(images_resized*255)

#예측 수행
Y_proba = model.predict(inputs)
print(Y_proba.shape)

# 최상위 K개 예측
top_K = keras.applications.resnet.decode_predictions(Y_proba,top=5)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    plot_color_image(images_resized[image_index])
    plt.show()
    for class_id, name, y_proba in top_K[image_index]:
        print(" {} - {:12s} {:.2f}%".format(class_id,name,y_proba * 100))
    print()

