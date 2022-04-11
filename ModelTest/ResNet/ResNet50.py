from tensorflow import keras
import ywcho.utils as ywutil
import numpy as np
import matplotlib.pyplot as plt


# 모델 로드
model = keras.applications.resnet.ResNet50()

# 정보 저장
ywutil.keras_model_info_save(model,__file__)

# classification test
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


fig, axs = plt.subplots(4, 8, figsize=(10,5))
for i in range(4):
  for j in range(8):
    axs[i,j].imshow(x_train[0,:,:,i*8+j])
    axs[i,j].axis('off')
plt.show()