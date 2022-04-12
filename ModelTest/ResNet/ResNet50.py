from tensorflow import keras
import ywcho.utils as ywutil
import numpy as np
import matplotlib.pyplot as plt


# 모델 로드
model = keras.applications.resnet.ResNet50()
ywutil.keras_model_info_save(model,__file__)

# classification test

