from tensorflow import keras
import ywcho.utils as ywutil

# 모델 로드
model = keras.applications.resnet.ResNet50()

# 정보 저장
ywutil.keras_model_info_save(model,__file__)