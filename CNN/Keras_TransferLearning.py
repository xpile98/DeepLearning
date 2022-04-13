# 전이 학습

# tf데이터셋으로 데이터 적재
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 훈련세트 불러오고 나누기
test_set, valid_set, train_set = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)


builder = tfds.builder('tf_flowers')
dataset_size =builder.info.splits["train"].num_examples     # 3670
class_names = builder.info.features["label"].names          # ["dandelion", "daisy"]
n_classes = builder.info.features["label"].num_classes      # 5

# 데이터 전처리
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224,224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# 훈련 세트 섞고 전처리 적용
batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# Xception Model Load
# include_top=False : 네트워크 최상층 전역평균풀링층, 밀집출력층 제외
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# 훈련 초기에 가중치 동결
for layer in base_model.layers:
    layer.trainable=False

# 모델 컴파일 & 훈련
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs= 5, validation_data=valid_set)

# 최상위 층 훈련 후 가중치 동결 해제
for layer in base_model.layers:
    layer.trainable=True

# 모델 컴파일 & 훈련
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs= 5, validation_data=valid_set)

