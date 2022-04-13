import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 모델 로드
model = keras.applications.resnet.ResNet50()
# ywutil.keras_model_info_save(model,__file__)

# >> load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# (60000, 28, 28)   # (60000,)  # (10000, 28, 28)   # (10000,)
x_train = x_train[:6000]
y_train = y_train[:6000]
x_test = x_test[:1000]
y_test = y_train[:1000]

# >> preprocessing
# 1) resize
x_train = tf.image.resize(x_train[..., tf.newaxis], (244, 244))
x_test = tf.image.resize(x_test[..., tf.newaxis], (244, 244))
# 2) 1ch -> 3ch
x_train = np.concatenate((x_train, x_train, x_train),axis=-1)
x_test = np.concatenate((x_test, x_test, x_test),axis=-1)
# 3) split
x_valid = x_train[4800:]
x_train = x_train[:4800]
y_valid = y_train[4800:]
y_train = y_train[:4800]
# call backs
class PrintValTrainRatioCallback(keras.callbacks.Callback):     # 사용자 콜백 함수
    def on_epoch_end(self, epoch, logs=None):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs['loss']))

checkpoint_cb = keras.callbacks.ModelCheckpoint('ResNet50_TransferLearning.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
user_cb = PrintValTrainRatioCallback()



# >> Trasfer Learning
# 기존 모델 인스턴스화
base_model = keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)
# 기본 모델 동결
base_model.trainable = False
# 새 모델 쌓기
n_classes = np.unique(y_test).size
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)
# 새 데이터에 대한 모델 훈련
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs= 10, validation_data=(x_valid, y_valid))

# >> Fine tuning
# 최상위 층 훈련 후 가중치 동결 해제
base_model.trainable=True
# 모델 컴파일 & 훈련
print("Fit Start 2")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(1e-5),
              metrics=["accuracy"])
history = model.fit(x_train,y_train,
                    epochs= 30, batch_size=32,
                    validation_data=(x_valid,y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb, user_cb])

# >> save history
hist_df = pd.DataFrame(history.history)
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


# >> learning curve
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])           # model.fit 메서드에서 반환된 history는 history.history로 접근해야한다.
plt.plot(history.history['val_accuracy'])       # also
plt.xlabel('eopchs')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
plt.subplot(1,2,2)
plt.plot(history.history['loss'])           # model.fit 메서드에서 반환된 history는 history.history로 접근해야한다.
plt.plot(history.history['val_loss'])       # also
plt.xlabel('eopchs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.show()
