import tensorflow as tf
from tensorflow import keras
import time

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# >> scale
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]                 # y는 정답 데이터기에 스케일 조정이 필요 없다.
X_test = X_test / 255.0

# >> make class-name list
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# >> make sequential model
# 1)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 2)
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation='relu'),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# >> summary
# model.summary()

# >> compile
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# >> train and validation
start = time.time()  # 시작 시간 저장
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=64)
print("time : ", time.time() - start)

# >> learning curve
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
# plt.show()

# predict
import numpy as np
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
print(y_proba)

y_pred = model.predict_classes(X_new)
print(np.array(class_names)[y_pred])        # ['Ankle boot' 'Pullover' 'Trouser']