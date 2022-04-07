from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)     # data => [train / test]
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)               # data => [(train / valid) / test]

# >> scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 출력층은 활성화 함수가 없는 하나의 뉴런 (하나의 값 예측 필요)
# 손실 함수로 mean_squared_error을 사용
# 데이터셋에 잡음이 많기에 과대적합 되지 않게 주의 필요!

input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
bn = keras.layers.BatchNormalization()(input_)
af = keras.layers.Activation('relu')(bn)
concat = keras.layers.Concatenate()([hidden2, af])      # input - bn - af
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])


# >> model compile and fit
model.compile(loss='mean_squared_error', optimizer='sgd')

# >> callbacks
class PrintValTrainRatioCallback(keras.callbacks.Callback):     # 사용자 콜백 함수
    def on_epoch_end(self, epoch, logs=None):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs['loss']))

checkpoint_cb = keras.callbacks.ModelCheckpoint('Keras_WideDeep1.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
user_cb = PrintValTrainRatioCallback()

history = model.fit(
    X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb, user_cb])
model = keras.models.load_model('Keras_WideDeep1.h5')   # 최상의 모델로 복원

mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)

# >> learning curve
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

