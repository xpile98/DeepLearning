from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. load data sets (train, test)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. split data sets (train -> train, valid)
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


# 3. make model
input_ = keras.layers.Input(shape=X_train.shape[1:])
flatten = keras.layers.Flatten(input_shape=[28,28])(input_)
hidden1 = keras.layers.Dense(300, activation="relu")(flatten)
hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)
bn = keras.layers.BatchNormalization()(flatten)
af = keras.layers.Activation('relu')(bn)
concat = keras.layers.Concatenate()([hidden2, af])      # input - bn - af
output = keras.layers.Dense(10, activation=keras.activations.softmax)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

# 4. compile
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])
# model.compile(optimizer=keras.optimizers.SGD,
#               loss=keras.losses.sparse_categorical_crossentropy,
#               metrics=keras.metrics.Accuracy)

# 5. fit
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=1,
                    validation_data=(X_valid, y_valid))


plt.plot(history.history['loss'])           # model.fit 메서드에서 반환된 history는 history.history로 접근해야한다.
plt.plot(history.history['val_loss'])       # also
plt.xlabel('eopchs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.show()

# 6. evaluate