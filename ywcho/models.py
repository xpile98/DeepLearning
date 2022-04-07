from tensorflow import keras

def cnn_sequential(output):
    """
    01.20.2022 convolution neural network model - sequential method
    output : number of output classes
    """

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(output, activation='softmax'))
    return model

