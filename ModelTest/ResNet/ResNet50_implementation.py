from tensorflow import keras




def conv1(layer):
    zpad = keras.layers.ZeroPadding2D(3)(layer)
    conv = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2)(zpad)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)
    zpad = keras.layers.ZeroPadding2D(3)(relu)
    pool = keras.layers.MaxPooling2D(strides=2)(zpad)
    return pool

def conv2_block1(layer):
    conv = keras.layers.Conv2D(kernel_size=(3,3))(layer)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)

    conv = keras.layers.Conv2D(kernel_size=(3,3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)

    conv = keras.layers.Conv2D(filters=256, kernel_size=(3,3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)

    conv = keras.layers.Conv2D(filters=256, kernel_size=(3,3))(layer)
    bnor2 = keras.layers.BatchNormalization()(conv)

    add = keras.layers.Add()(bnor, bnor2)
    out = keras.layers.ReLU()(add)
    return out

def conv2_block2(layer):
    conv = keras.layers.Conv2D(kernel_size=(3, 3))(layer)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)
    conv = keras.layers.Conv2D(kernel_size=(3, 3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)
    conv = keras.layers.Conv2D(filters=256, kernel_size=(3, 3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)

    add = keras.layers.Add()(bnor, layer)
    out = keras.layers.ReLU()(add)
    return out

def conv2_block3(layer):
    conv = keras.layers.Conv2D(kernel_size=(3, 3))(layer)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)
    conv = keras.layers.Conv2D(kernel_size=(3, 3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)
    relu = keras.layers.ReLU()(bnor)
    conv = keras.layers.Conv2D(filters=256, kernel_size=(3, 3))(relu)
    bnor = keras.layers.BatchNormalization()(conv)

    add = keras.layers.Add()(bnor, layer)
    out = keras.layers.ReLU()(add)
    return out

def conv3_block1(layer):





# >> ResNet50 implementation
input_ = keras.layers.Input(shape=(224,224,3))
x = conv1(input_)
x = conv2(x)
output = keras.layers.Dense(1000)(x)

model = keras.Model(inputs=[input_], outputs=[output])
model.summary()

