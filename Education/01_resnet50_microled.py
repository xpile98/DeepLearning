import timeit

import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# --- Model Load ---
base_model = ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(310,350,3),
    pooling=None)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.2)(x)
x = layers.Dense(9, activation='softmax')(x)
model = models.Model(base_model.input, x, name='Keras_ResNet50')

base_model.summary()
model.summary()
model.trainable = True

model.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])
keras.utils.plot_model(model, show_shapes=True, to_file='Keras_ResNet50.png')


# --- Image Load ---
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = 'D:\AI\Study\DeepLearning\Education\Dataset_microLED'
train_dir = os.path.join(base_dir,'train')
valid_dir = os.path.join(base_dir,'valid')
test_dir = os.path.join(base_dir,'test')

train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (310,350),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse_categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size = (310,350),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse_categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (310,350),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse_categorical'
)

# --- call backs ---
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
cb_list = [
    ModelCheckpoint('checkpoint'),
    EarlyStopping(monitor='val_loss', patience=3, mode='auto')
]

# --- train ---
start = timeit.default_timer()
history = model.fit_generator(train_generator, steps_per_epoch= 2700/32, epochs = 100,
                              validation_data=valid_generator, callbacks=cb_list)