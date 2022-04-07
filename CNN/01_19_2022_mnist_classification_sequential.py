from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import ywcho.models as yw

""" load mnist datasets """
# load datasets #####################################
(train_input, train_target), (test_input, test_target) = keras.datasets.mnist.load_data()

# check shape #####################################
# print(train_input.shape, train_target.shape)    # (60000, 28, 28) (60000,)

# show 10 input #####################################
# fig, axs = plt.subplots(1,10, figsize = (10,1))
# for i in range(10):
#     axs[i].imshow(train_input[i],cmap='gray_r')
#     axs[i].axis('off')
# plt.show()

# print 10 target   #####################################
# print([train_target[i] for i in range(10)])     # [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]

# 레이블 당 샘플 개수 확인    #####################################
# print(np.unique(train_target, return_counts=True))

""" datasets preprocessing """
# scale #####################################
train_scaled = train_input.reshape(-1,28,28,1) / 255.0

# train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)
print(train_scaled.shape, train_target.shape)
""" Convolution Neural Network """
# make model    #####################################
model = yw.cnn_sequential(output=10)
# model = keras.Sequential()
# model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu',input_shape=(28,28,1)))
# model.add(keras.layers.MaxPooling2D(2))
# model.add(keras.layers.Conv2D(64,kernel_size=(3,3),padding='same', activation='relu'))
# model.add(keras.layers.MaxPooling2D(2))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(100,activation='relu'))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Dense(10, activation='softmax'))

# summary   #####################################
model.summary()

# plot_model    #####################################
# keras.utils.plot_model(model)
keras.utils.plot_model(model,show_shapes=True,to_file='cnn-architecture.png',dpi=100)

""" 모델 컴파일과 훈련 """
# compile   #####################################
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

# train & save model, history   #####################################
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb,early_stopping_cb])
np.save('my_history.npy',history.history)

# load model and history    #####################################
# model = keras.models.load_model('best-cnn-model.h5')
# history=np.load('my_history.npy',allow_pickle='TRUE').item()

# loss graph    #####################################
# plt.plot(history['loss'])           # model.fit 메서드에서 반환된 history는 history.history로 접근해야한다.
# plt.plot(history['val_loss'])       # also
# plt.xlabel('eopchs')
# plt.ylabel('loss')
# plt.legend(['train', 'validation'])
# plt.show()

# evaluate  #####################################
model.evaluate(val_scaled, val_target)
plt.imshow(val_scaled[0].reshape(28,28),cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(0,10),preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = range(0,10)
print(classes[np.argmax(preds)])

""" final test """
test_scaled = test_input.reshape(-1,28,28,1) / 255.0
model.evaluate(test_scaled, test_target)
