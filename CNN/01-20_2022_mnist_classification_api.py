"""
This code uses a model from the code '01-19_2022_mnist_classification_sequential'
"""
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

from ywcho.utils import print_shapes

# load data
(train_input, train_target), (test_input, test_target) = keras.datasets.mnist.load_data()
print_shapes(train_input, train_target)

# scale data
train_scaled = train_input.reshape(-1,28,28,1) / 255.0
print_shapes(train_scaled)

# split data
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)
print_shapes(train_scaled,val_scaled,train_target, val_target)

# load model
model = keras.models.load_model('best-cnn-model.h5')
print(model.input)
conv_acti = keras.Model(model.input, model.layers[0].output)
conv_acti2 = keras.Model(model.input, model.layers[2].output)

# draw filter
conv = model.layers[0]
print_shapes(conv.weights[0], conv.weights[1])
conv_weights = conv.weights[0].numpy()

fig, axs = plt.subplots(4,8,figsize=(10,6))
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(conv_weights[:,:,0,i*8+j], vmin=-0.5, vmax=0.5)
        axs[i,j].axis('off')

# draw feature map
sample_start = 1000
sample_count = 1
model = conv_acti2
for sample in range(sample_start,sample_start + sample_count):
    #plt.imshow(train_input[sample], cmap='gray_r')
    # plt.show()

    # test sample
    inputs = train_input[sample:sample+1].reshape(-1,28,28,1) / 255.0
    feature_maps = model.predict(inputs)
    print_shapes(feature_maps)

    # draw feature_maps
    fig, axs = plt.subplots(4, 8, figsize=(10,6))
    for i in range(4):
        for j in range(8):
            axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
            axs[i,j].axis('off')
    plt.show()

