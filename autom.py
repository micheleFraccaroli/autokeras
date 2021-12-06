import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak
from keras_tuner.engine.hyperparameters import Choice

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)
max_flops = 140000000

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(horizontal_flip=False)(output_node)
output_node = ak.ConvBlock()(output_node)
output_node = autokeras.ImageBlock()(output_node)
output_node = ak.ResNetBlock(version="v2")(output_node)
output_node = ak.ClassificationHead(num_classes=10)(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=15, max_model_size={'flops': float(max_flops)}
)

clf.fit(x_train, y_train, epochs=50)
