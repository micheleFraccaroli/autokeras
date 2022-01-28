import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import autokeras as ak
from keras_tuner.engine.hyperparameters import Choice
from trains import Task

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
max_flops = 77479996

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageBlock()(output_node)
output_node = ak.ClassificationHead(num_classes=10)(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, tuner="bayesian", overwrite=True, max_trials=5, max_model_size={'flops': float(max_flops)}
)

clf.fit(x_train[:1000], y_train[:1000], epochs=3)
