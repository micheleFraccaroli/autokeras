# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from datetime import datetime
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow.python.util import nest
# from autokeras.utils.history import History
from tensorflow.keras.callbacks import History
from tqdm import tqdm
from flops_calculator import flop_calculator
import flops_losses
from colors import colors


def validate_num_inputs(inputs, num):
    inputs = nest.flatten(inputs)
    if not len(inputs) == num:
        raise ValueError(
            "Expected {num} elements in the inputs list "
            "but received {len} inputs.".format(num=num, len=len(inputs))
        )


def to_snake_case(name):
    intermediate = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    insecure = re.sub("([a-z])([A-Z])", r"\1_\2", intermediate).lower()
    return insecure


def check_tf_version() -> None:
    if parse(tf.__version__) < parse("2.3.0"):
        raise ImportError(
            "The Tensorflow package version needs to be at least 2.3.0 \n"
            "for AutoKeras to run. Currently, your TensorFlow version is \n"
            "{version}. Please upgrade with \n"
            "`$ pip install --upgrade tensorflow`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.".format(version=tf.__version__)
        )


def check_kt_version() -> None:
    if parse(keras_tuner.__version__) < parse("1.0.3"):
        raise ImportError(
            "The Keras Tuner package version needs to be at least 1.0.3 \n"
            "for AutoKeras to run. Currently, your Keras Tuner version is \n"
            "{version}. Please upgrade with \n"
            "`$ pip install --upgrade keras-tuner`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.".format(version=keras_tuner.__version__)
        )


def contain_instance(instance_list, instance_type):
    return any([isinstance(instance, instance_type) for instance in instance_list])


def evaluate_with_adaptive_batch_size(model, batch_size, verbose=1, **fit_kwargs):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.evaluate(
            x, verbose=verbose, **kwargs
        ),
        **fit_kwargs
    )


def predict_with_adaptive_batch_size(model, batch_size, verbose=1, **fit_kwargs):
    return run_with_adaptive_batch_size(
        batch_size,
        lambda x, validation_data, **kwargs: model.predict(
            x, verbose=verbose, **kwargs
        ),
        **fit_kwargs
    )


def fit_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    history = run_with_adaptive_batch_size(
        batch_size, lambda **kwargs: model.fit(**kwargs), **fit_kwargs
    )
    return model, history


def run_with_adaptive_batch_size(batch_size, func, **fit_kwargs):
    x = fit_kwargs.pop("x")
    validation_data = None
    if "validation_data" in fit_kwargs:
        validation_data = fit_kwargs.pop("validation_data")
    while batch_size > 0:
        try:
            history = func(x=x, validation_data=validation_data, **fit_kwargs)
            break
        except tf.errors.ResourceExhaustedError as e:
            if batch_size == 1:
                raise e
            batch_size //= 2
            print(
                "Not enough memory, reduce batch size to {batch_size}.".format(
                    batch_size=batch_size
                )
            )
            x = x.unbatch().batch(batch_size)
            if validation_data is not None:
                validation_data = validation_data.unbatch().batch(batch_size)
    return history


'''
Custom training loop here###############################################################################################
'''


def custom_training_loop(model, batch_size, max_flops, **fit_kwargs):
    @tf.function
    def _train_step(x, y, model, loss_fn, optimizer, train_epoch_accuracy, train_epoch_loss_avg, max_flops, actual_flops):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
            loss_value += abs(max_flops - actual_flops)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_epoch_accuracy.update_state(y, logits)
        train_epoch_loss_avg.update_state(loss_value)
        return loss_value

    @tf.function
    def _validation_step(x, y, loss_fn, model, val_epoch_accuracy, val_epoch_loss_avg):
        val_logits = model(x, training=False)
        loss = loss_fn(y, val_logits)
        val_epoch_accuracy.update_state(y, val_logits)
        val_epoch_loss_avg.update_state(loss)

    try:
        folder = str(datetime.now().strftime("%b-%d-%Y"))
        os.mkdir("../logs/{}".format(folder))
    except OSError as e:
        print(e)
    history = History()
    history.model = model
    print(model.summary())
    logs = {'loss': None, 'accuracy': None, 'val_loss': None, 'val_accuracy': None}
    writer = tf.summary.create_file_writer("logs/{}".format(folder))
    fc = flop_calculator()
    actual_flops = fc.get_flops(history.model)
    loss_fn = model.loss['classification_head_1']

    optimizer = model.optimizer
    history.on_train_begin()

    pbar = tqdm(range(fit_kwargs['epochs']))
    for epoch in pbar:
        train_epoch_loss_avg = tf.keras.metrics.Mean()
        train_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_epoch_loss_avg = tf.keras.metrics.Mean()
        val_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        model.metrics.extend((train_epoch_loss_avg, train_epoch_accuracy))
        model.metrics_names.extend(('loss', 'accuracy'))
        history.model.metrics.extend((train_epoch_loss_avg, train_epoch_accuracy))
        history.model.metrics_names.extend(('loss', 'accuracy'))
        # Training loop -
        for x, y in fit_kwargs['x']:
            pbar.set_description("TRAINING")
            loss_value = _train_step(x, y, model, loss_fn, optimizer, train_epoch_accuracy, train_epoch_loss_avg, max_flops, actual_flops)

        # Display metrics at the end of each epoch.
        train_acc = train_epoch_accuracy.result()
        train_loss = train_epoch_loss_avg.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))
        # print("Training loss over epoch: %.4f" % (float(train_loss),))
        # pbar.set_postfix({'Training acc': float(train_acc), 'Training loss': float(train_loss)})

        # Reset training metrics at the end of each epoch
        train_epoch_accuracy.reset_states()
        train_epoch_loss_avg.reset_states()

        # Run a validation loop at the end of each epoch.
        for xv, yv in fit_kwargs['validation_data']:
            # pbar.set_description("VALIDATION")
            _validation_step(xv, yv, loss_fn, model, val_epoch_accuracy, val_epoch_loss_avg)

        # Reset training metrics at the end of each epoch
        train_epoch_accuracy.reset_states()
        train_epoch_loss_avg.reset_states()

        val_acc = val_epoch_accuracy.result()
        val_loss = val_epoch_loss_avg.result()
        val_epoch_accuracy.reset_states()
        val_epoch_loss_avg.reset_states()
        # print("Validation acc: %.4f" % (float(val_acc),))
        # print("Validation loss: %.4f" % (float(val_loss),))
        pbar.set_postfix({'Training acc': float(train_acc), 'Training loss': float(train_loss), 'Valid acc': float(val_acc), 'Valid loss': float(val_loss)})

        with writer.as_default():
            tf.summary.scalar("Train Loss", train_epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar("Train Acc", train_epoch_accuracy.result(), step=epoch)
            tf.summary.scalar("Flops", actual_flops, step=epoch, description="Flops of the model")
            tf.summary.scalar("|max_flops - actual_flops|", abs(max_flops - actual_flops), step=epoch)

        with writer.as_default():
            tf.summary.scalar("Validation Loss", val_epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar("Val Acc", val_epoch_accuracy.result(), step=epoch)

        writer.flush()

        # if epoch % 10 == 0:
        #     model.save('tf_ckpts/model_{}@{}epoch'.format(model.name, epoch))
        #     print("Saved checkpoint for step    {} in {}".format(epoch, 'tf_ckpts'))

        logs['loss'] = train_loss.numpy()
        logs['accuracy'] = train_acc.numpy()
        logs['val_loss'] = val_loss.numpy()
        logs['val_accuracy'] = val_acc.numpy()
        history.on_epoch_end(epoch, logs=logs)
        # stopEarly = Callback_EarlyStopping(val_loss_results, min_delta=0.5, patience=20)
        # if stopEarly:
        #     print("Callback_EarlyStopping signal received at epoch= %d/%d" % (epoch, num_epochs))
        #     print("Terminating training ")
        #     model.save('tf_ckpts/model_{}@{}epoch'.format(model_name, epoch))
        #     print("Saved checkpoint for step {} in {}".format(epoch, 'tf_ckpts'))
        #     break
    writer.close()
    return model, history


'''
########################################################################################################################
'''


def get_hyperparameter(value, hp, dtype):
    if value is None:
        return hp
    return value


def add_to_hp(hp, hps, name=None):
    """Add the HyperParameter (self) to the HyperParameters.

    # Arguments
        hp: keras_tuner.HyperParameters.
        name: String. If left unspecified, the hp name is used.
    """
    if not isinstance(hp, keras_tuner.engine.hyperparameters.HyperParameter):
        return hp
    kwargs = hp.get_config()
    if name is None:
        name = hp.name
    kwargs.pop("conditions")
    kwargs.pop("name")
    class_name = hp.__class__.__name__
    func = getattr(hps, class_name)
    return func(name=name, **kwargs)
