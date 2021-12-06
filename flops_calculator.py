from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.keras.applications.vgg16 import VGG16
import node_manager as nm
import sys


class flop_calculator:
    def flop_calc(self, concrete_func):
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')

            run_meta = tf.compat.v1.RunMetadata()
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder().with_empty_output().float_operation())
            opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

            return flops

    def get_flops(self, model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
        flops = self.flop_calc(concrete_func)
        res_dict = nm.to_dict(flops)
        print(" ---- REPORT CUSTOM ------\n")
        print('TOTAL_FLOPS: {}\n'.format(flops.total_float_ops))
        print(res_dict)
        return flops.total_float_ops


if __name__ == '__main__':
    initial_model = tf.keras.applications.VGG16()

    model = keras.Model(
        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],
    )

    fc = flop_calculator()
    flops = fc.get_flops(model)


