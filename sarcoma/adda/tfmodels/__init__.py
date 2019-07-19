from tensorflow.keras.models import Model as _Model,Sequential
from tensorflow.keras.layers import Input,InputLayer
import tensorflow as tf
from sys import exit as xit
from pprint import pprint as pp
import re
from . import layer_utils
from .. import util
# class variable_scope_mgr():
#     def __init__(self, *args, **kwargs):
#     def __enter__(self):
#         return None
#     def __exit__(self, exc_type, exc_value, traceback):
#         return False
import contextlib

@contextlib.contextmanager
def NullContext():
    yield None
import prettytable

def _layer_summary_fields(layer, relevant_nodes):
    connections = []
    try:
        output_shape = layer.output_shape
    except AttributeError:
        output_shape = 'multiple'
    for node in layer._inbound_nodes:
        if relevant_nodes and node not in relevant_nodes:
            # node is not part of the current network
            continue
        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i].name
            inbound_node_index = node.node_indices[i]
            inbound_tensor_index = node.tensor_indices[i]
            # connections.append(inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')
            connections.append(inbound_layer)
    name = layer.name
    cls_name = layer.__class__.__name__
    if not connections:
        first_connection = ''
    else:
        first_connection = connections[0]
    try:
        activation=layer.activation.__name__
    except:
        activation="-"
    # try:
    #     initalizer=layer.kernel.initializer
    # except:
    #     initializer="-"

    fields = [
        name + ' (' + cls_name + ')', output_shape,
        layer.count_params(), first_connection, activation
    ]
    return fields, layer
class UpdModel(_Model):
    """docstring for Model."""
    def __init__(self, *args, keep_scope=True, **kwargs):
        self.did_call=False
        self.keep_scope=keep_scope
        # self.var_scope=tf.get_variable_scope().name
        self.var_scope=tf.get_variable_scope().name
        self.var_scope_re=re.compile(self.var_scope)
        self._did_update_collections=False
        super(UpdModel, self).__init__(*args,**kwargs)
    def summary(self, line_length=None, positions=None, print_fn=None):
        tbl=prettytable.PrettyTable(field_names=["Layer (type)", "Output Shape", "Param #", "Connected To", "Activation"])
        layers = self.layers
        relevant_nodes = []
        num_params=0
        for v in self._nodes_by_depth.values():
            relevant_nodes += v
        for l in layers:
            fields, lay=_layer_summary_fields(l, relevant_nodes)
            num_params+=lay.count_params()
            tbl.add_row(fields)
        tbl.add_row(["-","-",num_params,"-","-"])
        print_fn(tbl.get_string())


    def keras_summary(self, line_length=None, positions=None, print_fn=None, **kwargs):
        return super().summary(line_length=None, positions=None, print_fn=None, **kwargs)

    # def summary(self, line_length=None, positions=None, print_fn=None):
    #     """Prints a string summary of the network.
    #     Arguments:
    #         line_length: Total length of printed lines
    #             (e.g. set this to adapt the display to different
    #             terminal window sizes).
    #         positions: Relative or absolute positions of log elements
    #             in each line. If not provided,
    #             defaults to `[.33, .55, .67, 1.]`.
    #         print_fn: Print function to use. Defaults to `print`.
    #             It will be called on each line of the summary.
    #             You can set it to a custom function
    #             in order to capture the string summary.
    #     Raises:
    #         ValueError: if `summary()` is called before the model is built.
    #     """
    #     if not self.built:
    #       raise ValueError('This model has not yet been built. '
    #                        'Build the model first by calling `build()` or calling '
    #                        '`fit()` with some data, or specify '
    #                        'an `input_shape` argument in the first layer(s) for '
    #                        'automatic build.')
    #     layer_utils.print_summary(self,
    #                               line_length=line_length,
    #                               positions=positions,
    #                               print_fn=print_fn)

    def update_collections(self, inputs):
        if self._did_update_collections: return
        return
        # self._did_update_collections=True
        # print(self.name)
        # print(tf.get_variable_scope().name)
        # print("^^ Scope")

        col_upd=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        col_reg=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        pp(self.get_losses_for(inputs))
        print("^^lossesforinputs")
        for u in self.get_updates_for(inputs):
            if u not in col_upd:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
        for l in self.get_losses_for(inputs):
            if l not in col_reg:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l)

    # def __init__(self, *args,**kwargs):
    #     super(UpdModel, self).__init__(*args,**kwargs)
    #     pp(self.updates)
    #     for u in self.updates:
    #         tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    #     for l in self.losses:
    #         tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l)
    def __call__(self,inputs,*args,**kwargs):
        # print(self.name)
        # print(self.var_scope)
        # if not self.keep_scope:
            # is_right_variable_scope=True
        # else:
            # is_right_variable_scope=True if (self.var_scope_re.match(tf.get_variable_scope().name)) else False
        # with tf.name_scope(self.var_scope) if not is_right_variable_scope else NullContext():
            # print(self.name, "Called", f"vs: self.var_scope is {self.var_scope}, tf.var_scope is {tf.get_variable_scope().name}")
            # print(f"Correct scope? {is_right_variable_scope}")
        rv=super(UpdModel, self).__call__(inputs, *args,**kwargs)
            # print(f"Updates: {self.name}", self.get_updates_for(inputs))
            # print(f"{self.name}:", self.get_losses_for(inputs))
            # self.update_collections(inputs)
        return rv

# UpdModel=_Model

class TFModel(_Model):
    """docstring for TFModel."""
    def __init__(self, *buildargs, do_build=True, **buildkwargs):
        if do_build:
            self.architecture=self.build(*buildargs, **buildkwargs)
        else:
            self.architecture=None
        self.output_models={}
    def build(self,*args, input_shape=None, **kwargs):
        raise NotImplementedError("No model built?")
    def summary(self, config, shape=None):
        if shape is None:
            self.architecture.summary()
        else:
            mod=self.architecture
            arch=self.build(config,input_shape=shape)
            arch.build()
            arch.summary()
            return arch.output_shape[1:]
    def __call__(self, input_tensor):
        # if input_tensor not in self.output_models.keys():
            # input_layer=Input(batch_shape=input_tensor.shape)
            # self.output_models[input_tensor] = Model(inputs=[input_layer], outputs=[self.architecture(input_layer)])(input_tensor)
        input_layer=Input(batch_shape=input_tensor.shape)
        return _Model(inputs=[input_layer], outputs=[self.architecture(input_layer)])(input_tensor)
        # return self.output_models[input_tensor]
from tensorflow.python.util import tf_inspect
from typing import Callable
class LayerList(object):
    """
    https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#review-decorators-without-arguments
    No args can be sent to this decorator
    - __init__ gets myfunc as argument
    - __call__ gets called when myfunc(asdf) is called. Called is passed any arguments sent to myfunc
    """

    def __init__(self, function) -> Callable:
        self.func = function

    def __call__(self, *args, **kwargs) -> Callable:
        layer_list = self.func(*args, **kwargs)

        def _con_layer(input,**layers_kwargs) -> Callable:
            x = input
            for l in layer_list:
                this_layer_kwargs={}
                layer_accepted_args=tf_inspect.getfullargspec(l.call).args
                for k in layers_kwargs.keys():
                    if k in layer_accepted_args:
                        this_layer_kwargs[k]=layers_kwargs[k]
                # print(l,this_layer_kwargs)
                x = l(x,**this_layer_kwargs)
            return x
        return _con_layer


def filter_collection(collection, scope):
    c = []
    regex = re.compile(scope)
    for item in collection:
        if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
    return c
def filter_dense_kernels(collection):
    return filter_collection(collection, ".*/dense[^/]*?/kernel")
def filter_conv_kernels(collection):
    return filter_collection(collection, ".*/conv[^/]*?/kernel")

def filter_kernel_collections(main_collection, *scopes):
    collection=[]
    for scope in scopes:
        collection+=filter_collection(main_collection, f".*/{scope}[^/]*?/kernel")
    return collection


def register_kernel_L2_on_scope(scope, l=0.001, regularizer_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
    parent_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, f".*/?{scope}[^/]*")
    # pp(parent_collection)
    # l2_collection=[]
    l2_collection=filter_kernel_collections(parent_collection, "(?:dense|conv2d)")
    # pp(l2_collection)
    # pp(parent_collection)
    # xit()
    # conv_kernels=filter_conv_kernels(parent_collection)
    # l2_collection+=
    # pp(l2_collection)
    l2_loss=[tf.reduce_sum(item**2) for item in l2_collection]
    l2_loss=l*tf.reduce_sum(l2_loss)
    l2_loss=tf.identity(l2_loss,name=f"l2_regularization_{scope}")
    tf.add_to_collection(regularizer_collection, l2_loss)
    # Allow dense kernels
