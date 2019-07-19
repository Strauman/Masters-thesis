from tfmodels import TFModel
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Conv2DTranspose, Lambda, InputLayer, concatenate, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from . import util
from tfmodels import USequential, Remember
from sys import exit as xit
import tensorflow as tf

Input=tf.keras.Input

def conv_layer(filters):
    return Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=Activation("relu"))


def bn_layer():
    return tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None
    )


def pooling_layer():
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME', data_format=None)


def tr_conv_layer(filters):
    return Conv2DTranspose(int(filters), kernel_size=(1, 1), strides=(2, 2), padding='SAME')
    # return tf.layers.conv2d_transpose(Z, filters=int(filters), kernel_size=(1, 1), strides=2, padding="SAME", activation=tf.nn.relu)


class EncoderArchitecture(TFModel):
    def __init__(self,config):
        self.copies=[]
        self.input=Input(shape=[None,None], name="Encoder_input")
        self.build(config)

    def build(self, config, input_shape=None):
        self.model=None
        if not isinstance(config, util.Dottify):
            cfg = util.Dottify(config)
        else:
            cfg = config
        depth = cfg.depth
        filters = cfg.filters
        # if input_shape is not None:
            # arch.add(InputLayer(input_shape=input_shape))

        x=self.input
        x=Lambda(lambda Z: tf.expand_dims(Z, 3))(x)
        for _ in range(depth):
            filters *= 2
            x=conv_layer(filters)(x)
            x=conv_layer(filters)(x)
            self.copies.append(x)
            x=bn_layer()(x)
            x=pooling_layer()(x)
        filters *= 2
        x=conv_layer(filters)(x)
        x=conv_layer(filters)(x)
        x=bn_layer()(x)
        # return arch, copies
        self.architecture=x
        self.model=Model(inputs=[self.input], outputs=[self.architecture,*self.copies], name="Encoder")

    def __call__(self, input_tensor, training=None, mask=None):
        return self.model(input_tensor)

        # return self.architecture(input_tensor, training=None, mask=None)
        # if input_tensor not in self.output_models.keys():
        #     input_layer=Input(batch_shape=input_tensor.shape)
        #     self.output_models[input_tensor] = Model(inputs=[input_layer], outputs=[self.architecture(input_layer)])(input_tensor)
        # return self.output_models[input_tensor]


def out_shape(seq):
    # lay=Lambda(lambda x: lay(x))
    # oshape=seq(lay).output_shape
    oshape = seq._layers[-1].output_shape
    # seq.add(lay)
    return oshape

class DecoderArchitecture():
    """docstring for DecoderArchitecture."""
    def __init__(self,Z, config,copy_tensors):
        # self.input_tensor=InputLayer(input_shape=input_z.shape[1:])
        # self.copies=[Input(shape=c.shape[1:]) for c in copy_tensors[::-1]]
        # self.Z=Input(tensor=Zs)
        self.mod=None
        self.config=config
        self.Z=Input(shape=Z.shape[1:], name="Decoder_input")
        self.copies=[Input(shape=c.shape[1:], name="Copy_{}".format(i)) for i,c in enumerate(copy_tensors[::-1])]
        # self.Z=Input(tensor=Z)
        # self.copies=[Input(tensor=c) for c in copy_tensors[::-1]]
        self.build()

    def build(self, input_shape=None):
        config=self.config
        if not isinstance(config, util.Dottify):
            cfg = util.Dottify(config)
        else:
            cfg = config
        depth = cfg.depth
        filters = cfg.filters * (2**(depth + 1))
        x=self.Z
        x=conv_layer(filters)(x)
        for d in range(depth):
            filters /= 2
            # print("filters:", filters)
            # print("zshape:",arch._layers[-1].output_shape)
            x=tr_conv_layer(filters)(x)
            # print("->zshape:",arch._layers[-1].output_shape)
            # print("cshape:",self.copies[d].shape)
            # print(arch(lay).output_shape)
            x=concatenate([x,self.copies[d]], axis = 3)
            x=conv_layer(filters)(x)
            x=conv_layer(filters)(x)
            x=bn_layer()(x)
        x=conv_layer(1)(x)
        x=Activation('sigmoid')(x)
        x=Lambda(lambda Z: tf.squeeze(Z, 3))(x)
        self.architecture=x
        self.model=Model(inputs=[self.Z,*self.copies], outputs=[self.architecture], name="Decoder")
        return x
    def __call__(self,Z,copies):
        # self.copies=copies
        return self.model([Z,*copies[::-1]])

    # def build(self, config, input_shape=None):
    #     if not isinstance(config, util.Dottify):
    #         cfg = util.Dottify(config)
    #     else:
    #         cfg = config
    #     depth = cfg.depth
    #     filters = cfg.filters * (2**(depth + 1))
    #     arch = Sequential([])
    #     if input_shape is not None:
    #         arch.add(InputLayer(input_shape=input_shape))
    #     arch.add(conv_layer(filters))
    #     for d in range(depth):
    #         filters /= 2
    #         # print("filters:", filters)
    #         # print("zshape:",arch._layers[-1].output_shape)
    #         arch.add(tr_conv_layer(filters))
    #         # print("->zshape:",arch._layers[-1].output_shape)
    #         # print("cshape:",self.copies[d].shape)
    #         # print(arch(lay).output_shape)
    #         arch.add(Lambda(lambda y, d=d: tf.concat([y, self.copies[d]], axis=3)))
    #         arch.add(conv_layer(filters))
    #         arch.add(conv_layer(filters))
    #         arch.add(bn_layer())
    #     arch.add(conv_layer(1))
    #     arch.add(Activation('sigmoid'))
    #     arch.add(Lambda(lambda Z: tf.squeeze(Z, 3)))
    #     return arch
    # def __call__(self, input_tensor, copies):
    #     copies=copies[::-1]
    #
    #     # if input_tensor not in self.output_models.keys():
    #         # input_layer=Input(batch_shape=input_tensor.shape)
    #         # self.output_models[input_tensor] = Model(inputs=[input_layer], outputs=[self.architecture(input_layer)])(input_tensor)
    #     input_layer=Input(tensor=input_tensor)
    #     # copies_inputs=[Input(tensor=c) for c in copies]
    #     # return Model(inputs=[input_layer, *[c for c in self.copies]], outputs=[self.architecture(input_layer)])(input_tensor,*copies_inputs)
    #     mod=Model(inputs=[input_layer, *[c for c in self.copies]], outputs=[self.architecture(input_layer)])
    #     return mod([input_tensor,*copies])
        # return self.output_models[input_tensor]

    # def __call__(self,*args,**kwargs):
    #     self.architecture=Sequential(self.modlist)
    #     return super().__call__(*args,**kwargs)

# for d in range(depth):
#     filters *= 2
#     Z = compresslayers(Z, filters, is_training)
#     # copies.append(Z.__copy__())
#     copies.append(Z)
#     Z = down_sample(Z)
#     dprint(Z.get_shape())
