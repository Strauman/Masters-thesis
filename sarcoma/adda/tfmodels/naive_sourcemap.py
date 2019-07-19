from tfmodels import TFModel
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Conv2DTranspose,Lambda,InputLayer
from tensorflow.keras.models import Sequential
from . import util
import tensorflow as tf
def conv_layer(filters):
    return Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=Activation("relu"))
def pooling_layer():
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME', data_format=None)
def tr_conv_layer(filters):
    return Conv2DTranspose(int(filters), (3,3), strides=(2, 2), padding='SAME')

class EncoderArchitecture(TFModel):
    def build(self,config, input_shape=None):
        if not isinstance(config, util.Dottify):
            cfg=util.Dottify(config)
        else:
            cfg=config
        depth=cfg.depth
        filters=cfg.filters
        arch=Sequential([])
        if input_shape is not None:
            arch.add(InputLayer(input_shape=input_shape))
        arch.add(Lambda(lambda Z: tf.expand_dims(Z, 3)))
        for d in range(depth):
            filters*=2
            arch.add(conv_layer(filters))
            arch.add(conv_layer(filters))
            arch.add(pooling_layer())
        return arch

class DecoderArchitecture(TFModel):
    def build(self,config, input_shape=None):
        if not isinstance(config, util.Dottify):
            cfg=util.Dottify(config)
        else:
            cfg=config
        depth=cfg.depth
        filters=cfg.filters*(2**depth)
        arch=Sequential([])
        if input_shape is not None:
            arch.add(InputLayer(input_shape=input_shape))
        arch.add(conv_layer(filters))
        for d in range(depth):
            filters /= 2
            arch.add(tr_conv_layer(filters))
            arch.add(conv_layer(filters))
            arch.add(conv_layer(filters))
        arch.add(conv_layer(1))
        arch.add(Activation('sigmoid'))
        arch.add(Lambda(lambda Z: tf.squeeze(Z,3)))
        return arch
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
