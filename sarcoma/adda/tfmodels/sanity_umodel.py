from tfmodels import USequential
from tfmodels import TFModel, Remember
from tensorflow.keras.layers import Input,Dense,Flatten,Activation,Lambda
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf

class Architecture(TFModel):
    """docstring for TFModel."""
    def __init__(self):
        self.architecture=self.build()
        self.output_models={}
    def build(self):
        return USequential([
            Dense(5,activation=Activation('sigmoid')),
            Flatten(),
            Remember(Dense(1,activation=Activation('sigmoid'))),
            Lambda(lambda y: tf.squeeze(y,1)),
        ])

    def __call__(self, input_tensor):
        if input_tensor not in self.output_models.keys():
            input_layer=Input(batch_shape=input_tensor.shape)
            self.output_models[input_tensor] = Model(inputs=[input_layer], outputs=[self.architecture(input_layer)])(input_tensor)
        return self.output_models[input_tensor]
