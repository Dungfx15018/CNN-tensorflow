import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Add
import keras

class ResNet2Layer(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = 'relu'):
        super(ResNet2Layer, self).__init__()
        self.main = tf.keras.models.Sequential([
            Conv2D(filters, (3,3), strides = strides, padding = 'same'),
            BatchNormalization(),
            Activation(activation),
            Conv2D(filters, (3,3), padding = 'same'),
            BatchNormalization()
        ])
        self.activation = Activation(activation)
    def call(self, inputs):
        x = Add()([self.main(input), self.shortcut(inputs)])
        x = self.activation(x)
        return x
class ResNet3Layer(tf.keras.layers.Layer):
    def __init__(self, filters , strides = 1, activation = 'relu'):
        super(ResNet3Layer, self).__init__()
        self.main = tf.keras.models.Sequential([
            Conv2D(filters, (3,3), strides = strides, padding ='same'),
            BatchNormalization(),
            Activation(activation),
            Conv2D(filters * 4, (3,3), padding = 'same'),
            BatchNormalization()
        ])
        self.shortcut = tf.keras.models.Sequential([
            Conv2D(filters, (1,1), strides= strides, padding = 'same'),
            BatchNormalization()
        ])
        self.activation = Activation(activation)

    def call(self, inputs):
        x = Add()[self.main(input), self.shortcut(inputs)]
        x = self.activation(x)
        return x

class ResNetModel(tf.keras.models.Model):
    def __init__(self, input_shape, numclasses):
        super(ResNetModel, self).__init__()
        self.main = tf.keras.models.Sequential([
            Conv2D(64, (7,7), strides = (2,2), input_shape = input_shape, padding = 'same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')
        ])
        self.numclasses = numclasses


    def add_2d_blocks(self, num_blocks,filters, first_stride):
        for i in range(num_blocks):
            strides = first_stride if i == 0 else 1
            self.model.add(ResNet2Layer(filters, strides))

    def add_3d_blocks(self, num_blocks, filters,first_stride):
        for i in range(num_blocks):
            strides = first_stride if i ==0 else 1
            self.model.add(ResNet3Layer(filters, strides))


class ResNet18(tf.keras.layers.Layer):
    def __init__(self, input_shape, num_classes):
        super(ResNet18, self).__init__(input_shape, num_classes)


        self.add_2d_blocks(2, 64, first_stride = 1)
        self.add_2d_blocks(2, 128, first_stride=2)
        self.add_2d_blocks(2, 256, first_stride=2)
        self.add_2d_blocks(2, 512, first_stride=2)

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(num_classes, activation = 'softmax'))

    def call(self, inputs):
        return self.model(inputs)
class ResNet34(tf.keras.layers.Layer):
    def __int__(self, input_shape, num_classes):
        super(ResNet34, self).__int__(input_shape, num_classes)
