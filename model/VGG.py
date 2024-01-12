from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential


class VGG16(Sequential):
    def __init__(self, width, height, depth, classes=1000):
        super(VGG16, self).__init__()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
