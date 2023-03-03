from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling1D


class CNN:
    def __init__(self, input_shape, kernel_size, num_classes):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_classes = num_classes

    def CustomCNN(self):
        model = Sequential()
        model.add(Conv1D(input_shape=self.input_shape, filters=32, kernel_size=self.kernel_size,
                         padding='same', activation='relu', strides=1))
        model.add(Conv1D(64, (3), activation='relu'))
        model.add(MaxPooling1D(pool_size=(2), strides=1))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax', use_bias=True, name='last_layer'))
        return model
