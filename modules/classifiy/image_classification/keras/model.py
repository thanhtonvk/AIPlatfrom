import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import mobilenet, mobilenet_v2, mobilenet_v3, nasnet, regnet, resnet, resnet_rs, \
    resnet_v2, vgg16, vgg19, xception
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import AbstractRNNCell, Activation, ActivityRegularization, AveragePooling2D, AvgPool2D, \
    BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, LSTM, \
    Flatten, MaxPooling2D, MaxPool2D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, BinaryFocalCrossentropy, \
    SparseCategoricalCrossentropy, KLDivergence, MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad, Nadam

enc = OneHotEncoder()
losses_dict = {
    "BinaryCrossentropy": BinaryCrossentropy,
    "CategoricalCrossentropy": CategoricalCrossentropy,
    "BinaryFocalCrossentropy": BinaryFocalCrossentropy,
    "SparseCategoricalCrossentropy": SparseCategoricalCrossentropy,
    "KLDivergence": KLDivergence,
    "MeanSquaredError": MeanSquaredError,
    "MeanAbsolutePercentageError": MeanAbsolutePercentageError,
    "MeanAbsoluteError": MeanAbsoluteError
}
optimizes_dict = {
    "Adam": Adam,
    "SGD": SGD,
    "Adadelta": Adadelta,
    "Nadam": Nadam,
    "RMSprop": RMSprop,
    "Adagrad": Adagrad
}
structured_dict = {
    "MobileNet": {
        "preprocess": mobilenet.preprocess_input,
        "structured": {
            "MobileNet": mobilenet.MobileNet
        }
    },
    "MobileNetV2": {
        "preprocess": mobilenet_v2.preprocess_input,
        "structured": {
            "MobileNetV2": mobilenet_v2.MobileNetV2
        }
    },
    "MobileNetV3Large": {
        "preprocess": mobilenet_v3.preprocess_input,
        "structured": {
            "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
            "MobiletNetV3Small": tf.keras.applications.MobileNetV3Small
        }
    },
    "MobiletNetV3Small": {
        "preprocess": mobilenet_v3.preprocess_input,
        "structured": {
            "MobiletNetV3Small": tf.keras.applications.MobileNetV3Small
        }
    },
    "NASNetLarge": {
        "preprocess": nasnet.preprocess_input,
        "structured": {
            "NASNetLarge": tf.keras.applications.NASNetLarge,

        }
    },
    "NASNetMobile": {
        "preprocess": nasnet.preprocess_input,
        "structured": {
            "NASNetMobile": tf.keras.applications.NASNetMobile
        }
    },
    "RegNetX002": {
        "preprocess": regnet.preprocess_input,
        "structured": {
            "RegNetX002": tf.keras.applications.RegNetX002,

        }
    },
    "RegNetX160": {
        "preprocess": regnet.preprocess_input,
        "structured": {

            "RegNetX160": tf.keras.applications.RegNetX160,

        }
    },
    "RegNetX320": {
        "preprocess": regnet.preprocess_input,
        "structured": {

            "RegNetX320": tf.keras.applications.RegNetX320,

        }
    },
    "ResNet50": {
        "preprocess": resnet.preprocess_input,
        "structured": {
            "ResNet50": resnet.ResNet50,

        }
    },
    "ResNet101": {
        "preprocess": resnet.preprocess_input,
        "structured": {

            "ResNet101": resnet.ResNet101,

        }
    },
    "ResNet152": {
        "preprocess": resnet.preprocess_input,
        "structured": {

            "ResNet152": resnet.ResNet152

        }
    },
    "ResNetRS50": {
        "preprocess": resnet_rs.preprocess_input,
        "structured": {
            "ResNet50": resnet_rs.ResNetRS50,

        }
    },
    "ResNetRS101": {
        "preprocess": resnet_rs.preprocess_input,
        "structured": {

            "ResNetRS101": resnet_rs.ResNetRS101,

        }
    },
    "ResNetRS152": {
        "preprocess": resnet_rs.preprocess_input,
        "structured": {

            "ResNetRS152": resnet_rs.ResNetRS152
        }
    },
    "ResNet50V2": {
        "preprocess": resnet_v2.preprocess_input,
        "structured": {
            "ResNet50V2": resnet_v2.ResNet50V2,

        }
    },
    "ResNet101V2": {
        "preprocess": resnet_v2.preprocess_input,
        "structured": {

            "ResNet101V2": resnet_v2.ResNet101V2,

        }
    },
    "ResNet152V2": {
        "preprocess": resnet_v2.preprocess_input,
        "structured": {

            "ResNet152V2": resnet_v2.ResNet152V2
        }
    },
    "VGG16": {
        "preprocess": vgg16.preprocess_input,
        "structured": {
            "VGG16": vgg16.VGG16,
        }
    },
    "VGG19": {
        "preprocess": vgg19.preprocess_input,
        "structured": {
            "VGG19": vgg19.VGG19
        }
    },
    "Xception": {
        "preprocess": xception.preprocess_input,
        "structured": {
            "Xception": xception.Xception
        }
    }
}
layers_dict = {
    "AbstractRNNCell": AbstractRNNCell,
    "Activation": Activation,
    "ActivityRegularization": ActivityRegularization,
    "AveragePooling2D": AveragePooling2D,
    "AvgPool2D": AvgPool2D,
    "BatchNormalization": BatchNormalization,
    "Conv2D": Conv2D,
    "Conv2DTranspose": Conv2DTranspose,
    "Dense": Dense,
    "Dropout": Dropout,
    "GlobalAveragePooling2D": GlobalAveragePooling2D,
    "GlobalMaxPooling2D": GlobalMaxPooling2D,
    "LSTM": LSTM,
    "Flatten": Flatten,
    "MaxPolling2D": MaxPooling2D,
    "MaxPool2D": MaxPool2D
}


def build_model(layers, input_size, num_classes, num_channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, num_channels)))
    for layer in layers:
        model.add(layer)
    model.add(Dense(num_classes, activation="softmax"))
    return model


class ModelFromScratch:
    def __init__(self, num_classes=2, model=None, aug=False, tunner=False,
                 image_size=224, epochs=100, num_chanel=3, optimize='adam', loss='categorical_crossentropy',
                 image_path="", quantized=False, export_model='tf'):
        self.quantized = quantized
        self.export_model = export_model
        self.optimize = optimize
        self.epochs = epochs
        self.image_path = image_path
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, num_chanel)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss=loss, optimizer=optimize, metrics=['accuracy'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        X = []
        y = []
        for label in os.listdir(self.image_path):
            for file_name in os.listdir(f"{self.image_path}/{label}"):
                try:
                    image = cv2.imread(
                        f"{self.image_path}/{label}/{file_name}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(
                        image, (self.image_size, self.image_size))
                    image = image / 255.0
                    X.append(image)
                    y.append([label])
                except Exception as e:
                    print(e)
        X = np.array(X)
        y = enc.fit_transform(y).toarray()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self):
        fileweight = "best_weight.hdf5"
        checkpoint = ModelCheckpoint(
            fileweight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        batch_size = 64
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.15,
                                 batch_size=batch_size,
                                 epochs=self.epochs,
                                 callbacks=callbacks_list, verbose=0)
        self.model.load_weights(fileweight)
        return history

    def evaluate(self):
        grouth_trust = np.argmax(self.y_test, axis=1)
        pred = self.model.predict(self.X_test)
        predictions = np.argmax(pred, axis=1)

        print(confusion_matrix(grouth_trust, predictions))

        tn, fp, fn, tp = confusion_matrix(grouth_trust, predictions).ravel()
        Specificity = tn / (tn + fp)

        Sensitivity = tp / (tp + fn)

        Accuracy = accuracy_score(grouth_trust, predictions)
        Precision = precision_score(grouth_trust, predictions)
        Recall = recall_score(grouth_trust, predictions)
        f1_Score = f1_score(grouth_trust, predictions)
        Roc = roc_auc_score(grouth_trust, predictions)
        return Accuracy, Precision, Recall, f1_Score, Roc, Specificity, Sensitivity

    def export(self):
        path = ''
        if self.export_model == 'tf':
            path = './export/model.h5'
            self.model.save(path)
        else:
            if self.quantized:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_quant_model = converter.convert()
                path = './export/model_int8.tflite'
                with open(path, 'wb') as f:
                    f.write(tflite_quant_model)
            else:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                path = './export/model.tflite'
                tflite_quant_model = converter.convert()
                with open(path, 'wb') as f:
                    f.write(tflite_quant_model)
        return path


class TransferLearningModel:
    def __init__(self, structured, num_classes=2, aug=False, fully_connected_layer=None, tunner=False,
                 image_size=224, epochs=100, optimize='adam', loss='categorical_crossentropy', image_path="",
                 quantized=False, export_model='tf'):
        self.quantized = quantized
        self.export_model = export_model
        self.preprocess_input = structured_dict[structured]['preprocess']
        self.optimize = optimize
        self.epochs = epochs
        self.image_path = image_path
        self.image_size = image_size
        self.aug = aug
        base = structured_dict[structured]['structured'][structured](include_top=False, input_shape=(
            self.image_size, self.image_size, 3))
        for layer in base.layers:
            layer.trainable = False
        if fully_connected_layer is None:
            x = base.output
            x = layers_dict["GlobalAveragePooling2D"]()(x)
            x = layers_dict["Dense"](256, activation='relu')(x)
            x = layers_dict["Dropout"](0.2)(x)
            x = layers_dict["Dense"](num_classes, activation='softmax')(x)
            output_layer = x

            self.model = Model(base.input, output_layer)
        else:
            self.model = Model(base.input, fully_connected_layer)
        self.model.compile(optimizer=optimize,
                           loss=loss,
                           metrics=['accuracy'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        X = []
        y = []

        for label in os.listdir(self.image_path):
            for file_name in os.listdir(f"{self.image_path}/{label}"):
                try:
                    image = cv2.imread(
                        f"{self.image_path}/{label}/{file_name}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(
                        image, (self.image_size, self.image_size))
                    image = self.preprocess_input(image)
                    X.append(image)
                    y.append([label])
                except Exception as e:
                    print(e)
        X = np.array(X)
        y = enc.fit_transform(y).toarray()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self):
        fileweight = "best_weight.hdf5"
        checkpoint = ModelCheckpoint(
            fileweight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        batch_size = 64
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.15,
                                 batch_size=batch_size,
                                 epochs=self.epochs,
                                 callbacks=callbacks_list, verbose=0)
        self.model.load_weights(fileweight)
        return history

    def evaluate(self):
        grouth_trust = np.argmax(self.y_test, axis=1)
        pred = self.model.predict(self.X_test)
        predictions = np.argmax(pred, axis=1)

        print(confusion_matrix(grouth_trust, predictions))

        tn, fp, fn, tp = confusion_matrix(grouth_trust, predictions).ravel()
        Specificity = tn / (tn + fp)

        Sensitivity = tp / (tp + fn)

        Accuracy = accuracy_score(grouth_trust, predictions)
        Precision = precision_score(grouth_trust, predictions)
        Recall = recall_score(grouth_trust, predictions)
        f1_Score = f1_score(grouth_trust, predictions)
        Roc = roc_auc_score(grouth_trust, predictions)
        return Accuracy, Precision, Recall, f1_Score, Roc, Specificity, Sensitivity

    def export(self):
        path = ''
        if self.export_model == 'tf':
            path = 'export/model.h5'
            self.model.save('./' + path)
        else:
            if self.quantized:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_quant_model = converter.convert()
                path = 'export/model_int8.tflite'
                with open('./' + path, 'wb') as f:
                    f.write(tflite_quant_model)
            else:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                path = 'export/model.tflite'
                tflite_quant_model = converter.convert()
                with open('./' + path, 'wb') as f:
                    f.write(tflite_quant_model)
        return path
