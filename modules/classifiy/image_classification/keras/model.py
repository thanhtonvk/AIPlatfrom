from tensorflow.keras.layers import AbstractRNNCell, Activation, ActivityRegularization, AveragePooling2D, AvgPool2D, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, LSTM, Flatten
import os
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import mobilenet, mobilenet_v2, mobilenet_v3, nasnet, regnet, resnet, resnet_rs, resnet_v2, vgg16, vgg19, xception
import cv2
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

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
    "MobileNetV3": {
        "preprocess": mobilenet_v3.preprocess_input,
        "structured": {
            "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
            "MobiletNetV3Small": tf.keras.applications.MobileNetV3Small
        }
    },
    "NASNet": {
        "preprocess": nasnet.preprocess_input,
        "structured": {
            "NASNetLarge": tf.keras.applications.NASNetLarge,
            "NASNetMobile": tf.keras.applications.NASNetMobile
        }
    },
    "RegNet": {
        "preprocess": regnet.preprocess_input,
        "structured": {
            "RegNetX002": tf.keras.applications.RegNetX002,
            "RegNetX160": tf.keras.applications.RegNetX160,
            "RegNetX320": tf.keras.applications.RegNetX320,

        }
    },
    "ResNet": {
        "preprocess": resnet.preprocess_input,
        "structured": {
            "ResNet50": resnet.ResNet50,
            "ResNet101": resnet.ResNet101,
            "ResNet152": resnet.ResNet152

        }
    },
    "ResNetRS": {
        "preprocess": resnet_rs.preprocess_input,
        "structured": {
            "ResNet50": resnet_rs.ResNetRS50,
            "ResNetRS101": resnet_rs.ResNetRS101,
            "ResNetRS152": resnet_rs.ResNetRS152
        }
    },
    "ResNetV2": {
        "preprocess": resnet_v2.preprocess_input,
        "structured": {
            "ResNet50V2": resnet_v2.ResNet50V2,
            "ResNet101V2": resnet_v2.ResNet101V2,
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
    "Flatten": Flatten
}


class TransferLearningModel:
    def __init__(self, structured, base_model, num_classes, aug=False, fully_connected_layer=None, tunner=False, image_size=224, epochs=100, optimize='adam', loss='categorical_crossentropy', image_path=""):
        self.preprocess_input = structured['preprocess']
        self.optimize = optimize
        self.epochs = epochs
        self.image_path = image_path
        self.image_size = image_size
        self.aug = aug
        base = base_model(include_top=False, input_shape=(
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
        Specificity = tn/(tn + fp)

        Sensitivity = tp / (tp + fn)

        Accuracy = accuracy_score(grouth_trust, predictions)
        Precision = precision_score(grouth_trust, predictions)
        Recall = recall_score(grouth_trust, predictions)
        f1_Score = f1_score(grouth_trust, predictions)
        Roc = roc_auc_score(grouth_trust, predictions)
        return Accuracy, Precision, Recall, f1_Score, Roc, Specificity, Sensitivity

    def export(self, export_type='tf', quantized=False):
        if export_type == 'tf':
            self.model.save('model.h5')
        else:
            if quantized:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_quant_model = converter.convert()
                with open('model_int8.tflite', 'wb') as f:
                    f.write(tflite_quant_model)
            else:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_quant_model = converter.convert()
                with open('model.tflite', 'wb') as f:
                    f.write(tflite_quant_model)
