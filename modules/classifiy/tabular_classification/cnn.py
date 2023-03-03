from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from modules.classifiy.tabular_classification.cnn_backbone import CNN, MobileNet, VGG

backbone_model_dict = {
    "CNN": CNN.CNN,
    "MobileNetV1": MobileNet.MobileNet.MobileNet_v1,
    "MobileNetV2": MobileNet.MobileNet.MobileNet_v2,
    "MobileNetV3Small": MobileNet.MobileNet.MobileNet_v3_Small,
    "MobileNetV3Large": MobileNet.MobileNet.MobileNet_v3_Large,
    "VGG": VGG.VGG.VGG11
}


def one_hot_encoding(data):
    L_E = LabelEncoder()
    integer_encoded = L_E.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded_data = onehot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded_data


class CNN:
    def __init__(self, model, file_path_data, file_path_label, num_classes, optimize='adam',
                 loss='categorical_crossentropy', epochs=100):

        self.file_path_data = file_path_data
        self.file_path_label = file_path_label
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()
        self.input_shape = (self.X_train.shape[1], 1)
        self.kernel_size = self.input_shape[0] - 1
        self.num_classes = num_classes
        self.optimize = optimize
        self.loss = loss
        self.model = model(self.input_shape, self.kernel_size, self.num_classes).CustomCNN()
        self.model.compile(loss=self.loss, optimizer=self.optimize, metrics=['accuracy'])
        self.epochs = epochs

    def load_data(self):
        X = []
        y = []
        extension_file = self.file_path_data.split('.')[-1]
        if extension_file == 'csv':
            X = pd.read_csv(self.file_path_data)
            y = pd.read_csv(self.file_path_label)
        elif extension_file == 'xlsx':
            X = pd.read_excel(self.file_path_data)
            y = pd.read_excel(self.file_path_label)
        X = np.asarray(X, dtype='float')
        y = np.asarray(y, dtype='float')

        y = np.reshape(y, len(y))
        L_E = LabelEncoder()
        L_E.fit_transform(y)
        y = one_hot_encoding(y.ravel())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self):
        fileweight = "best_weight.hdf5"
        checkpoint = ModelCheckpoint(
            fileweight, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        batch_size = 64
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                                 batch_size=batch_size,
                                 epochs=self.epochs,
                                 callbacks=callbacks_list, verbose=0)
        self.model.load_weights(fileweight)

        return history

    def evaluate(self):
        grouth_trust = np.argmax(self.y_test, axis=1)
        pred = self.model.predict(self.X_test)
        predictions = np.argmax(pred, axis=1)

        tn, fp, fn, tp = confusion_matrix(grouth_trust, predictions).ravel()
        specificity = tn / (tn + fp)

        Sensitivity = tp / (tp + fn)

        Accuracy = accuracy_score(grouth_trust, predictions)
        Precision = precision_score(grouth_trust, predictions)
        Recall = recall_score(grouth_trust, predictions)
        f1_Score = f1_score(grouth_trust, predictions)
        Roc = roc_auc_score(grouth_trust, predictions)
        return Accuracy, Precision, Recall, f1_Score, Roc, specificity, Sensitivity

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


if __name__ == '__main__':
    model = backbone_model_dict['CNN']
    ml = CNN(model=model, file_path_data='D:\AIPlatform\data.xlsx', file_path_label='D:\AIPlatform\label.xlsx',
             num_classes=2)
    ml.train()
    print(ml.evaluate())
