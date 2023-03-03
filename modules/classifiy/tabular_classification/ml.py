from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import ExtraTreeClassifier
import pandas as pd
import numpy as np
import pickle

model_dict = {
    "XGBoost": XGBClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(kernel='linear', probability=True),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "CalibratedClassifierCV": CalibratedClassifierCV(),
    "ExtraTreeClassifier": ExtraTreeClassifier()
}


class ML:
    def __init__(self, model, file_path_data, file_path_label):
        self.model = model
        self.file_path_data = file_path_data
        self.file_path_label = file_path_label
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        Specificity = tn / (tn + fp)

        Sensitivity = tp / (tp + fn)

        Accuracy = accuracy_score(self.y_test, predictions)
        Precision = precision_score(self.y_test, predictions)
        Recall = recall_score(self.y_test, predictions)
        f1_Score = f1_score(self.y_test, predictions)
        Roc = roc_auc_score(self.y_test, predictions)
        return Accuracy, Precision, Recall, f1_Score, Roc, Specificity, Sensitivity

    def export_model(self):
        pickle.dump(model, open('model.pkl', 'wb'))


if __name__ == '__main__':
    model = model_dict['XGBoost']
    ml = ML(model, 'D:\AIPlatform\data.xlsx', 'D:\AIPlatform\label.xlsx')
    ml.train()
    print(ml.evaluate())
    ml.export_model()
