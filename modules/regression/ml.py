from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeRegressor
import pandas as pd
import numpy as np
import pickle

model_dict = {
    "XGBRegressor": XGBRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR(kernel='linear'),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "ExtraTreeRegressor": ExtraTreeRegressor()
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

        Accuracy = mean_squared_error(self.y_test, predictions)
        Precision = mean_absolute_error(self.y_test, predictions)
        return Accuracy, Precision

    def export_model(self):
        pickle.dump(self.model, open('model.pkl', 'wb'))
        return 'model.pkl'


if __name__ == '__main__':
    model = model_dict['XGBRegressor']
    ml = ML(model, 'D:\AIPlatform\data.xlsx', 'D:\AIPlatform\label.xlsx')
    ml.train()
    print(ml.evaluate())
    ml.export_model()
