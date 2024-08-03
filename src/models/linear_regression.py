from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import logging

class LinearRegressionModel:
    def __init__(self):
        try:
            self.model = LinearRegression()
            logging.info('LinearRegression initialized.')
        except Exception as e:
            logging.error(f'Error initializing LinearRegression: {e}')
            raise

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info('Linear Regression model trained.')
        except Exception as e:
            logging.error(f'Error training Linear Regression model: {e}')
            raise

    def evaluate(self, x, y):
        try:
            predictions = self.model.predict(x)
            mae = mean_absolute_error(predictions, y)
            logging.info('Linear Regression model evaluated.')
            return mae
        except Exception as e:
            logging.error(f'Error evaluating Linear Regression model: {e}')
            raise
