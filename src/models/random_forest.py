from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os
import logging

class RandomForestModel:
    def __init__(self, n_estimators=200, criterion='absolute_error'):
        try:
            self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
            logging.info('RandomForestRegressor initialized.')
        except Exception as e:
            logging.error(f'Error initializing RandomForestRegressor: {e}')
            raise

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info('Random Forest model trained.')
        except Exception as e:
            logging.error(f'Error training Random Forest model: {e}')
            raise

    def evaluate(self, x, y):
        try:
            predictions = self.model.predict(x)
            mae = mean_absolute_error(predictions, y)
            logging.info('Random Forest model evaluated.')
            return mae
        except Exception as e:
            logging.error(f'Error evaluating Random Forest model: {e}')
            raise

    def save_model(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as file:
                pickle.dump(self.model, file)
            logging.info(f'Random Forest model saved to {filename}.')
        except Exception as e:
            logging.error(f'Error saving Random Forest model: {e}')
            raise

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.model = pickle.load(file)
            logging.info(f'Random Forest model loaded from {filename}.')
            return self.model
        except Exception as e:
            logging.error(f'Error loading Random Forest model: {e}')
            raise
