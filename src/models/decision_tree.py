from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import logging

class DecisionTreeModel:
    def __init__(self, max_depth=3, max_features=10, random_state=567):
        try:
            self.model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state)
            logging.info('DecisionTreeRegressor initialized.')
        except Exception as e:
            logging.error(f'Error initializing DecisionTreeRegressor: {e}')
            raise

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info('Decision Tree model trained.')
        except Exception as e:
            logging.error(f'Error training Decision Tree model: {e}')
            raise

    def evaluate(self, x, y):
        try:
            predictions = self.model.predict(x)
            mae = mean_absolute_error(predictions, y)
            logging.info('Decision Tree model evaluated.')
            return mae
        except Exception as e:
            logging.error(f'Error evaluating Decision Tree model: {e}')
            raise
