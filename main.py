import logging
from src.data.make_dataset import load_data
from src.features.build_features import split_data
from src.models.linear_regression import LinearRegressionModel
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from src.visualization.visualize import plot_tree

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/final.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        x_train, x_test, y_train, y_test = split_data(df, 'price')
        logging.info('Data split into training and testing sets.')
    except Exception as e:
        logging.error(f'Error splitting data: {e}')
        return
    
    # Train and evaluate Linear Regression Model
    try:
        lr_model = LinearRegressionModel()
        lr_model.train(x_train, y_train)
        train_mae = lr_model.evaluate(x_train, y_train)
        test_mae = lr_model.evaluate(x_test, y_test)
        logging.info(f'Linear Regression - Train MAE: {train_mae}, Test MAE: {test_mae}')
    except Exception as e:
        logging.error(f'Error with Linear Regression model: {e}')
    
    # Train and evaluate Decision Tree Model
    try:
        dt_model = DecisionTreeModel()
        dt_model.train(x_train, y_train)
        train_mae = dt_model.evaluate(x_train, y_train)
        test_mae = dt_model.evaluate(x_test, y_test)
        logging.info(f'Decision Tree - Train MAE: {train_mae}, Test MAE: {test_mae}')
        plot_tree(dt_model.model, feature_names=list(df.drop('price', axis=1).columns))
        logging.info('Decision Tree plot displayed.')
    except Exception as e:
        logging.error(f'Error with Decision Tree model: {e}')
    
    # Train and evaluate Random Forest Model
    try:
        rf_model = RandomForestModel()
        rf_model.train(x_train, y_train)
        train_mae = rf_model.evaluate(x_train, y_train)
        test_mae = rf_model.evaluate(x_test, y_test)
        logging.info(f'Random Forest - Train MAE: {train_mae}, Test MAE: {test_mae}')
        rf_model.save_model('models/random_forest_model.pkl')
        logging.info('Random Forest model saved.')
    except Exception as e:
        logging.error(f'Error with Random Forest model: {e}')

if __name__ == '__main__':
    main()