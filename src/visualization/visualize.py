import matplotlib.pyplot as plt
from sklearn import tree
import logging

def plot_tree(model, feature_names):
    try:
        plt.figure(figsize=(20, 10))
        tree.plot_tree(model, feature_names=list(feature_names))
        plt.savefig('tree.png', dpi=300)  # Save the plot to a file first
        logging.info('Decision Tree plot saved successfully.')
        plt.show()  # This will pop up the window with the tree plot
        logging.info('Decision Tree plot displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying Decision Tree plot: {e}')
        raise
