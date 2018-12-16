""" """
# Standard dist imports
import logging
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
palette = sns.color_palette('Paired', 10)

# Project level imports
from data.dataset import Dataset
from utils.constants import *
from utils.logger import Logger

# Module level constants
DATA_DIR = '/Users/ktl014/Google Drive/ECE Classes/ECE 225/project/data/all'
N_ROWS = 5000000

def main():
    # Initialize Logger
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestr = time.strftime('%m%d%H%M')
    log_fn = os.path.join(log_dir, 'linear_regression_%s.log' % timestr)
    Logger(log_fn, logging.INFO)
    logger = logging.getLogger(__name__)
    Logger.section_break(title='Linear Regression')

    # Load dataset
    data_ = Dataset(data_dir=DATA_DIR, n_rows=N_ROWS)
    dataset = data_.prepare()
    dataset, labels = data_.get_labels(dataset)
    dataset = dataset.drop([Col.MONTH_NAME, Col.MONTH_YEAR, Col.WEEK_DAY],
                           axis=1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.25)

    # test_ = Dataset(data_dir=DATA_DIR, split=TEST)
    # test_x = test_.prepare()
    # test_y = test_x[Col.FA]

    model = Pipeline((
        ("standard_scaler", StandardScaler()),
         ("linear_reg", LinearRegression())))

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(rmse)


if __name__ == '__main__':
    main()