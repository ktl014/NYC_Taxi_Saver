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
from utils.eval_tools import evaluate
from utils.vis_tools import plot_prediction_analysis

# Module level constants
DATA_DIR = ''
N_ROWS = 5000000
SEED = 123
assert os.path.exists(DATA_DIR) # Must input data directory

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
    dataset = data_.prepare(train_model=True)

    model = LinearRegression()
    X_train, X_val, y_train, y_val = train_test_split(dataset, np.array(
        dataset['fare_amount']), stratify=dataset[Col.BIN_FA],
                                                      test_size=0.10, random_state=SEED)

    time_features = ['pickup_frac_day', 'pickup_frac_week', 'pickup_frac_year',
                     'pickup_Elapsed']

    features = ['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count',
                'pickup_latitude', 'pickup_longitude',
                'dropoff_latitude', 'dropoff_longitude'] + time_features

    model.fit(X_train[features], y_train)

    print('Intercept', round(model.intercept_, 4))
    print('abs_lat_diff coef: ', round(model.coef_[0], 4),
          '\tabs_lon_diff coef:', round(model.coef_[1], 4),
          '\tpassenger_count coef:', round(model.coef_[2], 4))

    y_train_pred, y_val_pred = evaluate(model, features, X_train, X_val,
                                      y_train, y_val)
    plot_prediction_analysis(y_val, y_val_pred)



if __name__ == '__main__':
    main()