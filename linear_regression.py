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
DATA_DIR = '/Users/ktl014/Google Drive/ECE Classes/ECE 225/project/data/all'
N_ROWS = 10000
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

    # Partition dataset
    X_train, X_val, y_train, y_val = train_test_split(dataset, np.array(
        dataset[Col.FA]), stratify=dataset[Col.BIN_FA], test_size=0.10,
                                                      random_state=SEED)
    # Select features
    time_features = [Col.pickup_frac_day,
                     Col.pickup_frac_week,
                     Col.pickup_frac_year,
                     Col.pickup_elapsed]

    features = [Col.ABS_LON_DIFF, Col.ABS_LAT_DIFF, Col.TRIP_DIST,
                Col.COUNT, Col.PU_LAT, Col.PU_LONG,
                Col.DO_LAT, Col.DO_LONG] + time_features

    # Train model
    model = LinearRegression()
    model.fit(X_train[features], y_train)

    logger.info('Intercept', round(model.intercept_, 4))
    logger.info('abs_lat_diff coef: ', round(model.coef_[0], 4),
          '\tabs_lon_diff coef:', round(model.coef_[1], 4),
          '\tpassenger_count coef:', round(model.coef_[2], 4))

    # Evaluate model
    y_train_pred, y_val_pred = evaluate(model, features, X_train, X_val,
                                      y_train, y_val)
    plot_prediction_analysis(y_val, y_val_pred, visualize=True)



if __name__ == '__main__':
    main()