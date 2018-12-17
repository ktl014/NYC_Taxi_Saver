""" """
from __future__ import  absolute_import

# Standard dist imports
import logging
import os
import time

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from data.nyctaxi_dataset import NYCTaxiDataset
from utils.logger import Logger
from utils.constants import *

# Module level constants
DATA_DIR = '/Users/ktl014/Google Drive/ECE Classes/ECE 225/project/data/all'


class Dataset(object):
    def __init__(self, data_dir=DATA_DIR, split=TRAIN, n_rows=None):
        csv_file = os.path.join(data_dir, '{}.csv'.format(split))
        self.logger = logging.getLogger(__name__)
        self.logger.info('READING DATASET')
        if n_rows:
            self.data = pd.read_csv(csv_file, nrows=n_rows)
        else:
            self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.db)

    def prepare(self, train_model=False):
        """Prepares dataset for conducting statistics"""
        since = time.time()
        self.nyc = NYCTaxiDataset()
        Logger.section_break('PREPARING DATASET')
        self.logger.debug(self.data.describe())
        # Clean null values
        data = self.nyc.clean_null_values(self.data)

        # Extract datetime features
        if train_model:
            data = self.nyc.extract_dateinfo(data)
        else:
            data = self.nyc.extract_datetime(data)

        # Clean outliers
        data = self.nyc.clean_outliers(data)

        # Generate trip distance and fare amount features
        data = self.nyc.generate_feature(data)

        # Sort values
        data = data.sort_values(by=Col.DATETIME,
                                ascending=False).reset_index(drop=True)

        Logger.section_break('List of Features')
        for i,c in enumerate(sorted(data.columns.values)):
            self.logger.info('{}. {}'.format(i+1, c))
        Logger.section_break('Dataset Statistics')
        self.logger.info('Dataset size: {}'.format(data.shape))
        self.logger.info('Elapsed time: {}'.format(time.time() - since))
        return data

    def transform(self):
        """Transform dataset for model input"""


    def get_labels(self, data):
        labels = data[Col.FA]
        data = data.drop(Col.FA, axis=1)
        return data, labels

if __name__ == '__main__':
    data_ = Dataset()
