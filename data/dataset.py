""" """
from __future__ import  absolute_import

# Standard dist imports
import os
import time

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from data.nyctaxi_dataset import NYCTaxiDataset
from utils.constants import *

# Module level constants
DATA_DIR = '/Users/ktl014/Google Drive/ECE Classes/ECE 225/project/data/all'


class Dataset(object):
    def __init__(self, data_dir=DATA_DIR, split=TRAIN):
        csv_file = os.path.join(data_dir, '{}.csv'.format(split))
        print('READING DATASET')
        self.data = pd.read_csv(csv_file, nrows=5000000)

    def __len__(self):
        return len(self.db)

    def prepare(self):
        since = time.time()
        self.nyc = NYCTaxiDataset()
        print('PREPARING DATASET')
        # Clean null values
        data = self.nyc.clean_null_values(self.data)

        # Extract datetime features
        data = self.nyc.extract_datetime(data)

        # Clean outliers
        data = self.nyc.clean_outliers(data)

        # Generate trip distance and fare amount features
        data = self.nyc.generate_feature(data)

        # Sort values
        data = data.sort_values(by=Col.DATETIME,
                                ascending=False).reset_index(drop=True)
        print('Dataset size: {}'.format(data.shape))
        print('Elapsed time: {}'.format(time.time() - since))
        return data

    def transform(self):
        """Transform dataset"""
        pass
