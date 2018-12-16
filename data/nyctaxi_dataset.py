"""Read """
from __future__ import  absolute_import

# Standard dist imports
import calendar
import logging
import os

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from utils.constants import *

# Module level constants
DATA_DIR = '/Users/ktl014/Google Drive/ECE Classes/ECE 225/project/data/all'


class NYCTaxiDataset(object):
    """Helper functions to prepare NYCDataset"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s |  %(levelname)s: %(message)s')
        self.logger.setLevel(logging.DEBUG)

    def clean_null_values(self, data):
        """Clean null values in coordinates"""
        self.logger.info('\t cleaning null values...')
        self.logger.debug("\t\tOld size: {}".format(data.shape))
        for i in COORD:
            data[i] = data[i].replace(0, np.nan)
            data = data[data[i].notnull()]
        self.logger.debug("\t\tNew size: {}".format(data.shape))
        return data

    def extract_datetime(self, data):
        """Extract datetime parameters from given datetime"""
        # Convert to date format
        fmt = "%Y-%m-%d %H:%M:%S"
        self.logger.info('\t extracting datetime variables...')
        data[Col.DATETIME] = data[Col.DATETIME].str.replace("UTC", "")
        data[Col.DATETIME] = pd.to_datetime(data[Col.DATETIME], format=fmt)

        # Extract year, month, day, and hour
        data[Col.YEAR] = pd.DatetimeIndex(data[Col.DATETIME]).year
        data[Col.MONTH] = pd.DatetimeIndex(data[Col.DATETIME]).month
        data[Col.MONTH_NAME] = data[Col.MONTH].apply(lambda x:
                                                     calendar.month_abbr[x])
        data[Col.MONTH_YEAR] = data[Col.YEAR].astype(str) + ' - ' + data[
            Col.MONTH_NAME]
        data[Col.WEEK_DAY] = data[Col.DATETIME].dt.weekday_name
        data[Col.DAY] = data[Col.DATETIME].dt.day
        data[Col.HOUR] = data[Col.DATETIME].dt.hour
        return data

    def clean_outliers(self, data):
        """Clean outliers"""
        self.logger.info('\t cleaning outliers...')
        self.logger.debug("\t\tOld size: {}".format(data.shape))
        # Passenger Count
        data = data[(data[Col.COUNT] > 0) & data[Col.COUNT] < 7]

        # Fare Amount
        data = data[(data[Col.FA] > 0) &
                    data[Col.FA] < data[Col.FA].quantile(.9999)]

        # Pickup and dropoff locations
        for i in COORD:
            data = data[(data[i] > data[i].quantile(.001)) &
                        (data[i] < data[i].quantile(.999))]
        self.logger.debug("\t\tNew size: {}".format(data.shape))
        return data

    def generate_feature(self, data):
        """Generate additional features from existing features"""
        data[Col.LOG_FA] = np.log(data[Col.FA])

        #TODO write distance function in d.utils
        R = 6373.0
        cdict = {c: np.radians(data[c]) for c in COORD}
        dist_lon = cdict[Col.DO_LONG] - cdict[Col.PU_LONG]
        dist_lat = cdict[Col.DO_LAT] - cdict[Col.PU_LAT]

        a = (np.sin(dist_lat / 2)) ** 2 + np.cos(cdict[Col.PU_LAT]) * np.cos(
            cdict[Col.DO_LAT]) * (np.sin(dist_lon / 2)) ** 2
        c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )
        d = R * c

        data[Col.TRIP_DIST] = d
        data[Col.LOG_TRIP] = np.log(data[Col.TRIP_DIST])
        self.logger.info('\t generated features...')
        return data
