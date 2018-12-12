"""Read """
from __future__ import  absolute_import

# Standard dist imports
import calendar
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

    def clean_null_values(self, data):
        """Clean null values in coordinates"""
        for i in COORD:
            data[i] = data[i].replace(0, np.nan)
            data = data[data[i].notnull()]
        print('\t cleaned null values...')
        return data

    def extract_datetime(self, data):
        """Extract datetime parameters from given datetime"""
        # Convert to date format
        fmt = "%Y-%m-%d %H:%M:%S"
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
        print('\t extracted datetime variables...')
        return data

    def clean_outliers(self, data):
        """Clean outliers"""
        data = data[(data[Col.COUNT] > 0) & data[Col.COUNT] < 7]
        data = data[(data[Col.FA] > 0) &
                    data[Col.FA] < data[Col.FA].quantile(.9999)]

        for i in COORD:
            data = data[(data[i] > data[i].quantile(.001)) &
                        (data[i] < data[i].quantile(.999))]
        print('\t cleaned outliers...')
        return data

    def generate_feature(self, data):
        """Generate additional features from existing features"""
        data[Col.LOG_FA] = np.log(data[Col.FA])

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
        print('\t generated features...')
        return data
