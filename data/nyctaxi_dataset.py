"""Prepares dataset specific for NYC-TaxiFares"""
from __future__ import  absolute_import

# Standard dist imports
import calendar
import logging
import os
import re

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from utils.constants import *
from utils.geospatial_tools import minkowski_distance, haversine_np

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

    def clean_outliers(self, data):
        """Clean outliers"""
        self.logger.info('\t cleaning outliers...')
        self.logger.debug("\t\tOld size: {}".format(data.shape))
        # Passenger Count outliers if greater than 7
        data = data[(data[Col.COUNT] > 0) & data[Col.COUNT] < 7]

        # Fare Amount outliers < 2.5 and > 100
        neg = "There are {} negative fares.".format(len(data[data['fare_amount'] < 0]))
        zero = "There are {} $0 fares.".format(len(data[data['fare_amount'] == 0]))
        greater = "There are {} fares greater than $100.".format(len(data[data['fare_amount'] > 100]))
        for msg in [neg, zero, greater]:
            self.logger.debug('\t{}'.format(msg))

        data = data[data[Col.FA].between(left=2.5, right=100)]

        # Pickup and dropoff locations
        for i in COORD:
            self.logger.debug('\t{:17}: 2.5% = {:5} \t 97.5% = {}'.
                              format(i.capitalize(),
                                     round(np.percentile(data[i], 2.5), 2),
                              round(np.percentile(data[i], 97.5), 2)))

        # Remove latitude and longtiude outliers
        data = data.loc[data[Col.PU_LAT].between(40, 42)]
        data = data.loc[data[Col.PU_LONG].between(-75, -72)]
        data = data.loc[data[Col.DO_LAT].between(40, 42)]
        data = data.loc[data[Col.DO_LONG].between(-75, -72)]

        self.logger.debug("\t\tNew size: {}".format(data.shape))
        return data

    def extract_dateinfo(self, data, date_col, drop=True, time=False,
                         start_ref=pd.datetime(1900, 1, 1),
                         extra_attr=False):
        """Extract Date (and time) Information from a DataFrame"""
        df = data.copy()

        # Extract the field
        fld = df[date_col]

        # Check the time
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        # Convert to datetime if not already
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[date_col] = fld = pd.to_datetime(fld,
                                                infer_datetime_format=True)

        # Prefix for new columns
        pre = re.sub('[Dd]ate', '', date_col)
        pre = re.sub('[Tt]ime', '', pre)

        # Basic attributes
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Days_in_month', 'is_leap_year']

        # Additional attributes
        if extra_attr:
            attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end',
                           'Is_quarter_start', 'Is_year_end', 'Is_year_start']

        # If time is specified, extract time information
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']

        # Iterate through each attribute
        for n in attr:
            df[pre + n] = getattr(fld.dt, n.lower())

        # Calculate days in year
        df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365

        if time:
            # Add fractional time of day (0 - 1) units of day
            df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (
                        df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24

            # Add fractional time of week (0 - 1) units of week
            df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[
                pre + 'frac_day']) / 7

            # Add fractional time of month (0 - 1) units of month
            df[pre + 'frac_month'] = (df[pre + 'Day'] + (
            df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] + 1)

            # Add fractional time of year (0 - 1) units of year
            df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[
                pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)

        # Add seconds since start of reference
        df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()

        if drop:
            df = df.drop(date_col, axis=1)

        return df

    def generate_feature(self, data):
        """Generate additional features from existing features"""
        self.logger.info('\t generating features...')
        # Calculate log of fare amount
        data[Col.LOG_FA] = np.log(data[Col.FA])

        # Bin fare amounts
        # Bin the fare and convert to string
        data[Col.BIN_FA] = pd.cut(data[Col.FA],
                                  bins=list(range(0, 50, 5))).astype(str)

        # Uppermost bin
        data.loc[data[Col.BIN_FA] == 'nan', Col.BIN_FA] = '[45+]'

        # Adjust bin so the sorting is correct
        data.loc[data[Col.BIN_FA] == '(5, 10]', Col.BIN_FA] = '(05, 10]'

        # Calculate haversine distance
        data[Col.TRIP_DIST] = haversine_np(data[Col.PU_LONG],
                                               data[Col.DO_LONG],
                                               data[Col.PU_LAT],
                                               data[Col.DO_LAT])

        data[Col.LOG_TRIP] = np.log(data[Col.TRIP_DIST])

        # Calculate absolute difference in latitude and longitude
        data[Col.ABS_LAT_DIFF] = (data[Col.DO_LAT] - data[Col.PU_LAT]).abs()
        data[Col.ABS_LON_DIFF] = (data[Col.DO_LONG] - data[Col.PU_LONG]).abs()

        # Calculate manhattan & euclid distance for long & lat
        data[Col.MANHAT] = minkowski_distance(data[Col.PU_LONG],
                                               data[Col.DO_LONG],
                                               data[Col.PU_LAT],
                                               data[Col.DO_LAT], p=1)

        data[Col.EUCLID] = minkowski_distance(data[Col.PU_LONG],
                                               data[Col.DO_LONG],
                                               data[Col.PU_LAT],
                                               data[Col.DO_LAT], p=2)

        return data

