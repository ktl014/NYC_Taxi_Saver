"""Visualization tools"""

# Standard dist imports

# Third party imports

# Project level imports
from utils.stats_tools import ecdf
from utils.constants import *

# Module level constants
#Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #operating system dependent modules of Python
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
import itertools


def plot_ecdf(data):
    # Example of ecdf
    xs, ys = ecdf(np.logspace(0, 2))
    plt.plot(xs, ys, '.')
    plt.ylabel('Percentile')
    plt.title('ECDF')

    xs, ys = ecdf(data[Col.FA])
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, '.')
    plt.ylabel('Percentile');
    plt.title('ECDF of Fare Amount');
    plt.xlabel('Fare Amount ($)');


from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_prediction_analysis(y, y_pred, figsize=(10, 4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))

    axs[1].hist(y - y_pred, bins=50)
    avg = np.mean(y - y_pred)
    std = np.std(y - y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title(
        'Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(
            avg, std))

    if title != '':
        fig.suptitle(title)



