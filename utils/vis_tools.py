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
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization


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
