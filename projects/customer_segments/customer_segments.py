# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs


# Pretty display for notebooks
# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# For each feature find the data points with extreme high or low values
log_data = np.log(data)

def quarter(a,presntage):
    q=(len(a)/presntage)
    print a[q]

for feature in log_data.keys():
     
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    
    Q1 = np.array( sorted(log_data[feature]))[(len(log_data[feature])*25)/100] 
    print feature 
    print Q1
#    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.array( sorted(log_data[feature]))[(len(log_data[feature])*75)/100]
    print feature
    print Q3
#    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
#    step = None
#    
#    # Display the outliers
#    print "Data points considered outliers for the feature '{}':".format(feature)
#    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
#    
## OPTIONAL: Select the indices for data points you wish to remove
#outliers  = []
#
## Remove the outliers, if any were specified
#good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)