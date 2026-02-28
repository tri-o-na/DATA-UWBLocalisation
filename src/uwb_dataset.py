"""
Created on Feb 6, 2017
@author: Klemen Bregar
"""

import os
import pandas as pd
from numpy import vstack

def import_from_files():
    """
    Read .csv files and store data into a DataFrame
    format: |LOS|NLOS|data...|
    """
    rootdir = '../data/raw/'
    output_arr = []
    first = True

    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            if not file.endswith('.csv'):
                continue  # skip non-CSV files
            filename = os.path.join(dirpath, file)
            print(f"Reading: {filename}")

            # Read CSV
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.to_numpy()

            # Stack arrays
            if first:
                output_arr = input_data
                first = False
            else:
                output_arr = vstack((output_arr, input_data))

    return output_arr


if __name__ == '__main__':
    print("Importing dataset to numpy array")
    print("-------------------------------")
    data = import_from_files()
    print("-------------------------------")
    print("Number of samples in dataset: %d" % len(data))
    print("Length of one sample: %d" % len(data[0]))
    print("-------------------------------")
    print("First 3 rows:")
    print(data[:3])