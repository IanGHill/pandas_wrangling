import pandas as pd
import numpy as np
import matplotlib.pylab as plt

file_name = "usedcars.csv"
headers = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

df = pd.read_csv(file_name, names=headers)
# print(df.head())

# Convert '?' values to NaN

df.replace("?", np.nan, inplace=True)
# print(df.head(5))

# Evaluate for missing data using .isnull() or .notnull()
missing_data = df.isnull()
# print(missing_data.head(5))

# Count the missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# Based on the summary above, each column has 205 rows of data and seven of the columns containing missing data:
# "normalized-losses": 41 missing data
# "num-of-doors": 2 missing data
# "bore": 4 missing data
# "stroke" : 4 missing data
# "horsepower": 2 missing data
# "peak-rpm": 2 missing data
# "price": 4 missing data

# You should only drop whole columns if most entries in the column are empty. In the data set, none of the columns are empty enough to drop entirely. You have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. Apply each method to different columns:

# Replace by mean:
# "normalized-losses": 41 missing data, replace them with mean
# "stroke": 4 missing data, replace them with mean
# "bore": 4 missing data, replace them with mean
# "horsepower": 2 missing data, replace them with mean
# "peak-rpm": 2 missing data, replace them with mean

# Replace by frequency:
# "num-of-doors": 2 missing data, replace them with "four".
# Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur

# Drop the whole row:
# "price": 4 missing data, simply delete the whole row
# Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you.

# Calculate the mean value for the "normalized-losses" column
# Replace "NaN" with mean value in "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
