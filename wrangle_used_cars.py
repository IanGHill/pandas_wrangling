import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot

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
# df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df.replace(
    {"normalized-losses": np.nan}, {"normalized-losses": avg_norm_loss}, inplace=True
)

# Calculate the mean value for the "bore" column
# Replace "NaN" with the mean value in the "bore" column
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)
# df["bore"].replace(np.nan, avg_bore, inplace=True)
df.replace({"bore": np.nan}, {"bore": avg_bore}, inplace=True)

# Calculate the mean value for the "stroke" column
# Replace "NaN" with the mean value in the "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df.replace({"stroke": np.nan}, {"stroke": avg_stroke}, inplace=True)

# Calculate the mean value for the "horsepower" column
# Replace "NaN" with the mean value in the "horsepower" column
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)
df.replace({"horsepower": np.nan}, {"horsepower": avg_horsepower}, inplace=True)

# Calculate the mean value for "peak-rpm" column
# Replace "NaN" with the mean value in the "peak-rpm" column
avg_peakrpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of Peak-Rpm:", avg_peakrpm)
df.replace({"peak-rpm": np.nan}, {"peak-rpm": avg_peakrpm}, inplace=True)

# For number of doors we would replace missing values by the most common or mode
door_mode = df["num-of-doors"].value_counts().idxmax()
print("Mode of num doors:", door_mode)
df.replace({"num-of-doors": np.nan}, {"num-of-doors": door_mode}, inplace=True)

# Finally, drop any rows that have no price data (Nan in price column):
df.dropna(subset=["price"], axis=0, inplace=True)
# reset the index as we dropped two rows:
df.reset_index(drop=True, inplace=True)
# print(df.head())

# list the data types
# print(df.dtypes)
# some data types are incorrect, so replace them:
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
# list the data types
# print(df.dtypes)

# Standardization is the process of transforming data into a common format,
# allowing the researcher to make the meaningful comparison.
# Example - Transform mpg to L/100km:
df["city-L/100km"] = 235 / df["city-mpg"]
df["highway-L/100km"] = 235 / df["highway-mpg"]
# check your transformed data
print(df.head())

# Normalization is the process of transforming values of several variables into a similar range.
# Typical normalizations include
# - scaling the variable so the variable average is 0
# - scaling the variable so the variance is 1
# - scaling the variable so the variable values range from 0 to 1

# replace (original value) by (original value)/(maximum value)
df["length"] = df["length"] / df["length"].max()
df["width"] = df["width"] / df["width"].max()
df["height"] = df["height"] / df["height"].max()
print(df[["length", "width", "height"]].head())

# Binning is a process of transforming continuous numerical variables into
# discrete categorical 'bins' for grouped analysis.
df["horsepower"] = df["horsepower"].astype(int, copy=True)
hist = plt.hist(df["horsepower"])

# # set x/y labels and plot title
# plt.xlabel("horsepower")
# plt.ylabel("count")
# plt.title("horsepower bins")
# # plt.show()

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(
    df["horsepower"], bins, labels=group_names, include_lowest=True
)
print(df[["horsepower", "horsepower-binned"]].head(10))
print(df["horsepower-binned"].value_counts())
plt.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
# plt.show()

# Indicator Variable - What is an indicator variable?
# An indicator variable (or dummy variable) is a numerical variable used to label categories.
#  They are called 'dummies' because the numbers themselves don't have inherent meaning.
# Why use indicator variables?
# You use indicator variables so you can use categorical variables for regression analysis in the later modules.
# Example
# The column "fuel-type" has two unique values: "gas" or "diesel".
# Regression doesn't understand words, only numbers. To use this attribute in regression analysis,
# you can convert "fuel-type" to indicator variables.
# Use the Panda method 'get_dummies' to assign numerical values to different categories of fuel type.
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())
dummy_variable_1.rename(
    columns={"gas": "fuel-type-gas", "diesel": "fuel-type-diesel"}, inplace=True
)
print(dummy_variable_1.head())
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis=1, inplace=True)
# print(df.head())

dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(
    columns={"std": "aspiration-std", "turbo": "aspiration-turbo"}, inplace=True
)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)
# print(df.head())

# Save the cleaned CSV as a new file:
df.to_csv("clean_used_car_df.csv")
