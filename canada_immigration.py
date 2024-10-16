import numpy as np
import pandas as pd

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx

df_can = pd.read_excel(
    "canada.xlsx",
    sheet_name="Canada by Citizenship",
    skiprows=range(20),
    skipfooter=2,
)

print("Data read into a pandas dataframe!")
# print(df_can.head())
# print(df_can.tail())
# print(df_can.info(verbose=True))
# print(df_can.columns)
# print(df_can.index)
# Note: The default type of intance variables `index` and `columns` are **NOT** `list`.
# print(type(df_can.columns))
# print(type(df_can.index))
# print(df_can.columns.tolist())
# print(df_can.index.tolist())
# size of dataframe (rows, columns)
print(df_can.shape)  # (195,43)

# Let's clean the dataset by removing unecessary columns
# in pandas axis=0 represents rows (default) and axis=1 represents columns.
df_can.drop(["AREA", "REG", "DEV", "Type", "Coverage"], axis=1, inplace=True)
print(df_can.head(2))
# Let's rename some of the columns to make the names clearer
df_can.rename(
    columns={"OdName": "Country", "AreaName": "Continent", "RegName": "Region"},
    inplace=True,
)
# print(df_can.columns)
# Let's add a total column to sum the immigration numbers over the entire period 1980-2013
df_can["Total"] = df_can.sum(axis=1, numeric_only=True)
# print(df_can["Total"])

# Check for nulls
print(df_can.isnull().sum())  # there are none in this case
# Get some summary stats:
# print(df_can.describe())

### Select Column
# There are two ways to filter on a column name:**

# Method 1: Quick and easy, but only works if the column name does NOT have spaces or special characters
# print(df_can.Country)  # returns series


# Method 2: More robust, and can filter on multiple columns.
# print(df_can["Country"])  # returns series
# print(df_can[["Country", "Total"]])  # returns dataframe
# print(df_can[["Country", 1980, 1981, 1982, 1983, 1984, 1985]])
# notice that 'Country' is string, and the years are integers.
# for the sake of consistency, we will convert all column names to string later on.

# Also notice that the index of the dataframe is a numeric range 0 - 194
# We can change the index to be the country column, so it is easier to select
# by country name
df_can.set_index("Country", inplace=True)
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
# print(df_can.head(3))
# optional: to remove the name of the index
# df_can.index.name = None

# View the full row data for Japan (all columns)
print(df_can.loc["Japan"])  # iloc would be df_can.iloc[87]
print(df_can[df_can.index == "Japan"])  #
print(df_can.loc["Japan", 2013])  # 982
print(df_can.loc["Japan", [1980, 1981, 1982, 1983, 1984, 1984]])

# Convert the column names into strings
df_can.columns = list(map(str, df_can.columns))
# check
# [print(type(x)) for x in df_can.columns.values]
# Since we converted the years to string, let's declare a variable that will
# allow us to easily call upon the full range of years:
# useful for plotting later on
years = list(map(str, range(1980, 2014)))
print(years)

# filter the datafrmae based on a condition - returns a boolean series
condition = df_can["Continent"] == "Asia"
print(condition)
# pass this condition into the dataframe
print(df_can[condition])  # retuns a df with only asian countries
# can add more complex conditions:
print(df_can[(df_can["Continent"] == "Asia") & (df_can["Region"] == "Southern Asia")])

# Can also sort the df:
df_can.sort_values(by="Total", ascending=False, axis=0, inplace=True)
top_5 = df_can.head(5)
print(top_5)
# Find out top 3 countries that contributes the most to immigration to Canda in the year 2010.
df_can.sort_values(by="2010", ascending=False, axis=0, inplace=True)
top3_2010 = df_can["2010"].head(3)
print(top3_2010)
