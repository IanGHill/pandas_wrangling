import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("clean_used_car_df.csv", header=0)
# print(df.head())
# print(df.dtypes)
# print(df.corr(numeric_only=True))
print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())
# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(
#     0,
# )
# plt.show()
# sns.boxplot(x="body-style", y="price", data=df)
# plt.show()
print(df.describe())
print(df.describe(include=["object"]))

# Value counts is a good way of understanding how many units of each characteristic/variable we have.
# We can apply the "value_counts" method on the column "drive-wheels".
# Don’t forget the method "value_counts" only works on pandas series, not pandas dataframes.
# As a result, we only include one bracket df['drive-wheels'], not two brackets df[['drive-wheels']].
print(df["drive-wheels"].value_counts())
print(df["drive-wheels"].value_counts().to_frame())
# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and
# rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels": "value_counts"}, inplace=True)
drive_wheels_counts.index.name = "drive-wheels"
print(drive_wheels_counts)
print(df.corr(numeric_only=True))
# engine-location as variable
engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts.rename(columns={"engine-location": "value_counts"}, inplace=True)
engine_loc_counts.index.name = "engine-location"
print(engine_loc_counts.head(10))

# Group by drive wheel type and get the average price for each group
df_group_one = df[["drive-wheels", "body-style", "price"]]
df_grouped = df_group_one.groupby(["drive-wheels"], as_index=False).agg(
    {"price": "mean"}
)
# print(df_grouped)

# Can group by more than one variable:
grouped_test1 = df_group_one.groupby(
    ["drive-wheels", "body-style"], as_index=False
).mean()
print(grouped_test1)
# This is easier visualised as a pivot table:
grouped_pivot = grouped_test1.pivot(index="drive-wheels", columns="body-style")
grouped_pivot = grouped_pivot.fillna(0)  # fill missing values with 0
print(grouped_pivot)

# use the grouped results to display a heatmap
# fig, ax = plt.subplots()
# im = ax.pcolor(grouped_pivot, cmap="RdBu")

# # label names
# row_labels = grouped_pivot.columns.levels[1]
# col_labels = grouped_pivot.index

# # move ticks and labels to the center
# ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# # insert labels
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(col_labels, minor=False)

# # rotate label if too long
# plt.xticks(rotation=90)

# fig.colorbar(im)
# plt.show()

pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print(
    "The Pearson Correlation Coefficient is",
    pearson_coef,
    " with a P-value of P =",
    p_value,
)
# The Pearson Correlation Coefficient is 0.5846418222655085  with a P-value of P = 8.076488270732338e-20
# Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant,
# although the linear relationship isn't extremely strong (~0.585).

df_test = df[["body-style", "price"]]
df_grp = df_test.groupby(["body-style"], as_index=False).mean()
print(df_grp["price"])
