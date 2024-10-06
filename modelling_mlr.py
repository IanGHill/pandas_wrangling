import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("clean_used_car_df.csv")
# print(df.head())


lm = LinearRegression()

# Multiple linear regression to predict price based on more than one variable:
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
lm.fit(Z, df["price"])
print(lm.intercept_)
print(lm.coef_)

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(
    0,
)
# plt.show()
print(df[["peak-rpm", "highway-mpg", "price"]].corr())


# How do we visualize a model for Multiple Linear Regression?
# This gets a bit more complicated because you can't visualize it with regression or residual plot.
# One way to look at the fit of the model is by looking at the distribution plot.
# We can look at the distribution of the fitted values that result from the model and compare it to
# the distribution of the actual values.
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df["price"], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)


plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")

# plt.show()
# plt.close()

# Calculate the R^2 and MSE
lm.fit(Z, df["price"])
print("The R-square is: ", lm.score(Z, df["price"]))
Y_predict_multifit = lm.predict(Z)
print(
    "The mean square error of price and predicted value using multifit is: ",
    mean_squared_error(df["price"], Y_predict_multifit),
)
# a lower MSE and a higher R-squared is a better fit model
# When comparing models, the model with the higher R-squared value is a better fit for the data.
# When comparing models, the model with the smallest MSE value is a better fit for the data.
