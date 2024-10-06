import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("clean_used_car_df.csv")
# print(df.head())

# Linear regression to predict price based on one variable:

lm = LinearRegression()

X = df[["highway-mpg"]]
Y = df["price"]
lm.fit(X, Y)
Yhat = lm.predict(X)
print(Yhat[0:5])
print(lm.intercept_)
print(lm.coef_)

# Use a residual plot to visulise the variance of the data
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df["highway-mpg"], y=df["price"])
# plt.show()

# Calculate the R^2
print("The R-square is: ", lm.score(X, Y))
Yhat = lm.predict(X)
print("The output of the first four predicted value is: ", Yhat[0:4])
# Calculate the MSE
mse = mean_squared_error(df["price"], Yhat)
print("The mean square error of price and predicted value is: ", mse)
