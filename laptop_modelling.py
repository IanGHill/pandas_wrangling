import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("laptops.csv", header=0)
# print(df.head(5))

# Simple Linear Regression

lm = LinearRegression()
X = df[["CPU_frequency"]]
Y = df["Price"]
lm.fit(X, Y)
Yhat = lm.predict(X)
print("The output of the first four predicted value is: ", Yhat[0:4])

# Generate the Distribution plot for the predicted values and that of the actual values.
# ax1 = sns.distplot(df["Price"], hist=False, color="r", label="Actual Value")

# Create a distribution plot for predicted values
# sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)

# plt.title("Actual vs Fitted Values for Price")
# plt.xlabel("Price")
# plt.ylabel("Proportion of laptops")
# plt.legend(["Actual Value", "Predicted Value"])
# plt.show()

# Evaluate the Mean Squared Error and R^2 score values for the model.

# Calculate the R^2
print("The R-square is: ", lm.score(X, Y))

# Calculate the MSE
mse = mean_squared_error(df["Price"], Yhat)
print("The mean square error of price and predicted value is: ", mse)

# Multiple Linear Regression - The parameters which have a low enough p-value so as to indicate
# strong relationship with the 'Price' value are 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD',
# 'CPU_core', 'OS', 'GPU' and 'Category'. Use all these variables to create a Multiple Linear Regression system.

lm1 = LinearRegression()
Z = df[
    ["CPU_frequency", "RAM_GB", "Storage_GB_SSD", "CPU_core", "OS", "GPU", "Category"]
]
lm1.fit(Z, Y)
Y_hat = lm1.predict(Z)

# Plot the Distribution graph of the predicted values as well as the Actual values
# ax1 = sns.distplot(df["Price"], hist=False, color="r", label="Actual Value")
# sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)

# plt.title("Actual vs Fitted Values for Price")
# plt.xlabel("Price")
# plt.ylabel("Proportion of laptops")
# plt.show()

# Calculate the R^2 and MSE
print("The R-square is: ", lm1.score(Z, df["Price"]))

print(
    "The mean square error of price and predicted value using multifit is: ",
    mean_squared_error(df["Price"], Y_hat),
)

# Polynomial Regression - Use the variable "CPU_frequency" to create Polynomial features.
# Try this for 3 different values of polynomial degrees.
#  Remember that polynomial fits are done using numpy.polyfit

X = X.to_numpy().flatten()
f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, ".", x_new, y_new, "-")
    plt.title(f"Polynomial Fit for Price ~ {Name}")
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel("Price of laptops")


# PlotPolly(p1, X, Y, "CPU_frequency")
# PlotPolly(p3, X, Y, "CPU_frequency")
# PlotPolly(p5, X, Y, "CPU_frequency")
# plt.show()
r_squared_1 = r2_score(Y, p1(X))
print("The R-square value for 1st degree polynomial is: ", r_squared_1)
print("The MSE value for 1st degree polynomial is: ", mean_squared_error(Y, p1(X)))
r_squared_3 = r2_score(Y, p3(X))
print("The R-square value for 3rd degree polynomial is: ", r_squared_3)
print("The MSE value for 3rd degree polynomial is: ", mean_squared_error(Y, p3(X)))
r_squared_5 = r2_score(Y, p5(X))
print("The R-square value for 5th degree polynomial is: ", r_squared_5)
print("The MSE value for 5th degree polynomial is: ", mean_squared_error(Y, p5(X)))

# Create a pipeline that performs parameter scaling, Polynomial Feature generation and Linear regression.
# Use the set of multiple features as before to create this pipeline.

Input = [
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression()),
]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z, Y)
ypipe = pipe.predict(Z)

print("MSE for multi-variable polynomial pipeline is: ", mean_squared_error(Y, ypipe))
print("R^2 for multi-variable polynomial pipeline is: ", r2_score(Y, ypipe))

# Model Refinement
# Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)
# Initialize a Ridge regressor that used hyperparameter alpha = 0.1.
# Fit the model using training data data subset. Print the R2 score for the testing data.
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test, yhat))

# Apply polynomial transformation to the training parameters with degree=2.
# Use this transformed feature set to fit the same regression model, as above, using the training subset.
# Print the R2 score for the testing subset.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test, y_hat))
