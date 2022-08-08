import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import sqrt

train = pd.read_csv("training.csv")
print(train.head())

test = pd.read_csv("testing.csv")
print(test.head())

model = LinearRegression()

X_train = train.drop("Fahrenheit", axis=1)

y_train = train.loc[:, "Fahrenheit"]

#LineerRegression
model.fit(X_train, y_train)

X_test = test.drop("Fahrenheit", axis=1)
y_test = test.loc[:, "Fahrenheit"]

predictions = model.predict(X_test)

comparison = pd.DataFrame({"Actual Values": y_test, "Predictions": predictions})

# .head() method lists only top 5 row of the predictions
print(comparison.head())
