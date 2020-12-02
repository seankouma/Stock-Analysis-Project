import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import datetime
import joblib
from math import sqrt


data = pd.read_csv("result.csv", sep=',', names=['date', 'open', 'high', 'low', 'close', 'volume'])

def train_data():
    X = data.drop(["high"], axis=1)
    y = data["high"]
    new_dates = []
    dates = data["date"]
    for day in dates:
        day = datetime.datetime.strptime(day, "%Y-%m-%d")
        day2 = (day - datetime.datetime(1970, 1, 1)).total_seconds()
        new_dates.append(day2)
    X["date"] = new_dates
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    pipeline = make_pipeline(preprocessing.StandardScaler(),
                             RandomForestRegressor(n_estimators=100))

    # hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    #                    'randomforestregressor__max_depth': [None, 5, 3, 1], }

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    #print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))

    joblib.dump(clf, 'stock_predictor.pkl')

def predict_price():
    clf = joblib.load('stock_predictor.pkl')
    print("-" * 48)
    print("Enter the details of the date you would like to predict")
    print("\n")
    option = input("Year: ")
    year = 2020
    option = input("Month number (00): ")
    month = 11
    option = input("Day number (00): ")
    theday = 15

    day = str(year) + "-" + str(month) + "-" + str(theday)
    day = datetime.datetime.strptime(day, "%Y-%m-%d")
    date = (day - datetime.datetime(1970, 1, 1)).total_seconds()

    day_x = str(year) + "-" + str(month) + "-" + str(theday)
    day_x = datetime.datetime.strptime(day_x, "%Y-%m-%d")
    date_x = (day_x - datetime.datetime(1970, 1, 1)).total_seconds()

    X = [[date, date_x]]
    print("\n")
    print("-" * 48)
    print("The stock high is predicted to be: " + str(clf.predict(X)[0]))
    print("The stock high was actually: " + str(get_the_price(day)))
    print("-" * 48)
    print("\n")

def get_the_price(date):
    price = data["date"]
    temp = data["high"]

    for i in range(0, len(price)):
        day = datetime.datetime.strptime(price[i], "%Y-%m-%d")
        if (day == date):
            return temp[i]

if __name__ == "__main__":
    train_data()
    predict_price()
