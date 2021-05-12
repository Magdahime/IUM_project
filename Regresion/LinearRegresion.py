from data_preprocessing.DataLoader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from data_preprocessing.DataLoader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score


class BayesClassifier:

    def __init__(self) -> None:
        super().__init__()


def unite_sets(deliveries, products, sessions, users):
    deliveries["deltas"] = deliveries["delivery_timestamp"] - deliveries["purchase_timestamp"]

    # divide category
    categories = products.category_path.str.split(';', expand=True)
    products = pd.concat([products, categories], axis=1)
    products = products.drop(columns=['category_path'])
    products = products.rename(
        columns={0: "primary_category", 1: "secondary_category", 2: "tertiary_category", 3: "quaternary_category"})


    deliveries_sessions = pd.merge(deliveries, sessions, left_on="purchase_id", right_on="purchase_id")
    deliveries_sessions_users = pd.merge(deliveries_sessions, users, left_on="user_id", right_on="user_id")
    deliveries_sessions_users_products = pd.merge(deliveries_sessions_users, products, left_on="product_id", right_on="product_id")

    deliveries_sessions_users_products.to_csv('../out.csv')
    return deliveries_sessions_users_products


def one_hot_encode(train, s):
    y = pd.get_dummies(train.city, prefix=s)
    train = train.join(other=y)
    train = train.loc[:, train.columns != s]
    return train


if __name__ == '__main__':
    d = DataLoader.load_data_from_path("../newData")

    products = d.products
    deliveries = d.deliveries
    sessions = d.sessions
    users = d.users

    unitated = unite_sets(deliveries, products, sessions, users)

    # nie wnosi zadnej informacji wiec wyrzucamy daną kolumnę
    unitated = unitated.loc[:, unitated.columns != 'event_type']
    unitated = unitated.loc[:, unitated.columns != 'name']
    unitated = unitated.loc[:, unitated.columns != 'street']
    unitated = unitated.loc[:, unitated.columns != 'product_name']
    unitated = unitated.loc[:, unitated.columns != 'delivery_timestamp']
    unitated = unitated.loc[:, unitated.columns != 'purchase_timestamp']
    unitated = unitated.loc[:, unitated.columns != 'timestamp']
    unitated = unitated.loc[:, unitated.columns != 'purchase_id']
    unitated = unitated.loc[:, unitated.columns != 'product_id']
    unitated = unitated.loc[:, unitated.columns != 'user_id']
    unitated = unitated.loc[:, unitated.columns != 'session_id']

    # unitated['purchase_timestamp'] = pd.to_datetime(unitated['purchase_timestamp'])
    # unitated['purchase_timestamp'] = unitated['purchase_timestamp'].map(dt.datetime.toordinal)
    #
    # unitated['delivery_timestamp'] = pd.to_datetime(unitated['delivery_timestamp'])
    # unitated['delivery_timestamp'] = unitated['delivery_timestamp'].map(dt.datetime.toordinal)
    #
    # unitated['timestamp'] = pd.to_datetime(unitated['timestamp'])
    # unitated['timestamp'] = unitated['timestamp'].map(dt.datetime.toordinal)

    # train['deltas'] = pd.to_datetime(train['deltas'])
    # train['deltas'] = train['deltas'].map(dt.datetime.toordinal)

    # tworzenie one hot encoding dla kolumny city
    y = pd.get_dummies(unitated.city, prefix='city')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'city']

    y = pd.get_dummies(unitated.primary_category, prefix='primary_category')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'primary_category']

    y = pd.get_dummies(unitated.secondary_category, prefix='secondary_category')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'secondary_category']

    y = pd.get_dummies(unitated.tertiary_category, prefix='tertiary_category')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'tertiary_category']

    y = pd.get_dummies(unitated.quaternary_category, prefix='quaternary_category')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'quaternary_category']

    y = pd.get_dummies(unitated.delivery_company, prefix='delivery_company')
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != 'delivery_company']

    unitated['deltas'] = pd.to_numeric(unitated['deltas'].dt.days, downcast='integer')

    # normalizacja deltas
    # unitated['deltas'] = unitated['deltas'] /unitated['deltas'].abs().max()

    train, test = train_test_split(unitated, test_size=0.2)
    X_train = train.loc[:, train.columns != 'deltas']
    y_train = train['deltas']

    X_test = test.loc[:, test.columns != 'deltas']
    y_test = test['deltas']

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
