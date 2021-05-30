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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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

    # deliveries_sessions_users_products.to_csv('../out.csv')
    return deliveries_sessions_users_products


def one_hot_encode(df, s):
    y = pd.get_dummies(df.city, prefix=s)
    df = df.join(other=y)
    df = df.loc[:, df.columns != s]
    return df


def delete_unwanted_columns(unitated):
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
    return unitated


def one_hot_encode(unitated, s):
    y = pd.get_dummies(unitated[s], prefix=s)
    unitated = unitated.join(other=y)
    unitated = unitated.loc[:, unitated.columns != s]
    return unitated


if __name__ == '__main__':
    d = DataLoader.load_data_from_path("../newData")

    products = d.products
    deliveries = d.deliveries
    sessions = d.sessions
    users = d.users

    unitated = unite_sets(deliveries, products, sessions, users)

    unitated = delete_unwanted_columns(unitated)

    unitated = one_hot_encode(unitated,'city')
    unitated = one_hot_encode(unitated, 'primary_category')
    unitated = one_hot_encode(unitated, 'secondary_category')
    unitated = one_hot_encode(unitated, 'tertiary_category')
    unitated = one_hot_encode(unitated, 'quaternary_category')
    unitated = one_hot_encode(unitated, 'delivery_company')

    unitated['deltas'] = pd.to_numeric(unitated['deltas'].dt.days, downcast='integer')

    # normalizacja deltas
    # unitated['deltas'] = unitated['deltas'] /unitated['deltas'].abs().max()

    train, test = train_test_split(unitated, test_size=0.2)
    X_train = train.loc[:, train.columns != 'deltas']
    y_train = train['deltas']

    X_train.to_csv('../X_train.csv')

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
    print('mean absolute error: \n', mean_absolute_error(y_test, y_pred))
