import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_preprocessing.DataLoader import DataLoader
from data_preprocessing.DataPreprocessing import DataPreprocessing


class BayesClassifier:

    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':
    data = DataLoader.load_data_from_path("../data")

    unitated = DataPreprocessing.prepareDate(data)

    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'city')
    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'primary_category')
    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'secondary_category')
    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'tertiary_category')
    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'quaternary_category')
    unitated = DataPreprocessing.one_hot_encode_column(unitated, 'delivery_company')

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
