import pandas as pd

from data_preprocessing.Data import Data


class DataPreprocessing:
    def __init__(self) -> None:
        super().__init__()

    # Function for labeling rows
    @staticmethod
    def __labelTimeOfDay(row):
        hour = row['purchase_timestamp'].hour
        if (hour >= 6 and hour < 12):
            return "Morning"
        elif (hour >= 12 and hour < 18):
            return "Afternoon"
        elif (hour >= 18 and hour < 24):
            return "Evening"
        else:
            return "Night"

    @staticmethod
    def prepareDate(data: Data):
        deliveries = data.deliveries
        products = data.products
        sessions = data.sessions
        users = data.users

        deliveries = DataPreprocessing.prepossess_deliveries(deliveries)
        # products = DataPreprocessing.prepossess_products(products)

        df_all_data = DataPreprocessing.merge_all_data(deliveries, products, sessions, users)

        DataPreprocessing.add_feature_time_of_day(df_all_data)
        DataPreprocessing.add_feature_weekday(df_all_data)

        df_all_data = DataPreprocessing.delete_unwanted_columns(df_all_data)

        # united = one_hot_encode(united, 'city')
        # united = one_hot_encode(united, 'delivery_company')
        # united = one_hot_encode(united, 'time_of_day')
        # united = one_hot_encode(united, 'weekday')

        # united['deltas'] = pd.to_numeric(united['deltas'].dt.days, downcast='integer')

        df_all_data["delivery_company"] = df_all_data["delivery_company"].fillna(9999)  # TODO remove null values

        df_all_data.to_csv("out.csv")
        return df_all_data

    @staticmethod
    def add_feature_weekday(df_all_data):
        df_all_data['weekday'] = df_all_data['purchase_timestamp'].dt.day_name()

    @staticmethod
    def add_feature_time_of_day(df_all_data):
        df_all_data.loc[:, 'time_of_day'] = df_all_data.apply(lambda row: DataPreprocessing.__labelTimeOfDay(row),
                                                              axis=1)

    @staticmethod
    def prepossess_products(products):
        categories = products.category_path.str.split(';', expand=True)
        products = pd.concat([products, categories], axis=1)
        products = products.drop(columns=['category_path'])
        products = products.rename(
            columns={0: "primary_category", 1: "secondary_category", 2: "tertiary_category", 3: "quaternary_category"})
        return products

    @staticmethod
    def prepossess_deliveries(deliveries):
        deliveries["deltas"] = deliveries["delivery_timestamp"] - deliveries["purchase_timestamp"]
        return deliveries

    @staticmethod
    def merge_all_data(deliveries, products, sessions, users):
        deliveries_sessions = pd.merge(deliveries, sessions, left_on="purchase_id", right_on="purchase_id")
        deliveries_sessions_users = pd.merge(deliveries_sessions, users, left_on="user_id", right_on="user_id")
        deliveries_sessions_users_products = pd.merge(deliveries_sessions_users, products, left_on="product_id",
                                                      right_on="product_id")

        return deliveries_sessions_users_products

    @staticmethod
    def delete_unwanted_columns(df):
        df = df.loc[:, df.columns != 'event_type']
        df = df.loc[:, df.columns != 'name']
        df = df.loc[:, df.columns != 'street']
        df = df.loc[:, df.columns != 'product_name']
        df = df.loc[:, df.columns != 'delivery_timestamp']
        df = df.loc[:, df.columns != 'timestamp']
        df = df.loc[:, df.columns != 'purchase_id']
        df = df.loc[:, df.columns != 'product_id']
        df = df.loc[:, df.columns != 'user_id']
        df = df.loc[:, df.columns != 'session_id']
        df = df.loc[:, df.columns != 'purchase_timestamp']
        df = df.loc[:, df.columns != 'category_path']

        return df

    @staticmethod
    def one_hot_encode_column(df, column_name):
        y = pd.get_dummies(df[column_name], prefix=column_name)
        df = df.join(other=y)
        df = df.loc[:, df.columns != column_name]
        return df
