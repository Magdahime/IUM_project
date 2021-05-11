from data_preprocessing.DataLoader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

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


if __name__ == '__main__':
    d = DataLoader.load_data_from_path("../newData")

    products = d.products
    deliveries = d.deliveries
    sessions = d.sessions
    users = d.users
    
    unitated = unite_sets(deliveries, products, sessions, users)

    print(unitated.shape)
    train, test = train_test_split(unitated, test_size=0.2)

    print(train.shape)
    print(test.shape)

