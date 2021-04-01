import os

import pandas as pd
from data_preprocessing.Data import Data


class DataLoader:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load_data_from_path(directory: str):
        with open(os.path.join(directory, "deliveries.jsonl")) as file:
            deliveries = pd.read_json(file, lines=True,
                                      dtype={"purchase_timestamp": "datetime64", "delivery_timestamp": "datetime64"})

        with open(os.path.join(directory, "products.jsonl")) as file:
            products = pd.read_json(file, lines=True)

        with open(os.path.join(directory, "sessions.jsonl")) as file:
            sessions = pd.read_json(file, lines=True)

        with open(os.path.join(directory, "users.jsonl")) as file:
            users = pd.read_json(file, lines=True)

        data = Data(deliveries=deliveries,
                    products=products,
                    sessions=sessions,
                    users=users)
        # print(data.deliveries.dtypes)
        return data


if __name__ == '__main__':
    d = DataLoader.load_data_from_path("../data")
