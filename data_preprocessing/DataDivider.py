import os

import pandas as pd
from data_preprocessing.Data import Data


class DataDivider:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def divide_data(data):
        train, test = train_test_split(df, test_size=0.2)