from typing import Dict

import pandas as pd
import os
import html


def load_data_from_path(directory: str) -> Dict:
    dfDict = {}
    for entry in os.scandir(directory):
        if entry.path.endswith(".jsonl") and entry.is_file():
            fileHandler = open(entry.path, encoding='utf-8')
            fileContents = html.unescape(fileHandler.read())
            label = entry.name.split('.')[0]
            dfDict[label] = pd.read_json(fileContents, lines=True)
    return dfDict

if __name__ == '__main__':
    load_data_from_path(directory = r'data/')
