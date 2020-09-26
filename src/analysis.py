"""Import Path"""
from pathlib import Path

import pandas as pd

EXAMPLE_FILE = 'account.csv'

folder_path = Path("res/")

FILE_PATH = str(folder_path / EXAMPLE_FILE)

dataset = pd.read_csv(FILE_PATH, sep=";")

print(dataset)
