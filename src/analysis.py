# Import packages
import os

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

example_file = 'account.csv'

folder_path = Path("res/")

file_path = str(folder_path / example_file)

dataset = pd.read_csv(file_path, sep=";")

print(dataset)
