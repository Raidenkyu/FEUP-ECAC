from pathlib import Path
import pandas as pd


def read_dataset(dataset_name):
    folder_path = Path("res/")

    file_path = str(folder_path / dataset_name)
    dataset = pd.read_csv(file_path, sep=";")

    return dataset


def parse_data():
    account_dataset = read_dataset("account.csv")
    client_dataset = read_dataset("client.csv")
    disp_dataset = read_dataset("disp.csv")
    district_dataset = read_dataset("district.csv")

    return account_dataset, client_dataset, disp_dataset, district_dataset


def parse_train():
    loan_train_dataset = read_dataset("loan_train.csv")
    trans_train_dataset = read_dataset("trans_train.csv")

    return loan_train_dataset, trans_train_dataset


def parse_test():
    loan_test_dataset = read_dataset("loan_test.csv")
    trans_test_dataset = read_dataset("trans_test.csv")

    return loan_test_dataset, trans_test_dataset
