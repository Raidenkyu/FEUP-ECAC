import sys
import database as db
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import prepare_dataset
from knn import knn_loan

model_switcher = {
    "knn": knn_loan
}

account_dataset, client_dataset, disp_dataset, district_dataset = db.parse_data()
loan_train_dataset, trans_train_dataset = db.parse_train()
loan_test_dataset, trans_test_dataset = db.parse_test()

loan_train_dataset = prepare_dataset(loan_train_dataset)

train, test = train_test_split(
    loan_train_dataset, test_size=0.25, random_state=42, shuffle=True
)

if len(sys.argv) < 2:
    exit

model = model_switcher.get(sys.argv[1], knn_loan)

result = model(train, test, loan_test_dataset)

result.to_csv('output.csv', index=False)
