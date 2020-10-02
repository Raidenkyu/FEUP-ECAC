import database as db
import pandas as pd
from sklearn.model_selection import train_test_split

from knn import knn_model, knn_predict

account_dataset, client_dataset, disp_dataset, district_dataset = db.parse_data()
loan_train_dataset, trans_train_dataset = db.parse_train()
loan_test_dataset, trans_test_dataset = db.parse_test()

train, test = train_test_split(
    loan_train_dataset, test_size=0.25, random_state=42, shuffle=True
)

knn, scaler = knn_model(train, test)
result = knn_predict(knn, scaler, loan_test_dataset)

result.to_csv('output.csv', index=False)
