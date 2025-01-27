import sys
import database as db
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import prepare_development_dataset, prepare_evaluation_dataset, forward_selection

from knn import knn_loan
from svm import svm_loan
# max_AUC, current_AUC #switch these last two imports to being from xg_boost to test that algorithm
from crforest import crforest_loan
from xg_boost import xg_boost
from gbm import gbm

model_switcher = {
    "knn": knn_loan,
    "svm": svm_loan,
    "forest": crforest_loan,
    "xgboost": xg_boost,
    "gbm": gbm
}

account_dataset, client_dataset, disp_dataset, district_dataset = db.parse_data()
loan_train_dataset, trans_train_dataset = db.parse_train()
loan_test_dataset, trans_test_dataset = db.parse_test()

loan_train_dataset = prepare_development_dataset(
    loan_train_dataset, trans_train_dataset, disp_dataset, account_dataset, district_dataset, client_dataset)

train, test = train_test_split(
    loan_train_dataset, test_size=0.25, random_state=42, shuffle=True)


selected_features = forward_selection(train.drop(
    columns=["status"]), train.iloc[:, -1])


loan_test_dataset = prepare_evaluation_dataset(
    loan_test_dataset, trans_test_dataset, disp_dataset, account_dataset, district_dataset, client_dataset)

if len(sys.argv) < 2:
    print('Warning: No algorithm was chosen.')
    print('Usage: python3 src <knn|svm>.')
    exit()

model = model_switcher.get(sys.argv[1], crforest_loan)

result = model(train, test, loan_test_dataset, selected_features)

result.to_csv('output.csv', index=False)
