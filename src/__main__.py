import sys
import database as db
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import prepare_development_dataset, prepare_evaluation_dataset

from knn import knn_loan
from svm import svm_loan
from crforest import crforest_loan, max_AUC, current_AUC #switch these last two imports to being from xg_boost to test that algorithm
from xg_boost import xg_boost

model_switcher = {
    "knn": knn_loan,
    "svm": svm_loan,
    "forest": crforest_loan,
    "xgboost": xg_boost 
}

account_dataset, client_dataset, disp_dataset, district_dataset = db.parse_data()
loan_train_dataset, trans_train_dataset = db.parse_train()
loan_test_dataset, trans_test_dataset = db.parse_test()

loan_train_dataset = prepare_development_dataset(
    loan_train_dataset, trans_train_dataset, disp_dataset, account_dataset, district_dataset, client_dataset)

train, test = train_test_split(loan_train_dataset, test_size=0.35, random_state=42, shuffle=True)


loan_test_dataset = prepare_evaluation_dataset(
    loan_test_dataset, trans_test_dataset, disp_dataset, account_dataset, district_dataset, client_dataset)

if len(sys.argv) < 2:
    print('Warning: No algorithm was chosen.')
    print('Usage: python3 src <knn|svm>.')
    exit()
def remove_outliers(dataset):
    high = 0.999
    low = 1 - high
    quant_df = dataset.quantile([low, high])

    filtered_dataset = dataset.loc[:, dataset.columns != 'status']
    filtered_dataset = filtered_dataset.apply(lambda x: x[(x > quant_df.loc[low, x.name]) &
                                                          (x < quant_df.loc[high, x.name])], axis=0)

    filtered_dataset = pd.concat(
        [filtered_dataset, dataset.loc[:, 'status']], axis=1)
    filtered_dataset.dropna(inplace=True)
    print(filtered_dataset)

    return filtered_dataset


def prepare_development_dataset(dataset, trans, disp, account, district, client):
    joined_dataset = join_and_encode_dataset(
        dataset, trans, disp, account, district, client
    )

    # joined_dataset = remove_outliers(joined_dataset)

    return joined_dataset


def prepare_evaluation_dataset(dataset, trans, disp, account, district, client):
    joined_dataset = join_and_encode_dataset(
        dataset, trans, disp, account, district, client
    )

    return joined_dataset

while(1):
   
    model = model_switcher.get(sys.argv[1], knn_loan)

    result = model(train, test, loan_test_dataset)
    
    result.to_csv('output.csv', index=False)


