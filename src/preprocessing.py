import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
import statsmodels.api as sm


def join_aux(dataset, disp, account, district, client):
    joined_client = disp.set_index("client_id", drop=False).join(
        client.set_index("client_id"), rsuffix='_other')

    joined_account = joined_client.set_index("account_id", drop=False).join(
        account.set_index("account_id"), rsuffix='_other')

    joined_district = joined_account.set_index("district_id", drop=False).join(
        district.set_index("code "), rsuffix='_other')

    joined = joined_district.set_index("account_id", drop=False).join(
        dataset.set_index("account_id", drop=False), rsuffix='_other')

    joined = joined.drop(
        columns=["disp_id", "client_id", "account_id", "district_id",
                 "district_id_other", "account_id_other", "date_other", "type"]
    ).dropna(subset=["loan_id"]).set_index("loan_id")

    enc = OrdinalEncoder()
    joined.iloc[:, 0:21] = enc.fit_transform(joined.iloc[:, 0:21])

    joined = joined.groupby('loan_id').mean()

    return joined


def join_client(disp, client, district):

    joined_client = district.set_index("code ", drop=False).join(
        client.set_index("district_id", drop=False))

    joined_client = disp.set_index("client_id", drop=False).join(
        joined_client.set_index("client_id"))

    joined_client = joined_client[["account_id", "ratio of urban inhabitants "]].groupby(
        "account_id").min()  # ADICIONAR FATORES EXTERNOS AQUI

    return joined_client


def join_trans(dataset, trans):
    trans_min_balance = trans[[
        "account_id", "balance"]].groupby("account_id").min()

    trans_average_balance = trans[[
        "account_id", "balance"]].groupby("account_id").mean()

    trans_average_amount = trans[[
        "account_id", "amount"]].groupby("account_id").mean()

    trans_average_type = trans[[
        "account_id", "type"]].groupby("account_id").agg(lambda x: x.value_counts().index[0])

    trans_count = trans[[
        "account_id"]].groupby("account_id").size().to_frame(name='trans_count')

    dataset['date'] = dataset['date'] // 10000

    joined_trans = trans_average_amount.join(trans_min_balance)

    joined_trans = joined_trans.join(trans_average_type)
    joined_trans = joined_trans.join(trans_count)
    joined_trans = joined_trans.replace(['withdrawal', 'credit'], [0, 1])

    joined_trans = joined_trans.join(
        trans_average_balance, lsuffix="_account_minimum", rsuffix="_account_average")

    return joined_trans


def join_and_encode_dataset(dataset, trans, disp, account, district, client):
    joined1 = join_trans(dataset, trans)
    joined2 = join_client(disp, client, district)
    joined3 = joined1.join(joined2)

    joined = dataset.set_index("account_id", drop=False).join(
        joined3, lsuffix='_loan', rsuffix='_account_average'
    ).reindex(columns=["loan_id", "date", "account_id", "amount_loan",
                       "payments", "amount_account_average", "balance_account_minimum",
                       "balance_account_average", "trans_count", "type", "ratio of urban inhabitants ", "status"])

    joined = joined.set_index("loan_id").drop(
        columns=["account_id"]
    )

    # more options can be specified also
    with pd.option_context('display.max_columns', None):
        print(joined3)

    return joined


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

    selected_features = forward_selection(joined_dataset.drop(
        columns=["status"]), joined_dataset.iloc[:, -1])

    return joined_dataset, selected_features


def prepare_evaluation_dataset(dataset, trans, disp, account, district, client):
    joined_dataset = join_and_encode_dataset(
        dataset, trans, disp, account, district, client
    )

    return joined_dataset


def up_sampling(train_dataset):
    train_dataset_classes = resample(
        train_dataset[train_dataset["status"] == -1],
        n_samples=len(train_dataset[train_dataset["status"] != -1]))

    return pd.concat(
        [train_dataset_classes, train_dataset[train_dataset["status"] == 1]])


def down_sampling(train_dataset):
    train_dataset_classes = resample(
        train_dataset[train_dataset["status"] == 1],
        n_samples=len(train_dataset[train_dataset["status"] != 1]), replace=False)

    return pd.concat(
        [train_dataset_classes, train_dataset[train_dataset["status"] == -1]])


def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns
    best_features = []
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(
                data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break

    print(best_features)

    return best_features
