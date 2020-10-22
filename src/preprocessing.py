import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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


def crime95_f(data):

   
    vara = int(data["no. of commited crimes '95 "].min())
    d = pd.DataFrame({
        'account_id': data['account_id'],
        'crime_rate_95': vara / 
        data["no. of inhabitants"]
    })
    
    return d

def crime96_f(data):

   
    vara = int(data["no. of commited crimes '96 "].min())
    d = pd.DataFrame({
        'account_id': data['account_id'],
        'crime_rate_96': vara / 
        data["no. of inhabitants"]
    })
    
    return d


def join_client(disp, client, district):

    joined_client = district.set_index("code ", drop=False).join(
        client.set_index("district_id", drop=False))

    joined_client = disp.set_index("client_id", drop=False).join(
        joined_client.set_index("client_id"))

    joined_client = joined_client.replace('?', np.nan)
    joined_client = joined_client.fillna(joined_client.mean())

    crime_95 = joined_client[[
        "account_id", "no. of commited crimes '95 ", "no. of inhabitants"]].groupby("account_id").apply(crime95_f)

    crime_96 = joined_client[[
        "account_id", "no. of commited crimes '96 ", "no. of inhabitants"]].groupby("account_id").apply(crime96_f)


    joined_client = joined_client[["account_id", "ratio of urban inhabitants ", "region", "no. of inhabitants",
                                   "no. of cities ", "average salary ", "no. of enterpreneurs per 1000 inhabitants "
                                   ,"unemploymant rate '95 ",
                       "unemploymant rate '96 "]].groupby("account_id").min()  # ADICIONAR FATORES EXTERNOS AQUI

    joined_client = joined_client.replace(['Prague', 'central Bohemia', 'east Bohemia',
                                           'south Bohemia', 'north Bohemia', 'west Bohemia', 'north Moravia', 'south Moravia'],
                                          [0, 1, 2, 3, 4, 5, 6, 7])

    joined_client = joined_client.join(crime_95, rsuffix="_95")
    joined_client = joined_client.join(crime_96, rsuffix="_96")

    c = joined_client.select_dtypes(np.floating).columns
    joined_client[c] = imp.fit_transform(joined_client[c])

    joined_client = joined_client.astype('float64')
    print(joined_client)

    return joined_client


def percentage_credit(series):
    return series.isin(['credit']).sum(axis=0)/len(series)


def join_trans(dataset, trans):
    trans_min_balance = trans[[
        "account_id", "balance"]].groupby("account_id").min()

    trans_average_balance = trans[[
        "account_id", "balance"]].groupby("account_id").mean()

    trans_average_amount = trans[[
        "account_id", "amount"]].groupby("account_id").mean()

    trans_average_type = trans[[
        "account_id", "type"]].groupby("account_id").agg({'type': percentage_credit})

    trans_count = trans[[
        "account_id"]].groupby("account_id").size().to_frame(name='trans_count')

    dataset['date'] = dataset['date'] // 10000

    joined_trans = trans_average_amount.join(trans_min_balance)
    joined_trans = joined_trans.join(trans_average_type)
    joined_trans = joined_trans.join(trans_count)

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
                       "balance_account_average", "trans_count", "type",
                       "ratio of urban inhabitants ", "region", "no. of inhabitants",
                       "no. of cities ", "average salary ", "unemploymant rate '95 ",
                       "unemploymant rate '96 ", "no. of enterpreneurs per 1000 inhabitants ",
                       "crime_rate_95", "crime_rate_96", "status"])

    joined = joined.set_index("loan_id").drop(
        columns=["account_id"]
    )

    # more options can be specified also
    with pd.option_context('display.max_columns', None):
        print(joined)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    c = joined.select_dtypes(np.floating).columns
    joined[c] = imp.fit_transform(joined[c])
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
