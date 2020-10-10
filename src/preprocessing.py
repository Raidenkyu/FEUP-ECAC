import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


def join_and_encode_dataset(dataset, disp, account, district, client):
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


def prepare_development_dataset(dataset, disp, account, district, client):
    joined_dataset = join_and_encode_dataset(
        dataset, disp, account, district, client
    )

    #joined_dataset = remove_outliers(joined_dataset)

    return joined_dataset


def prepare_evaluation_dataset(dataset, disp, account, district, client):
    joined_dataset = join_and_encode_dataset(
        dataset, disp, account, district, client
    )

    return joined_dataset
