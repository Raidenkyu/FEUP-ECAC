import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


def prepare_dataset(train_dataset, disp, account, district, client):
    joined_client = disp.set_index("client_id", drop=False).join(
        client.set_index("client_id"), rsuffix='_other')

    joined_account = joined_client.set_index("account_id", drop=False).join(
        account.set_index("account_id"), rsuffix='_other')

    joined_district = joined_account.set_index("district_id", drop=False).join(
        district.set_index("code "), rsuffix='_other')

    joined = joined_district.set_index("account_id", drop=False).join(
        train_dataset.set_index("account_id", drop=False), rsuffix='_other')

    joined = joined.drop(
        columns=["disp_id", "client_id", "account_id", "district_id",
                 "district_id_other", "account_id_other", "date_other"]
    ).dropna(subset=["loan_id"]).set_index("loan_id")

    joined = joined[~joined.index.duplicated(keep='first')]

    enc = OrdinalEncoder()
    joined.iloc[:, 0:21] = enc.fit_transform(joined.iloc[:, 0:21])

    for col in joined.columns:
        print(col)

    print(joined)
    return joined
