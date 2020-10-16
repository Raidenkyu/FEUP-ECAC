import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import resample


def crforest_loan(train_dataset, test_dataset, eval_dataset):
    train_dataset_classes = resample(
        train_dataset[train_dataset["status"] == -1],
        n_samples=len(train_dataset[train_dataset["status"] != -1]))

    train_dataset = pd.concat(
        [train_dataset_classes, train_dataset[train_dataset["status"] == 1]])

    X_test = test_dataset.drop(columns=["status"]).values
    y_test = test_dataset.iloc[:, -1].values
    X_train = train_dataset.drop(columns=["status"]).values
    y_train = train_dataset.iloc[:, -1].values

    scaler = StandardScaler()

    # X_train, y_train = .fit_resample(X_train, y_train)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred, zero_division=0)))
    print(f"AUC: {roc_auc_score(y_test, y_pred)}")

    X_eval = eval_dataset.drop(columns=["status"]).values
    X_eval = scaler.transform(X_eval)

    id_array = map(lambda x: int(x), eval_dataset.index.values)
    y_pred = map(lambda x: int(x), clf.predict(X_eval))

    result = pd.DataFrame({
        'Id': id_array,
        'Predicted': y_pred
    })

    return result
