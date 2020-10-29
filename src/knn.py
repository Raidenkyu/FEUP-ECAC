import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn import metrics

from smote import smote_sampling
from plot import plot_auc


def knn_loan(train_dataset, test_dataset, eval_dataset, selected_features):
    X_test = test_dataset.drop(columns=["status"])
    y_test = test_dataset.iloc[:, -1]
    X_train = train_dataset.drop(columns=["status"])
    y_train = train_dataset.iloc[:, -1]

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_test = X_test.values
    y_test = y_test.values
    X_train = X_train.values
    y_train = y_train.values

    X_train, y_train = smote_sampling(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    auc = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        auc.append(roc_auc_score(y_test, pred_i))

    best_k = auc.index(max(auc)) + 1

    fpr, tpr, _thresholds = metrics.roc_curve(y_test, pred_i)
    print(f"AUC: {metrics.auc(fpr, tpr)}")

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred, zero_division=0)))
    print(f"AUC: {roc_auc_score(y_test, y_pred)}")
    print(f"Best K: {best_k}")

    plot_auc(knn, X_test, y_test, "knn")

    X_eval = eval_dataset.drop(columns=["status"])
    X_eval = X_eval[selected_features].values
    X_eval = scaler.transform(X_eval)

    id_array = map(lambda x: int(x), eval_dataset.index.values)
    y_pred = map(lambda x: int(x), knn.predict(X_eval))

    result = pd.DataFrame({
        'Id': id_array,
        'Predicted': y_pred
    })

    return result
