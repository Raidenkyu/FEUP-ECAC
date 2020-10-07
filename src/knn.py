import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def knn_loan(train_dataset, test_dataset, eval_dataset):
    X_test = test_dataset.iloc[:, 0:5].values
    y_test = test_dataset.iloc[:, 6].values
    X_train = train_dataset.iloc[:, 0:5].values
    y_train = train_dataset.iloc[:, 6].values

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        # print("Predicted: {}     Real: {}".format(pred_i, y_test))
        error.append(np.mean(pred_i != y_test))

    best_k = error.index(min(error))

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred)))

    X_eval = eval_dataset.iloc[:, 0:5].values
    X_eval = scaler.transform(X_eval)

    y_pred = knn.predict(X_eval)
    result = pd.DataFrame({
        'Id': eval_dataset.iloc[:, 0].values,
        'Predicted': y_pred
    })

    print(result)

    return result
