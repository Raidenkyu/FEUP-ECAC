import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from preprocessing import down_sampling, forward_selection
from smote import smote_sampling
from plot import plot_auc


def crforest_loan(train_dataset, test_dataset, eval_dataset, selected_features):
    #train_dataset = down_sampling(train_dataset)

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

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    current_AUC = roc_auc_score(y_test, y_pred)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred, zero_division=0)))
    print(f"Current AUC: {current_AUC}")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    plot_auc(clf, X_test, y_test, "random_forest")

    X_eval = eval_dataset.drop(columns=["status"])
    X_eval = X_eval[selected_features].values
    X_eval = scaler.transform(X_eval)

    id_array = map(lambda x: int(x), eval_dataset.index.values)
    y_pred = map(lambda x: int(x), clf.predict(X_eval))

    result = pd.DataFrame({
        'Id': id_array,
        'Predicted': y_pred
    })

    return result
