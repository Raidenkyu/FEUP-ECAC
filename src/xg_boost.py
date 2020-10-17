import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score


def xg_boost(train_dataset, test_dataset, eval_dataset):
    X_test = test_dataset.drop(columns=["status"]).values
    y_test = test_dataset.iloc[:, -1].values
    X_train = train_dataset.drop(columns=["status"]).values
    y_train = train_dataset.iloc[:, -1].values

    scaler = StandardScaler()

    #X_train, y_train = .fit_resample(X_train, y_train)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print(model)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred, zero_division=0)))
    print(f"AUC: {roc_auc_score(y_test, y_pred)}")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    X_eval = eval_dataset.drop(columns=["status"]).values
    X_eval = scaler.transform(X_eval)

    id_array = map(lambda x: int(x), eval_dataset.index.values)
    y_pred = map(lambda x: int(x), model.predict(X_eval))

    result = pd.DataFrame({
        'Id': id_array,
        'Predicted': y_pred
    })

    return result