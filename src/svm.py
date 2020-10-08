import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def svm_loan(train_dataset, test_dataset, eval_dataset):
    X_test = test_dataset.iloc[:, 0:21].values
    y_test = test_dataset.iloc[:, 22].values
    X_train = train_dataset.iloc[:, 0:21].values
    y_train = train_dataset.iloc[:, 22].values

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred)))

    X_eval = eval_dataset.iloc[:, 0:21].values
    X_eval = scaler.transform(X_eval)

    id_array = map(lambda x: int(x), eval_dataset.index.values)
    y_pred = map(lambda x: int(x), svclassifier.predict(X_eval))

    y_pred = svclassifier.predict(X_eval)
    result = pd.DataFrame({
        'Id': id_array,
        'Predicted': y_pred
    })

    print(result)

    return result
