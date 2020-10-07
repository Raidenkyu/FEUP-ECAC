import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def svm_loan(train_dataset, test_dataset, eval_dataset):
    X_test = test_dataset.iloc[:, 0:5].values
    y_test = test_dataset.iloc[:, 6].values
    X_train = train_dataset.iloc[:, 0:5].values
    y_train = train_dataset.iloc[:, 6].values

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(str(confusion_matrix(y_test, y_pred)))
    print(str(classification_report(y_test, y_pred)))

    X_eval = eval_dataset.iloc[:, 0:5].values

    y_pred = svclassifier.predict(X_eval)
    result = pd.DataFrame({
        'Id': eval_dataset.iloc[:, 0].values,
        'Predicted': y_pred
    })

    print(result)

    return result
