import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from smote import smote_sampling
from plot import plot_auc


def crforest_loan(train_dataset, test_dataset, eval_dataset, selected_features):
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

    #clf = RandomForestClassifier(max_depth=2, random_state=0)
    #clf.fit(X_train, y_train)
    clf = parameters_tuner(X_train, y_train)

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


def random_tuner(X_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_


def search_tuner(X_train, y_train, random_params):
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=random_params,
                               cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    return grid_search


def parameters_tuner(X_train, y_train):
    #random_params = random_tuner(X_train, y_train)
    random_params = {
        'bootstrap': [True],
        'max_depth': [20, 80, 90, 100, 110],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [1600, 1700, 1800, 1900]
    }
    return search_tuner(X_train, y_train, random_params)
