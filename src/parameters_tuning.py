import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


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
