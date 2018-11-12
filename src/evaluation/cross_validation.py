from sklearn.model_selection import GridSearchCV


def cross_validate(new_pipeline_funct, hyperparams_grid, X_train, y_train, name, n_folds=3, verbose=True):
    """
    Return the best hyperparameters.
    """
    print("Cross-Validation Grid Search for: '{}'...".format(name))

    pipeline = new_pipeline_funct()

    grid_search = GridSearchCV(
        pipeline, hyperparams_grid, iid=True, cv=n_folds, return_train_score=False, verbose=False, scoring="accuracy",
        n_jobs=4, pre_dispatch=8
    )
    grid_search.fit(X_train, y_train)

    if verbose:
        print("Best hyperparameters for '{}' ({}-folds cross validation accuracy score={}):".format(
            name, n_folds, grid_search.best_score_))
    best_params = grid_search.best_params_
    return best_params


def get_best_classifier_from_cross_validation(
        hyperparams_grid, new_pipeline_funct, X_train, y_train, name, verbose=True):
    """
    Return a new `new_pipeline_funct` trained on the data with the best hyperparameters.
    """
    best_params = cross_validate(new_pipeline_funct, hyperparams_grid, X_train, y_train, name, verbose=True)
    if verbose:
        print(best_params)
        print("")

    best_pipeline = new_pipeline_funct()
    best_pipeline.set_params(
        **best_params
    )

    best_pipeline.fit(X_train, y_train)
    return best_pipeline
