from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR


def get_all_models():
    """
    Returns a list of regression models with their corresponding hyperparameter grids for grid search.

    This function provides a comprehensive set of regression models, including:
    1. Linear Regression (baseline)
    2. Ridge Regression
    3. Lasso Regression
    4. K-Nearest Neighbors Regressor
    5. Kernel Ridge Regression
    6. Gradient Boosting Regressor
    7. Support Vector Regression (SVR)
    8. Random Forest Regressor

    Each model is accompanied by a dictionary of hyperparameters to be used in a grid search
    for model tuning. The hyperparameter ranges are chosen to cover a spectrum from simple
    to complex models, allowing for effective model selection and comparison.

    Returns:
    list of tuples: Each tuple contains (model_name, model_instance, hyperparameter_grid)
        - model_name (str): A string identifier for the model
        - model_instance (object): An instance of the sklearn model
        - hyperparameter_grid (dict): A dictionary of hyperparameters for grid search

    Note:
    - Ensure all required libraries (sklearn) are installed before using this function.
    - The hyperparameter grids can be adjusted based on specific dataset characteristics or computational constraints.
    """
    linear_params = {}  # Linear Regression doesn't have hyperparameters to tune

    ridge_params = {
        'alpha': [0.1, 1, 10, 100]
    }

    lasso_params = {
        'alpha': [0.1, 1, 10, 100]
    }

    knn_params = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    kernel_ridge_params = {
        'alpha': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': [0.1, 1, 10]
    }

    gradient_boosting_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    }

    svr_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.1, 0.5]
    }

    random_forest_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    return [
        ('linear', LinearRegression(), linear_params),
        ('ridge', Ridge(), ridge_params),
        ('lasso', Lasso(), lasso_params),
        ('knn', KNeighborsRegressor(), knn_params),
        ('kernel-ridge', KernelRidge(), kernel_ridge_params),
        ('gradient-boosting', GradientBoostingRegressor(), gradient_boosting_params),
        ('svr', SVR(), svr_params),
        ('random-forest', RandomForestRegressor(), random_forest_params),
    ]