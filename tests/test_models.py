import pytest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from pathlib import Path
from models import get_all_models


def test_get_all_models_returns_list():
    models = get_all_models()
    assert isinstance(models, list)

def test_get_all_models_correct_length():
    models = get_all_models()
    assert len(models) == 8  # Now 8 models after removing CatBoost

def test_get_all_models_structure():
    models = get_all_models()
    for model_tuple in models:
        assert len(model_tuple) == 3
        assert isinstance(model_tuple[0], str)
        assert isinstance(model_tuple[2], dict)

def test_get_all_models_names():
    expected_names = ['linear', 'ridge', 'lasso', 'knn', 'kernel-ridge', 
                      'gradient-boosting', 'svr', 'random-forest']
    models = get_all_models()
    actual_names = [model[0] for model in models]
    assert set(actual_names) == set(expected_names)

def test_get_all_models_instances():
    expected_instances = [LinearRegression, Ridge, Lasso, KNeighborsRegressor, 
                          KernelRidge, GradientBoostingRegressor, SVR, 
                          RandomForestRegressor]
    models = get_all_models()
    for model, expected_class in zip(models, expected_instances):
        assert isinstance(model[1], expected_class)

def test_linear_regression_params():
    models = get_all_models()
    linear_params = next(model[2] for model in models if model[0] == 'linear')
    assert linear_params == {}  # Linear Regression should have no hyperparameters

def test_random_forest_params():
    models = get_all_models()
    rf_params = next(model[2] for model in models if model[0] == 'random-forest')
    assert 'n_estimators' in rf_params
    assert 'max_depth' in rf_params
    assert 'min_samples_split' in rf_params
    assert 'min_samples_leaf' in rf_params
    assert 'max_features' in rf_params

if __name__ == "__main__":
    pytest.main()