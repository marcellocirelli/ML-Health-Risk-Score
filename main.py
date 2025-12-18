import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Loads the dataset
data = pd.read_csv('dataset.csv')

# Defines the target variable and features
target_variable = 'healthRiskScore'

features = data.drop(columns = [target_variable]).columns
X = data[features]
y = data[target_variable]

# Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Initializes CV object
cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

# Parameter distribution for Randomized Forest
rfr_param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [None, 9],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}

# Parameter distribution for Gradient Boosting
gbr_param_dist = {
    "n_estimators": [200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "subsample": [0.9, 1.0],
}

# Initializes and trains the models
dtr = DecisionTreeRegressor(random_state = 42)
rfr = RandomForestRegressor(random_state = 42)
gbr = GradientBoostingRegressor(random_state = 42)
dtr.fit(X_train, y_train)
rfr.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Optimizes using RandomizedSearchCV
def tune_randomized(estimator, param_dist, X_train, y_train, cv):
    search = RandomizedSearchCV(
        estimator = estimator,
        param_distributions = param_dist,
        scoring = "neg_root_mean_squared_error",
        cv = cv,
        n_jobs = -1,
        random_state = 42,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best

# Optimizes using GridSearchCV
def tune_grid(estimator, param_grid, X_train, y_train, cv):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best

#  RFR optimized with RSCV
rfr_rscv = tune_randomized(
    RandomForestRegressor(random_state=42),
    rfr_param_dist,
    X_train, y_train,
    cv
)
# GBR optimized with RSCV
gbr_rscv = tune_randomized(
    GradientBoostingRegressor(random_state=42),
    gbr_param_dist,
    X_train, y_train,
    cv
)

# RFR optimized with GSCV
rfr_gscv = tune_grid(
    RandomForestRegressor(random_state=42),
    rfr_param_dist,
    X_train, y_train,
    cv
)

# GBR optimized with GSCV
gbr_gscv = tune_grid(
    GradientBoostingRegressor(random_state=42),
    gbr_param_dist,
    X_train, y_train,
    cv
)

# Fits model for Regularization
def fit_regularized(model, X_tr, Y_tr, **param_updates):
    m = clone(model)
    if param_updates:
        m.set_params(**param_updates)
    m.fit(X_tr, Y_tr)
    return m

# Initialization of regularized models
rfr_rscv_depth = fit_regularized(rfr_rscv, X_train, y_train, max_depth = 10)
rfr_rscv_leaf = fit_regularized(rfr_rscv, X_train, y_train, min_samples_leaf = 2)
rfr_gscv_depth = fit_regularized(rfr_gscv, X_train, y_train, max_depth = 10)
rfr_gscv_leaf = fit_regularized(rfr_gscv, X_train, y_train, min_samples_leaf = 2)
gbr_rscv_depth = fit_regularized(gbr_rscv, X_train, y_train, max_depth = 2)
gbr_rscv_leaf = fit_regularized(gbr_rscv, X_train, y_train, min_samples_leaf = 3)
gbr_gscv_depth = fit_regularized(gbr_gscv, X_train, y_train, max_depth = 2)
gbr_gscv_leaf = fit_regularized(gbr_gscv, X_train, y_train, min_samples_leaf = 3)

# Calculates and reports metrics
def report_metrics(model, X_t, y_t, name: str):
    y_pred = model.predict(X_t)
    print(name)
    r2 = r2_score(y_t, y_pred)
    mse = mean_squared_error(y_t, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f} | R2 score: {r2:.4f}\n")

print("-- Baseline Metrics --")
report_metrics(dtr, X_test, y_test, "Decision Tree Baseline")
report_metrics(rfr, X_test, y_test, "Random Forest Baseline")
report_metrics(gbr, X_test, y_test, "Gradient Boosting Baseline")
print("-- Optimized Metrics --")
report_metrics(rfr_rscv, X_test, y_test, "Random Forest (RandomizedSearchCV)")
report_metrics(rfr_gscv, X_test, y_test, "Random Forest (GridSearchCV)")
report_metrics(gbr_rscv, X_test, y_test, "Gradient Boosting (RandomizedSearchCV)")
report_metrics(gbr_gscv, X_test, y_test, "Gradient Boosting (GridSearchCV)")
print("-- Optimized & Regularized Metrics --")
report_metrics(rfr_rscv_depth, X_test, y_test, "Random Forest (RandomizedSearchCV, depth-regularized)")
report_metrics(rfr_rscv_leaf, X_test, y_test, "Random Forest (RandomizedSearchCV, leaf-regularized)")
report_metrics(rfr_gscv_depth, X_test, y_test, "Random Forest (GridSearchCV, depth-regularized)")
report_metrics(rfr_gscv_leaf, X_test, y_test, "Random Forest (GridSearchCV, leaf-regularized)")
report_metrics(gbr_rscv_depth, X_test, y_test, "Gradient Boosting (RandomizedSearchCV, depth-regularized)")
report_metrics(gbr_rscv_leaf, X_test, y_test, "Gradient Boosting (RandomizedSearchCV, leaf-regularized)")
report_metrics(gbr_gscv_depth, X_test, y_test, "Gradient Boosting (GridSearchCV, depth-regularized)")
report_metrics(gbr_gscv_leaf, X_test, y_test, "Gradient Boosting (GridSearchCV, leaf-regularized)")