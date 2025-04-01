# SEA LEVEL PREDICTION

# DEVELOPED BY: 
# RUKSANA SALIM

# IMPORTING MODULES

import numpy as np
import pandas as pd
import datetime as dt
import optuna
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# Assume dataset format
data =  pd.read_csv('dummy_data.csv')# exclusive for each tide gauge station


# Split features and target
X = data.drop(columns=["Time","Sl_cor", "Station"])
y = data["Sl_cor"].values.reshape(-1, 1)

# Scale features using StandardScaler
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

# Define time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize stacked predictions
n_samples = X_scaled.shape[0]
n_folds = tscv.get_n_splits()
n_base_learners = 4
stacked_predictions = np.zeros((n_samples, n_base_learners))

stations = data['Station'].unique()
station_predictions_for_test = {}

def objective(trial):
    # Define hyperparameters for each model
    elastic_alpha = trial.suggest_float("elastic_alpha", 0.01, 1.0)
    elastic_l1_ratio = trial.suggest_float("elastic_l1_ratio", 0.01, 1.0)
    
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
    rf_max_depth = trial.suggest_int("rf_max_depth", 3, 20)
    
    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 15)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
    
    cat_n_estimators = trial.suggest_int("cat_n_estimators", 50, 300)
    cat_depth = trial.suggest_int("cat_depth", 3, 15)
    
    # Initialize base models
    elastic_net = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_l1_ratio)
    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=0)
    xgb = XGBRegressor(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, random_state=0)
    cat = CatBoostRegressor(n_estimators=cat_n_estimators, depth=cat_depth, verbose=0, random_state=0)
    
    meta_features = []
    y_actuals = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train base models
        elastic_net.fit(X_train, y_train.ravel())
        rf.fit(X_train, y_train.ravel())
        xgb.fit(X_train, y_train.ravel())
        cat.fit(X_train, y_train.ravel())
        
        # Get predictions
        elastic_pred = elastic_net.predict(X_val)
        rf_pred = rf.predict(X_val)
        xgb_pred = xgb.predict(X_val)
        cat_pred = cat.predict(X_val)
        
        # Store validation predictions for stacked model training
        stacked_predictions[val_idx, 0] = elastic_pred
        stacked_predictions[val_idx, 1] = rf_pred
        stacked_predictions[val_idx, 2] = xgb_pred
        stacked_predictions[val_idx, 3] = cat_pred
        
        # Store test predictions per station
        for station in stations:
            station_data = data[data['Station'] == station]
            X_test_station = station_data.drop(columns=['Station', 'sea_level']).values
            
            if station not in station_predictions_for_test:
                station_predictions_for_test[station] = np.zeros((len(X_test_station), n_folds * n_base_learners))
            
            X_test_station_scaled = scaler_x.transform(X_test_station)
            
            station_predictions_for_test[station][:, fold * n_base_learners] = elastic_net.predict(X_test_station_scaled)
            station_predictions_for_test[station][:, fold * n_base_learners + 1] = rf.predict(X_test_station_scaled)
            station_predictions_for_test[station][:, fold * n_base_learners + 2] = xgb.predict(X_test_station_scaled)
            station_predictions_for_test[station][:, fold * n_base_learners + 3] = cat.predict(X_test_station_scaled)
        
    # Train meta-learner
    meta_features = np.array(meta_features)
    y_actuals = np.array(y_actuals)
    meta_learner = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_l1_ratio)
    meta_learner.fit(meta_features, y_actuals.ravel())
    y_meta_pred = meta_learner.predict(meta_features)
    
    # Evaluate R² score
    return r2_score(y_actuals, y_meta_pred)

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best parameters
best_params = study.best_params
print("Best parameters found:", best_params)

# Save models
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(elastic_net, "elastic_net.pkl")
joblib.dump(rf, "random_forest.pkl")
joblib.dump(xgb, "xgboost.pkl")
joblib.dump(cat, "catboost.pkl")
joblib.dump(meta_learner, "meta_learner.pkl")

print("Models and scaler saved successfully.")

