# scripts/train_models.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, r2, mape

def train_and_evaluate_models(X, y_log, y_actual):
    models = {
        'gb': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55),
        'xgb': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55),
        'lgb': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55, verbose=-1)
    }

    predictions = {}
    r2_scores = {}

    for name, model in models.items():
        model.fit(X, y_log)
        log_preds = cross_val_predict(model, X, y_log, cv=5)
        preds = np.exp(log_preds)
        rmse, mae, r2, mape = calculate_metrics(y_actual, preds)
        print(f"Results for {name.upper()}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")

        predictions[name] = preds
        r2_scores[name] = r2
        joblib.dump(model, f'{name}_model.pkl')

    # Calculate ensemble
    total_r2 = sum(r2_scores.values())
    weights = {k: v / total_r2 for k, v in r2_scores.items()}

    ensemble_pred = sum(weights[name] * predictions[name] for name in predictions)
    rmse, mae, r2, mape = calculate_metrics(y_actual, ensemble_pred)
    print(f"Results for Ensemble: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")

    joblib.dump(weights, 'ensemble_weights.pkl')
    return weights
