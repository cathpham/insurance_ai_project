import numpy as np
import joblib

def predict_ensemble(X_input):
    gb = joblib.load('gb_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    lgb_model = joblib.load('lgb_model.pkl')
    weights = joblib.load('ensemble_weights.pkl')

    preds = {
        'gb': np.exp(gb.predict(X_input)),
        'xgb': np.exp(xgb_model.predict(X_input)),
        'lgb': np.exp(lgb_model.predict(X_input))
    }

    ensemble_pred = sum(weights[name] * preds[name] for name in preds)
    return ensemble_pred