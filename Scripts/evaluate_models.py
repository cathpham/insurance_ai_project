import numpy as np
import joblib
from scripts.metrics import calculate_metrics

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    return calculate_metrics(y_test, y_pred)