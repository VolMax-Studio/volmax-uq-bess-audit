import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

class ConformalPredictor:
    def __init__(self, alpha=0.10, adaptive=True):
        self.alpha = alpha
        self.adaptive = adaptive
        self.q = None
        self.error_model = None
        self.error_scaler = None
        
    def fit(self, X_cal, y_cal, y_pred_cal, X_train=None, y_train=None, y_pred_train=None):
        n_cal = len(X_cal)
        
        if not self.adaptive:
            # Standard Split Conformal Prediction
            residuals = np.abs(y_cal - y_pred_cal)
            # Find the adjusted quantile
            quantile_idx = int(np.ceil((n_cal + 1) * (1 - self.alpha))) / n_cal
            quantile_idx = min(quantile_idx, 1.0) # Ensure it does not exceed 1.0
            self.q = np.quantile(residuals, quantile_idx)
            print(f"Standard Conformal Calibration Complete. Quantile threshold (q): {self.q:.4f}")
        else:
            # Additive Locally Adaptive Conformal Prediction
            train_residuals = np.abs(y_train - y_pred_train)
            
            self.error_model = Ridge(alpha=10.0)
            self.error_model.fit(X_train, train_residuals)
            
            # Predict expected error on calibration set
            g_cal = self.error_model.predict(X_cal)
            
            # Compute non-conformity scores: residual minus predicted error
            cal_residuals = np.abs(y_cal - y_pred_cal)
            nonconf_scores = cal_residuals - g_cal
            
            # Find adjusted quantile
            quantile_idx = int(np.ceil((n_cal + 1) * (1 - self.alpha))) / n_cal
            quantile_idx = min(quantile_idx, 1.0)
            self.q = np.quantile(nonconf_scores, quantile_idx)
            print(f"Additive Adaptive Conformal Calibration Complete. Quantile threshold (q): {self.q:.4f}")
            
    def predict_intervals(self, X, y_pred):
        if not self.adaptive:
            # Uniform width intervals
            lower_bound = y_pred - self.q
            upper_bound = y_pred + self.q
        else:
            # Additive, locally adaptive intervals
            g = self.error_model.predict(X)
            g = np.maximum(g, 0.0) # Ensure predicted difficulty is non-negative
            lower_bound = y_pred - g - self.q
            upper_bound = y_pred + g + self.q
            
        return lower_bound, upper_bound

def compute_uq_metrics(y_true, lower, upper, alpha=0.10):
    # Coverage probability
    inside = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(inside)
    
    # Width metrics (converted back to linear scale of cycles for physical meaning)
    # y is in log10(cycles)
    widths = 10**upper - 10**lower
    mpiw = np.mean(widths)
    sharpness = np.std(widths)
    
    # Winkler Score (evaluated in linear cycles scale)
    y_true_lin = 10**y_true
    lower_lin = 10**lower
    upper_lin = 10**upper
    
    winkler_scores = []
    for y_val, l_val, u_val in zip(y_true_lin, lower_lin, upper_lin):
        width = u_val - l_val
        if y_val < l_val:
            penalty = (2.0 / alpha) * (l_val - y_val)
        elif y_val > u_val:
            penalty = (2.0 / alpha) * (y_val - u_val)
        else:
            penalty = 0.0
        winkler_scores.append(width + penalty)
        
    mean_winkler = np.mean(winkler_scores)
    
    return {
        "picp": picp,
        "mpiw": mpiw,
        "sharpness": sharpness,
        "winkler": mean_winkler
    }
