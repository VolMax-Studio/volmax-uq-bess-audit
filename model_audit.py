import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def load_data(csv_path="../Battery_Health_Portfolio/severson_features.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Exclude standard bad cells
    exclude_cells = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"}
    df = df[~df["cell_id"].isin(exclude_cells)].copy()
    
    return df

def prepare_features(df):
    # Select features corresponding to the multi-feature model in Severson et al.
    feature_cols = [
        "log_var_dq",
        "log_min_dq",
        "cap_diff_100_2",
        "ir_diff_100_2",
        "temp_slope_2_100"
    ]
    
    X = df[feature_cols].copy()
    y = np.log10(df["cycle_life"].values)
    
    return X, y, df["partition"].values, df["cell_id"].values

def train_baseline_model():
    df = load_data()
    X, y, partitions, cell_ids = prepare_features(df)
    
    # Identify In-Distribution (ID) vs Out-of-Distribution (OOD)
    # train and primary batches are In-Distribution. secondary batch is OOD.
    id_mask = (partitions == "train") | (partitions == "primary")
    ood_mask = (partitions == "secondary")
    
    X_id, y_id, cell_ids_id = X[id_mask], y[id_mask], cell_ids[id_mask]
    X_ood, y_ood, cell_ids_ood = X[ood_mask], y[ood_mask], cell_ids[ood_mask]
    
    # Split In-Distribution into Train (80%) and Calibration (20%) for Conformal Prediction
    np.random.seed(42)
    indices = np.arange(len(X_id))
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(X_id))
    train_indices = indices[:split_idx]
    cal_indices = indices[split_idx:]
    
    X_train, y_train = X_id.iloc[train_indices], y_id[train_indices]
    X_cal, y_cal = X_id.iloc[cal_indices], y_id[cal_indices]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_ood_scaled = scaler.transform(X_ood)
    
    # Fit point-prediction model (ElasticNet)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate point predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_cal = model.predict(X_cal_scaled)
    y_pred_ood = model.predict(X_ood_scaled)
    
    print("--- Baseline Model Point Evaluation (Cycle Life) ---")
    print(f"Train RMSE: {mean_squared_error(10**y_train, 10**y_pred_train, squared=False):.2f} cycles")
    print(f"Cal RMSE:   {mean_squared_error(10**y_cal, 10**y_pred_cal, squared=False):.2f} cycles")
    print(f"OOD RMSE:   {mean_squared_error(10**y_ood, 10**y_pred_ood, squared=False):.2f} cycles (Secondary)")
    
    return {
        "model": model,
        "scaler": scaler,
        "train": (X_train_scaled, y_train),
        "cal": (X_cal_scaled, y_cal),
        "ood": (X_ood_scaled, y_ood),
        "cell_ids_cal": cell_ids_id[cal_indices],
        "cell_ids_ood": cell_ids_ood
    }

if __name__ == "__main__":
    train_baseline_model()
