#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

# Re-use loaders from model_audit
from model_audit import load_data, prepare_features

def main():
    df = load_data()
    X, y, partitions, cell_ids = prepare_features(df)
    
    # Scale exactly like model_audit
    id_mask = (partitions == "train") | (partitions == "primary")
    ood_mask = (partitions == "secondary")
    
    X_id, y_id, cell_ids_id = X[id_mask], y[id_mask], cell_ids[id_mask]
    X_ood, y_ood, cell_ids_ood = X[ood_mask], y[ood_mask], cell_ids[ood_mask]
    
    # 80/20 train/cal split
    np.random.seed(42)
    indices = np.arange(len(X_id))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X_id))
    train_indices = indices[:split_idx]
    cal_indices = indices[split_idx:]
    
    X_train, y_train = X_id.iloc[train_indices], y_id[train_indices]
    X_cal, y_cal = X_id.iloc[cal_indices], y_id[cal_indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_ood_scaled = scaler.transform(X_ood)
    
    # Fit regularized covariance (Ledoit-Wolf) on X_train_scaled to prevent ill-conditioned matrix inversion
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf()
    cov.fit(X_train_scaled)
    
    # Compute Mahalanobis
    m_train = cov.mahalanobis(X_train_scaled)
    m_cal = cov.mahalanobis(X_cal_scaled)
    m_ood = cov.mahalanobis(X_ood_scaled)
    
    print("=== MAHALANOBIS DISTANCE STATS (d^2) WITH LEDOIT-WOLF ===")
    print(f"Train: min={np.min(m_train):.2f}, mean={np.mean(m_train):.2f}, max={np.max(m_train):.2f}")
    print(f"Cal:   min={np.min(m_cal):.2f}, mean={np.mean(m_cal):.2f}, max={np.max(m_cal):.2f}")
    print(f"OOD:   min={np.min(m_ood):.2f}, mean={np.mean(m_ood):.2f}, max={np.max(m_ood):.2f}")
    
    # Find outliers in OOD
    threshold = 15.0
    outlier_idx = np.where(m_ood > threshold)[0]
    print(f"\nFound {len(outlier_idx)} outliers in OOD set with d^2 > {threshold}:")
    
    for idx in outlier_idx:
        cell_id = cell_ids_ood[idx]
        dist = m_ood[idx]
        raw_feat = X_ood.iloc[idx].to_dict()
        scaled_feat = X_ood_scaled[idx]
        cycle_life = 10**y_ood[idx]
        print(f"\nCell ID: {cell_id} | Mahalanobis d^2: {dist:.2f} | Cycle Life: {cycle_life:.1f}")
        print(f"  Raw Features: {raw_feat}")
        print(f"  Scaled Features: {scaled_feat.tolist()}")
        
    # Let's check Cal outliers too
    outlier_cal_idx = np.where(m_cal > threshold)[0]
    print(f"\nFound {len(outlier_cal_idx)} outliers in Cal set with d^2 > {threshold}:")
    for idx in outlier_cal_idx:
        cell_id = cell_ids_id[cal_indices[idx]]
        dist = m_cal[idx]
        cycle_life = 10**y_cal[idx]
        print(f"  Cell ID: {cell_id} | Mahalanobis d^2: {dist:.2f} | Cycle Life: {cycle_life:.1f}")

if __name__ == "__main__":
    main()
