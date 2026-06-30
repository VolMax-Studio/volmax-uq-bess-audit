import numpy as np
import pandas as pd
from model_audit import train_baseline_model
from conformal_wrapper import ConformalPredictor, compute_uq_metrics
import json
import os

def run_stress_test():
    # 1. Train baseline and prepare sets
    data = train_baseline_model()
    
    model = data["model"]
    scaler = data["scaler"]
    X_train, y_train = data["train"]
    X_cal, y_cal = data["cal"]
    X_ood, y_ood = data["ood"]
    
    y_pred_train = model.predict(X_train)
    y_pred_cal = model.predict(X_cal)
    y_pred_ood = model.predict(X_ood)
    
    # 2. Fit predictors
    predictor_std = ConformalPredictor(alpha=0.10, adaptive=False)
    predictor_std.fit(X_cal, y_cal, y_pred_cal)
    
    predictor_ada = ConformalPredictor(alpha=0.10, adaptive=True)
    predictor_ada.fit(X_cal, y_cal, y_pred_cal, X_train, y_train, y_pred_train)
    
    # 3. Predict Intervals
    lower_cal_std, upper_cal_std = predictor_std.predict_intervals(X_cal, y_pred_cal)
    lower_cal_ada, upper_cal_ada = predictor_ada.predict_intervals(X_cal, y_pred_cal)
    
    lower_ood_std, upper_ood_std = predictor_std.predict_intervals(X_ood, y_pred_ood)
    lower_ood_ada, upper_ood_ada = predictor_ada.predict_intervals(X_ood, y_pred_ood)
    
    # 4. Evaluate metrics
    metrics_cal_std = compute_uq_metrics(y_cal, lower_cal_std, upper_cal_std)
    metrics_cal_ada = compute_uq_metrics(y_cal, lower_cal_ada, upper_cal_ada)
    
    metrics_ood_std = compute_uq_metrics(y_ood, lower_ood_std, upper_ood_std)
    metrics_ood_ada = compute_uq_metrics(y_ood, lower_ood_ada, upper_ood_ada)
    
    # 5. Fit Hybrid OOD Detector
    from ood_detector import HybridOODDetector
    detector = HybridOODDetector(contamination=0.05)
    detector.fit(X_train)
    
    # Get scores and predictions
    scores_cal = detector.get_scores(X_cal)
    scores_ood = detector.get_scores(X_ood)
    
    pred_cal = detector.predict(X_cal)
    pred_ood = detector.predict(X_ood)
    
    print("\n=== UNIFORM COVERAGE (STANDARD CONFORMAL) ===")
    print(f"Calibration Split Coverage (PICP): {metrics_cal_std['picp']*100:.2f}% (Target: 90%)")
    print(f"Calibration Split MPIW:           {metrics_cal_std['mpiw']:.2f} cycles")
    print(f"OOD (Secondary) Coverage (PICP):  {metrics_ood_std['picp']*100:.2f}%")
    print(f"OOD (Secondary) MPIW:             {metrics_ood_std['mpiw']:.2f} cycles")
    print(f"OOD (Secondary) Winkler Score:    {metrics_ood_std['winkler']:.2f}")
    
    print("\n=== LOCALLY ADAPTIVE COVERAGE ===")
    print(f"Calibration Split Coverage (PICP): {metrics_cal_ada['picp']*100:.2f}% (Target: 90%)")
    print(f"Calibration Split MPIW:           {metrics_cal_ada['mpiw']:.2f} cycles")
    print(f"OOD (Secondary) Coverage (PICP):  {metrics_ood_ada['picp']*100:.2f}%")
    print(f"OOD (Secondary) MPIW:             {metrics_ood_ada['mpiw']:.2f} cycles")
    print(f"OOD (Secondary) Winkler Score:    {metrics_ood_ada['winkler']:.2f}")
    
    print("\n=== HYBRID OOD DETECTION ===")
    print(f"In-Distribution Cal OOD Flag Rate: {np.mean(pred_cal)*100:.2f}%")
    print(f"Out-of-Distribution Test Flag Rate: {np.mean(pred_ood)*100:.2f}% (Target: High detection)")
    
    # Save results to JSON
    results = {
        "standard_cal": metrics_cal_std,
        "standard_ood": metrics_ood_std,
        "adaptive_cal": metrics_cal_ada,
        "adaptive_ood": metrics_ood_ada,
        "cal_y_true": (10**y_cal).tolist(),
        "cal_y_pred": (10**y_pred_cal).tolist(),
        "cal_lower_std": (10**lower_cal_std).tolist(),
        "cal_upper_std": (10**upper_cal_std).tolist(),
        "cal_lower_ada": (10**lower_cal_ada).tolist(),
        "cal_upper_ada": (10**upper_cal_ada).tolist(),
        "ood_y_true": (10**y_ood).tolist(),
        "ood_y_pred": (10**y_pred_ood).tolist(),
        "ood_lower_std": (10**lower_ood_std).tolist(),
        "ood_upper_std": (10**upper_ood_std).tolist(),
        "ood_lower_ada": (10**lower_ood_ada).tolist(),
        "ood_upper_ada": (10**upper_ood_ada).tolist(),
        "cell_ids_cal": data["cell_ids_cal"].tolist(),
        "cell_ids_ood": data["cell_ids_ood"].tolist(),
        # OOD Scores
        "mahalanobis_cal": scores_cal["mahalanobis"].tolist(),
        "mahalanobis_ood": scores_ood["mahalanobis"].tolist(),
        "isolation_cal": scores_cal["isolation"].tolist(),
        "isolation_ood": scores_ood["isolation"].tolist(),
        "pred_cal": pred_cal.tolist(),
        "pred_ood": pred_ood.tolist()
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/audit_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved evaluation results to results/audit_results.json")
    
    return data # return training data for further plotting use

if __name__ == "__main__":
    run_stress_test()
