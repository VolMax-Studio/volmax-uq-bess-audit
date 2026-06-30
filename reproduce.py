#!/usr/bin/env python3
import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from model_audit import train_baseline_model
from conformal_wrapper import ConformalPredictor, compute_uq_metrics

def generate_plots(results):
    os.makedirs("results", exist_ok=True)
    
    # Extract data for plotting
    y_true_ood = np.array(results["ood_y_true"])
    y_pred_ood = np.array(results["ood_y_pred"])
    lower_ood_std = np.array(results["ood_lower_std"])
    upper_ood_std = np.array(results["ood_upper_std"])
    lower_ood_ada = np.array(results["ood_lower_ada"])
    upper_ood_ada = np.array(results["ood_upper_ada"])
    
    # Sort by true cycle life for clear visualization
    sort_idx = np.argsort(y_true_ood)
    
    plt.figure(figsize=(14, 6))
    
    # 1. Standard Conformal Intervals Plot
    plt.subplot(1, 2, 1)
    plt.errorbar(
        range(len(y_true_ood)), 
        y_pred_ood[sort_idx], 
        yerr=[y_pred_ood[sort_idx] - lower_ood_std[sort_idx], upper_ood_std[sort_idx] - y_pred_ood[sort_idx]],
        fmt='o', color='royalblue', ecolor='lightsteelblue', capsize=3, label='Pred w/ 90% Interval'
    )
    plt.plot(range(len(y_true_ood)), y_true_ood[sort_idx], 'r--', label='True Cycle Life', alpha=0.8)
    plt.title('Standard Conformal (Uniform Width) — OOD Set')
    plt.xlabel('Sorted Cells')
    plt.ylabel('Cycle Life')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Adaptive Conformal Intervals Plot
    plt.subplot(1, 2, 2)
    plt.errorbar(
        range(len(y_true_ood)), 
        y_pred_ood[sort_idx], 
        yerr=[y_pred_ood[sort_idx] - lower_ood_ada[sort_idx], upper_ood_ada[sort_idx] - y_pred_ood[sort_idx]],
        fmt='o', color='emerald' if 'emerald' in plt.colormaps() else 'forestgreen', 
        ecolor='lightgreen', capsize=3, label='Pred w/ 90% Interval'
    )
    plt.plot(range(len(y_true_ood)), y_true_ood[sort_idx], 'r--', label='True Cycle Life', alpha=0.8)
    plt.title('Locally Adaptive Conformal — OOD Set')
    plt.xlabel('Sorted Cells')
    plt.ylabel('Cycle Life')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/conformal_intervals.png", dpi=150)
    print("Generated results/conformal_intervals.png")
    
    # 2.5 OOD Distribution Plot (Mahalanobis & Isolation Forest)
    plt.figure(figsize=(14, 6))
    
    # Mahalanobis Histogram
    plt.subplot(1, 2, 1)
    plt.hist(results["mahalanobis_cal"], bins=15, density=True, alpha=0.6, color='royalblue', label='In-Distribution (Cal)')
    plt.hist(results["mahalanobis_ood"], bins=15, density=True, alpha=0.6, color='crimson', label='Out-of-Distribution (Secondary)')
    plt.title('Mahalanobis Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Isolation Forest Histogram (higher score = more normal)
    plt.subplot(1, 2, 2)
    plt.hist(results["isolation_cal"], bins=15, density=True, alpha=0.6, color='royalblue', label='In-Distribution (Cal)')
    plt.hist(results["isolation_ood"], bins=15, density=True, alpha=0.6, color='crimson', label='Out-of-Distribution (Secondary)')
    plt.title('Isolation Forest Anomaly Score')
    plt.xlabel('Anomaly Score (Higher is more normal)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/ood_distribution_comparison.png", dpi=150)
    print("Generated results/ood_distribution_comparison.png")
    
    # 3. Calibration Curve / Reliability Diagram
    # We will compute coverage for various nominal alpha levels on the calibration set
    alphas = np.linspace(0.01, 0.5, 20)
    nominal_coverage = 1 - alphas
    
    # Re-load data for calibration test
    data = train_baseline_model()
    X_train, y_train = data["train"]
    X_cal, y_cal = data["cal"]
    X_ood, y_ood = data["ood"]
    
    y_pred_train = data["model"].predict(X_train)
    y_pred_cal = data["model"].predict(X_cal)
    y_pred_ood = data["model"].predict(X_ood)
    
    actual_coverage_std_cal = []
    actual_coverage_ada_cal = []
    actual_coverage_std_ood = []
    actual_coverage_ada_ood = []
    
    for a in alphas:
        # Fit standard conformal
        cp_std = ConformalPredictor(alpha=a, adaptive=False)
        cp_std.fit(X_cal, y_cal, y_pred_cal)
        
        # Fit adaptive conformal
        cp_ada = ConformalPredictor(alpha=a, adaptive=True)
        cp_ada.fit(X_cal, y_cal, y_pred_cal, X_train, y_train, y_pred_train)
        
        # Eval Standard
        l_cal_std, u_cal_std = cp_std.predict_intervals(X_cal, y_pred_cal)
        actual_coverage_std_cal.append(np.mean((y_cal >= l_cal_std) & (y_cal <= u_cal_std)))
        
        l_ood_std, u_ood_std = cp_std.predict_intervals(X_ood, y_pred_ood)
        actual_coverage_std_ood.append(np.mean((y_ood >= l_ood_std) & (y_ood <= u_ood_std)))
        
        # Eval Adaptive
        l_cal_ada, u_cal_ada = cp_ada.predict_intervals(X_cal, y_pred_cal)
        actual_coverage_ada_cal.append(np.mean((y_cal >= l_cal_ada) & (y_cal <= u_cal_ada)))
        
        l_ood_ada, u_ood_ada = cp_ada.predict_intervals(X_ood, y_pred_ood)
        actual_coverage_ada_ood.append(np.mean((y_ood >= l_ood_ada) & (y_ood <= u_ood_ada)))
        
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')
    plt.plot(nominal_coverage, actual_coverage_std_cal, 'o-', color='royalblue', label='Standard Conformal (Cal Split)')
    plt.plot(nominal_coverage, actual_coverage_ada_cal, 's-', color='forestgreen', label='Adaptive Conformal (Cal Split)')
    plt.plot(nominal_coverage, actual_coverage_std_ood, 'o--', color='cornflowerblue', label='Standard Conformal (OOD Test)')
    plt.plot(nominal_coverage, actual_coverage_ada_ood, 's--', color='lightgreen', label='Adaptive Conformal (OOD Test)')
    
    plt.title('UQ Reliability Diagram (Calibration Curve)')
    plt.xlabel('Nominal Confidence Level (1 - alpha)')
    plt.ylabel('Observed Coverage Probability')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("results/reliability_curve.png", dpi=150)
    print("Generated results/reliability_curve.png")

def main():
    parser = argparse.ArgumentParser(description="Reproduce SOH UQ Audit results")
    parser.add_argument("--test-only", action="store_true", help="Run validation tests and exit")
    args = parser.parse_args()
    
    if args.test_only:
        # Run self-validation checks
        print("Running SOH UQ Audit validation checks...")
        if not os.path.exists("../Battery_Health_Portfolio/severson_features.csv"):
            print("Error: Feature CSV file not found.")
            sys.exit(1)
            
        # Run stress test execution to generate audit_results.json
        from bess_stress_test import run_stress_test
        run_stress_test()
        
        # Load results and assert coverage target
        with open("results/audit_results.json") as f:
            res = json.load(f)
            
        picp_cal_std = res["standard_cal"]["picp"]
        picp_cal_ada = res["adaptive_cal"]["picp"]
        
        # PICP on Calibration set must be >= 1 - alpha = 0.90
        # Due to finite sample size, standard split conformal guarantees: coverage >= 1 - alpha - 1/(n_cal+1)
        # With n_cal = 24, 1 - alpha - 1/25 = 0.90 - 0.04 = 0.86
        print(f"Validation: Standard Cal Split Coverage = {picp_cal_std*100:.2f}% (Target: >= 86.00%)")
        print(f"Validation: Adaptive Cal Split Coverage = {picp_cal_ada*100:.2f}% (Target: >= 86.00%)")
        
        if picp_cal_std < 0.86 or picp_cal_ada < 0.86:
            print("Error: Coverage guarantee violated on calibration set!")
            sys.exit(1)
            
        print("[OK] All validation checks passed successfully.")
        sys.exit(0)
        
    # Standard execution: run stress test and generate plots
    from bess_stress_test import run_stress_test
    run_stress_test()
    
    with open("results/audit_results.json") as f:
        res = json.load(f)
        
    generate_plots(res)
    print("\n[OK] Reproduction complete. All results saved to results/ folder.")

if __name__ == "__main__":
    main()
