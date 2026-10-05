"""
evaluate.py - Quantitative evaluation and waveform visualization routines.
Adheres strictly to proposal Section 4.1, 4.2, 4.3:
- Primary metric: PRD (%) = 100 * ||x - x_hat||_2 / ||x||_2
- Secondary metrics: RMSE, SNR_rec (dB) = 20 * log10(100 / PRD)
- Undefined PRD handling: if ||x||_2^2 < 1e-12, mark undefined, count separately, do not set PRD=0.
- Per-subject aggregation:
  p_{s, r} = mean PRD of valid windows for subject s and seed r.
  For CNN: p_s = mean across 3 seeds.
  For OMP: p_s = score of subject s.
  Overall Macro-PRD = mean across 15 subjects, with sample standard deviation.
- At M=77: Paired difference per subject & relative reduction: 100 * (PRD_OMP - PRD_CNN) / PRD_OMP.
- Pre-registered waveform visualization: 25th, 50th, 75th percentiles and 1 high-error failure case.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def compute_window_metrics(X_true, X_pred):
    """
    Computes PRD, RMSE, and SNR_rec for each window in the batch.
    X_true, X_pred: [N_windows, 256]
    Returns dictionary of per-window metrics and invalid window count.
    """
    X_true,X_pred = np.asarray(X_true,dtype=np.float64),np.asarray(X_pred,dtype=np.float64)
    if X_true.shape != X_pred.shape or X_true.ndim != 2:
        raise ValueError('Metric arrays must have equal [windows,N] shape')
    finite_reference = np.isfinite(X_true).all(axis=1)
    finite_prediction = np.isfinite(X_pred).all(axis=1)
    diff = X_true - X_pred
    diff_norm2 = np.sum(diff**2, axis=1)
    true_norm2 = np.sum(X_true**2, axis=1)
    
    N = X_true.shape[1]
    rmse = np.sqrt(diff_norm2 / N)
    
    valid_mask = finite_reference & finite_prediction & (true_norm2 >= 1e-12)
    invalid_count = int(np.sum(~valid_mask))
    
    prd = np.full(len(X_true),np.nan,dtype=np.float64)
    snr_rec = np.full(len(X_true),np.nan,dtype=np.float64)
    
    # Calculate for valid windows
    valid_diff_norm = np.sqrt(diff_norm2[valid_mask])
    valid_true_norm = np.sqrt(true_norm2[valid_mask])
    
    valid_prd = 100.0 * (valid_diff_norm / valid_true_norm)
    prd[valid_mask] = valid_prd
    
    # SNR_rec = 10 * log10(||x||^2 / ||x - x_hat||^2)
    # Avoid log of zero
    with np.errstate(divide='ignore',invalid='ignore'):
        snr_rec[valid_mask] = 10.0 * np.log10(true_norm2[valid_mask]/diff_norm2[valid_mask])
    reason = np.full(len(X_true),'ok',dtype='<U32')
    reason[true_norm2 < 1e-12] = 'zero_reference_energy'
    reason[~finite_reference] = 'nonfinite_reference'
    reason[~finite_prediction] = 'nonfinite_prediction'
    
    return {
        "prd": prd,
        "rmse": rmse,
        "snr_rec": snr_rec,
        "valid_mask": valid_mask,
        "invalid_count": invalid_count,
        "failure_reason":reason,
        "decoder_failures":int(np.sum(~finite_prediction)),
        "zero_energy_count":int(np.sum(finite_reference & (true_norm2 < 1e-12))),
        "exact_reconstructions":int(np.sum(valid_mask & (diff_norm2 == 0)))
    }

def aggregate_subject_metrics(metrics_dict, subject_ids):
    """
    Aggregates window metrics into per-subject means (p_s).
    Returns dict mapping subject_id -> {'prd': float, 'rmse': float, 'snr_rec': float, 'count': int}
    """
    unique_subs = np.unique(subject_ids)
    sub_results = {}
    valid_mask = metrics_dict["valid_mask"]
    
    for s in unique_subs:
        mask = (subject_ids == s) & valid_mask
        if np.any(mask):
            sub_results[s] = {
                "prd": float(np.mean(metrics_dict["prd"][mask])),
                "rmse": float(np.mean(metrics_dict["rmse"][mask])),
                "snr_rec": float(np.mean(metrics_dict["snr_rec"][mask])),
                "count": int(np.sum(mask))
            }
        else:
            sub_results[s] = {"prd": float("nan"), "rmse": float("nan"), "snr_rec": float("nan"), "count": 0}
        finite_rmse = (subject_ids == s) & np.isfinite(metrics_dict['rmse'])
        sub_results[s]['rmse'] = float(np.mean(metrics_dict['rmse'][finite_rmse])) if np.any(finite_rmse) else float('nan')
        sub_results[s]['total_count'] = int(np.sum(subject_ids == s))
        sub_results[s]['invalid_count'] = int(np.sum((subject_ids == s) & ~valid_mask))
            
    return sub_results

def compute_macro_statistics(sub_results):
    """
    Computes macro mean and sample standard deviation across all 15 subjects.
    """
    prds = [v["prd"] for v in sub_results.values() if not np.isnan(v["prd"])]
    rmses = [v["rmse"] for v in sub_results.values() if not np.isnan(v["rmse"])]
    snrs = [v["snr_rec"] for v in sub_results.values() if not np.isnan(v["snr_rec"])]
    
    return {
        "macro_prd_mean": float(np.mean(prds)),
        "macro_prd_std": float(np.std(prds, ddof=1)) if len(prds) > 1 else 0.0,
        "macro_rmse_mean": float(np.mean(rmses)),
        "macro_rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "macro_snr_mean": float(np.mean(snrs)),
        "macro_snr_std": float(np.std(snrs, ddof=1)) if len(snrs) > 1 else 0.0,
        "num_subjects": len(prds)
    }

def plot_reconstruction_examples(
    X_true, X_omp, X_cnn, prd_omp, prd_cnn, subject_ids, output_path="d:/IOT/CuoiKy/results/waveform_comparison.png", fs=64
):
    """
    Plots waveform comparison for 25th, 50th, 75th percentiles and 1 high-error failure case.
    Proposal Section 4.3 rule.
    """
    valid_mask = ~np.isnan(prd_cnn) & ~np.isnan(prd_omp)
    indices = np.where(valid_mask)[0]
    
    # Sort by CNN PRD
    sorted_idx = indices[np.argsort(prd_cnn[indices])]
    n_valid = len(sorted_idx)
    
    p25_idx = sorted_idx[int(0.25 * n_valid)]
    p50_idx = sorted_idx[int(0.50 * n_valid)]
    p75_idx = sorted_idx[int(0.75 * n_valid)]
    # High-error failure case (near 99th percentile to avoid pure artifact explosion)
    pfail_idx = sorted_idx[int(0.99 * n_valid)]
    
    selected_indices = [
        ("25th Percentile (Good)", p25_idx),
        ("50th Percentile (Median)", p50_idx),
        ("75th Percentile (Challenging)", p75_idx),
        ("99th Percentile (High Error Failure)", pfail_idx)
    ]
    
    time_axis = np.arange(X_true.shape[1]) / fs
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for ax, (title, idx) in zip(axes, selected_indices):
        s_id = subject_ids[idx]
        ax.plot(time_axis, X_true[idx], label="Ground Truth (Filtered & Norm)", color="black", linewidth=1.5)
        ax.plot(time_axis, X_omp[idx], label=f"OMP-DCT (PRD={prd_omp[idx]:.1f}%)", color="tab:blue", linestyle="--", alpha=0.85)
        ax.plot(time_axis, X_cnn[idx], label=f"CNN 1D (PRD={prd_cnn[idx]:.1f}%)", color="tab:red", linestyle="-.", alpha=0.85)
        
        ax.set_title(f"{title} - Subject {s_id} | Window #{idx}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude (z-score)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)
        
    axes[-1].set_xlabel("Time (seconds, 4s window @ 64Hz)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Waveform comparison figure saved to {output_path}")

if __name__ == "__main__":
    print("Testing evaluate.py...")
    N = 256
    X = np.random.randn(50, N).astype(np.float32)
    X_pred = X + 0.1 * np.random.randn(50, N).astype(np.float32)
    subs = np.array(["S1"] * 25 + ["S2"] * 25)
    
    m = compute_window_metrics(X, X_pred)
    print("Mean PRD:", np.mean(m["prd"]))
    sub_m = aggregate_subject_metrics(m, subs)
    print("Subject metrics:", sub_m)
    macro = compute_macro_statistics(sub_m)
    print("Macro stats:", macro)
