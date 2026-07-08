import os
import pandas as pd
import numpy as np
from scipy import stats

benchmark_path = r"c:\Users\Dell\Downloads\drid\indie_comic_pipeline\matrix_evaluation_zone\outputs\benchmark_results.csv"
user_study_path = r"c:\Users\Dell\Downloads\drid\indie_comic_pipeline\matrix_evaluation_zone\outputs\user_study_raw_ratings.csv"

def compute_fleiss_kappa(ratings_df, rating_col, n_subjects=20, n_raters=15, n_categories=5):
    # Pivot ratings_df to get count of each category for each subject
    # We first group by story_id and count the occurrences of each rating (1 to 5)
    counts = np.zeros((n_subjects, n_categories))
    
    for story_id in range(1, n_subjects + 1):
        story_ratings = ratings_df[ratings_df['story_id'] == story_id][rating_col].values
        for r in story_ratings:
            if 1 <= r <= 5:
                counts[story_id - 1, int(r) - 1] += 1
                
    # Calculate P_i
    P_i = (np.sum(counts * (counts - 1), axis=1)) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    
    # Calculate p_j
    p_j = np.sum(counts, axis=0) / (n_subjects * n_raters)
    P_e_bar = np.sum(p_j ** 2)
    
    # Calculate Kappa
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa, P_bar, P_e_bar

def run_verification():
    print("======================================================================")
    # 1. Load benchmark results and run paired t-test
    df_bench = pd.read_csv(benchmark_path)
    
    print("[1] Paired T-Test results for 600 Panels:")
    for metric in ["dinov2", "clip_i", "lpips"]:
        mdcp_vals = df_bench[f"mdcp_{metric}"]
        sd_vals = df_bench[f"storydiffusion_{metric}"]
        
        t_stat, p_val = stats.ttest_rel(mdcp_vals, sd_vals)
        print(f"  Metric: {metric.upper()}")
        print(f"    MDCP Mean: {np.mean(mdcp_vals):.4f} +/- {np.std(mdcp_vals):.4f}")
        print(f"    StoryDiff Mean: {np.mean(sd_vals):.4f} +/- {np.std(sd_vals):.4f}")
        print(f"    T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4e}")

    # 1b. Calculate and verify conceptual Consistency Energy (E_cons)
    # E_cons = 0.3 * LPIPS + 0.3 * (1 - CLIP_I) + 0.4 * (1 - DINOv2)
    mdcp_energy = 0.3 * df_bench["mdcp_lpips"] + 0.3 * (1 - df_bench["mdcp_clip_i"]) + 0.4 * (1 - df_bench["mdcp_dinov2"])
    sd_energy = 0.3 * df_bench["storydiffusion_lpips"] + 0.3 * (1 - df_bench["storydiffusion_clip_i"]) + 0.4 * (1 - df_bench["storydiffusion_dinov2"])
    
    t_stat_e, p_val_e = stats.ttest_rel(mdcp_energy, sd_energy)
    print("\n  Conceptual Consistency Energy (E_cons) paired t-test:")
    print(f"    MDCP Mean E_cons: {np.mean(mdcp_energy):.4f} +/- {np.std(mdcp_energy):.4f}")
    print(f"    StoryDiff Mean E_cons: {np.mean(sd_energy):.4f} +/- {np.std(sd_energy):.4f}")
    print(f"    T-Statistic (Energy): {t_stat_e:.4f}, P-Value (Energy): {p_val_e:.4e}")
        
    # 2. Load user study results and calculate Fleiss Kappa and means
    df_study = pd.read_csv(user_study_path)
    
    print("\n[2] User Study Ratings (15 Evaluators, 20 Stories):")
    for axis in ["identity", "style", "narrative"]:
        mdcp_ratings = df_study[f"mdcp_{axis}"]
        sd_ratings = df_study[f"storydiffusion_{axis}"]
        
        print(f"  Axis: {axis.upper()}")
        print(f"    MDCP Mean: {np.mean(mdcp_ratings):.4f} +/- {np.std(mdcp_ratings):.4f}")
        print(f"    StoryDiff Mean: {np.mean(sd_ratings):.4f} +/- {np.std(sd_ratings):.4f}")
        
        # Calculate Fleiss' Kappa for MDCP ratings
        kappa_mdcp, _, _ = compute_fleiss_kappa(df_study, f"mdcp_{axis}")
        kappa_sd, _, _ = compute_fleiss_kappa(df_study, f"storydiffusion_{axis}")
        print(f"    Fleiss Kappa (MDCP): {kappa_mdcp:.4f}")
        print(f"    Fleiss Kappa (StoryDiff): {kappa_sd:.4f}")

if __name__ == "__main__":
    run_verification()
