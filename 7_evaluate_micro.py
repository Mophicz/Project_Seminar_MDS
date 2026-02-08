import os

import pandas as pd
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np  

# --- Configuration ----
input_filename = 'goldstandard_3_und_averbis_annotation.csv'
output_filename = 'levenshtein_scores_3.csv'
LST_PLOT_3 = 1.0
LST_PLOT_4 = np.arange(0.0, 1.1, 0.1)
# ----------------------

if __name__ == "__main__":
    df = pd.read_csv(os.path.join('Evaluation', input_filename), delimiter=';')

    # Replace NaN with empty strings for text
    df[['Goldstandard', 'Averbis']] = df[['Goldstandard', 'Averbis']].fillna("")

    # Force confidence to numeric, replace NaN with 0.0
    if 'Confidence' in df.columns:
        df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce').fillna(0.0)
    else:
        df['Confidence'] = 0.0

    # Calculate Levenshtein Score
    df['Levenshtein Score'] = df.apply(
        lambda row: Levenshtein.ratio(str(row['Goldstandard']), str(row['Averbis'])), axis=1
    ).astype('float32')

    # Save the results
    df.to_csv(os.path.join('Evaluation', output_filename), index=False, sep=';')
    
    # Pre-calculate arrays
    scores = df['Levenshtein Score'].values
    confidences = df['Confidence'].values
    has_gold = (df['Goldstandard'] != "").values
    has_pred = (df['Averbis'] != "").values

    # Total actual positives in Gold Standard (Constant denominator for Recall)
    total_gold_positives = np.sum(has_gold)

    # ==========================================
    # Plot 1: Histogram of Levenshtein Scores
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.hist(df['Levenshtein Score'], bins=30, label='Levenshtein Score', color='steelblue', edgecolor='black')
    plt.xlabel('Levenshtein Score')
    plt.ylabel('Absolute Häufigkeit')
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    
    # ==========================================
    # Plot 2: Metrics over Levenshtein Threshold (Stringency Sweep)
    # ==========================================
    lev_thresholds = np.linspace(0.01, 1, 1000)
    precisions = []
    recalls = []
    f1_scores = []

    for t in lev_thresholds:
        # TP: Match good enough AND real gold word
        tp = np.sum(has_gold & has_pred & (scores >= t))
        # FP: Pred exists, but bad score or no Gold
        fp = np.sum((has_gold & has_pred & (scores < t)) | (~has_gold & has_pred))
        # FN: Total Gold - TP
        # fn = total_gold_positives - tp
        fn = np.sum(has_gold & ((~has_pred) | (has_gold & has_pred & (scores < t))))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    line_p, = plt.plot(lev_thresholds, precisions, color='royalblue', label='Precision')
    line_r, = plt.plot(lev_thresholds, recalls, color='firebrick', label='Recall')
    line_f1, = plt.plot(lev_thresholds, f1_scores, color='rebeccapurple', label='F1 Score')
    
    plt.text(1.0, precisions[-1], f" {precisions[-1]:.2f}", color=line_p.get_color(), fontweight='bold', ha='left', va='center')
    plt.text(1.0, recalls[-1], f" {recalls[-1]:.2f}", color=line_r.get_color(), fontweight='bold', ha='left', va='center')
    plt.text(1.0, f1_scores[-1], f" {f1_scores[-1]:.2f}", color=line_f1.get_color(), fontweight='bold', ha='left', va='center')
    
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.xlabel('Levenshtein Score Threshold')
    plt.grid(axis='y', alpha=0.5)
    plt.legend()
    plt.show()

    # ==========================================
    # Plot 3: Metrics over Confidence Threshold (Single Primary Match Threshold)
    # ==========================================
    conf_thresholds = np.linspace(0.0, 1.0, 1000)
    precisions_conf = []
    recalls_conf = []
    f1_conf = []

    for t in conf_thresholds:
        is_kept = (confidences >= t)
        
        tp = np.sum(has_gold & has_pred & is_kept & (scores >= LST_PLOT_3))
        fp = np.sum(has_pred & is_kept & ((~has_gold) | (scores < LST_PLOT_3)))
        fn = total_gold_positives - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0 
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions_conf.append(p)
        recalls_conf.append(r)
        f1_conf.append(f1)

    plt.figure(figsize=(10, 6))
    line_p, = plt.plot(conf_thresholds, precisions_conf, color='royalblue', label='Precision')
    line_r, = plt.plot(conf_thresholds, recalls_conf, color='firebrick', label='Recall')
    line_f1, = plt.plot(conf_thresholds, f1_conf, color='rebeccapurple', label='F1 Score')

    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.xlabel(f'Confidence Threshold (with LST = {LST_PLOT_3})')
    plt.grid(axis='y', alpha=0.5)
    plt.legend(loc='lower left')
    plt.show()

    # ==========================================
    # Plot 4: Multiple PR Curves
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # color gradient from light to dark blue
    cmap = plt.get_cmap('Blues')
    
    # We loop over different definitions of "Correct Match" (e.g. 0.0 vs 1.0)
    for i, match_t in enumerate(LST_PLOT_4):
        local_precisions = []
        local_recalls = []
        
        # Sweep Confidence for this specific Match Threshold
        for conf_t in conf_thresholds:
            is_kept = (confidences >= conf_t)
            
            # TP must now satisfy the LOOP's match threshold
            tp = np.sum(has_gold & has_pred & is_kept & (scores >= match_t))
            fp = np.sum(has_pred & is_kept & ((~has_gold) | (scores < match_t)))
            fn = total_gold_positives - tp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            local_precisions.append(p)
            local_recalls.append(r)
        
        # Calculate color based on index (0.0 to 1.0)
        fraction = i / max(len(LST_PLOT_4) - 1, 1)
        color_val = 0.2 + (fraction * 0.8)
        
        plt.plot(local_recalls, local_precisions,  
                 color=cmap(color_val))
        
        if i == 0 or i == len(LST_PLOT_4) - 1:
            plt.text(local_recalls[0], local_precisions[0]-0.01, f" LST = {match_t}", color=cmap(color_val), fontweight='bold', ha='center', va='top')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.show()
    