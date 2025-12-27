import pandas as pd
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np  # Added for threshold range generation

# --- Configuration ----
input_filename = 'gold_standard_vs_prediction.csv'
output_filename = 'gold_standard_vs_prediction_scores.csv'
# ----------------------

if __name__ == "__main__":
    df = pd.read_csv(input_filename, delimiter=';')

    # FIX: Replace NaN with empty strings
    df = df.fillna("")

    df['Levenshtein Score'] = df.apply(
        lambda row: Levenshtein.ratio(str(row['Gold Standard']), str(row['Prediction'])), axis=1
    ).astype('float32')

    # save the results
    df.to_csv(output_filename, index=False, sep=';')
    
    # --- Plot 1: Histogram of Scores ---
    plt.figure(figsize=(10, 6))
    plt.hist(df['Levenshtein Score'], bins=30, label='Levenshtein Score')
    plt.xlabel('Levenshtein Score')
    plt.ylabel('Absolute Häufigkeit')
    plt.show()
    
    # --- Plot 2: Precision, Recall, F1 over Threshold ---
    thresholds = np.linspace(0.01, 1, 1000)
    precisions = []
    recalls = []
    f1_scores = []

    # Get values for faster processing
    scores = df['Levenshtein Score'].values
    has_gold = (df['Gold Standard'] != "").values
    has_pred = (df['Prediction'] != "").values

    for t in thresholds:
        # TP: Score is high enough AND it corresponds to a real Gold word
        tp = np.sum((scores >= t) & has_gold)
        
        # FP: We predicted a word (has_pred), but score was too low (or Gold was empty)
        fp = np.sum((scores < t) & has_pred)
        
        # FN: There was a Gold word (has_gold), but our score was too low
        fn = np.sum((scores < t) & has_gold)
        
        # Calculate Metrics
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    
    line_p, = plt.plot(thresholds, precisions, label='Precision')
    line_r, = plt.plot(thresholds, recalls, label='Recall')
    line_f1, = plt.plot(thresholds, f1_scores, label='F1 Score')
    
    # Label for Precision
    plt.text(1.0, precisions[-1], f" {precisions[-1]:.2f}", 
             color=line_p.get_color(), fontweight='bold', 
             ha='left', va='center')

    # Label for Recall
    plt.text(1.0, recalls[-1], f" {recalls[-1]:.2f}", 
             color=line_r.get_color(), fontweight='bold', 
             ha='left', va='center')

    # Label for F1 Score
    plt.text(1.0, f1_scores[-1], f" {f1_scores[-1]:.2f}", 
             color=line_f1.get_color(), fontweight='bold', 
             ha='left', va='center')
    
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.xlabel('Levenshtein Threshold')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.show()