import re
import os

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment

# --- Configuration ----
gold_standard = 'gold_standard_3.csv'
model_annotation = 'medications_35.csv'
output_filename = 'goldstandard_3_und_model_annotation.csv'
# ----------------------

def natural_sort_key(s):
    """
    Key function for natural sorting. 
    Splits string into text and numbers so '2.txt' comes before '10.txt'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def preprocess(df):
    # 1. Create a copy to avoid modifying the original dataframe in-place
    df_clean = df.copy()

    # 2. Iterate over columns to clean text
    # We select only columns of type 'object' (strings) to avoid errors on numbers
    for col in df_clean.select_dtypes(include=['object']).columns:
        # astype(str) ensures we handle any mixed types gracefully
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    return df_clean

def get_best_alignment(df_gold, df_pred):
    # 1. Clean Data
    df_gold.columns = [c.strip() for c in df_gold.columns]
    df_pred.columns = [c.strip() for c in df_pred.columns]
    
    # Ensure medication columns are strings
    df_gold['medication'] = df_gold['medication'].astype(str)
    df_pred['medication'] = df_pred['medication'].astype(str)

    # 2. Fix Filename Discrepancies (Encoding/Typos)
    gs_files = df_gold['filename'].unique()
    pred_files = df_pred['filename'].unique()
    file_mapping = {}
    
    for p_file in pred_files:
        if p_file not in gs_files:
            # Fuzzy match filename to find the correct Gold Standard file
            best_match = None
            best_ratio = 0.0
            for g_file in gs_files:
                ratio = SequenceMatcher(None, p_file, g_file).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = g_file
            
            if best_ratio > 0.7: # High confidence match
                file_mapping[p_file] = best_match
    
    if file_mapping:
        df_pred['filename'] = df_pred['filename'].replace(file_mapping)

    # 3. Perform Global Linear Assignment per File
    all_filenames = sorted(list(set(df_gold['filename'].unique()) | set(df_pred['filename'].unique())))
    matched_data = []

    for fname in all_filenames:
        gs_meds = df_gold[df_gold['filename'] == fname]['medication'].tolist()
        pred_meds = df_pred[df_pred['filename'] == fname]['medication'].tolist()
        
        n_gs, n_pred = len(gs_meds), len(pred_meds)
        cost_matrix = np.zeros((n_gs, n_pred))
        
        # Build Cost Matrix (Cost = 1 - Similarity)
        for i in range(n_gs):
            for j in range(n_pred):
                s1, s2 = gs_meds[i], pred_meds[j]
                sim = SequenceMatcher(None, s1, s2).ratio()
                
                # Heuristic: Boost score if one is a significant substring of the other
                if len(s1) > 4 and len(s2) > 4 and (s1 in s2 or s2 in s1):
                    sim = max(sim, 0.8)
                    
                cost_matrix[i, j] = 1 - sim
        
        # Hungarian Algorithm
        if n_gs > 0 and n_pred > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []
            
        # Filter matches by threshold
        matched_indices_gs = set()
        matched_indices_pred = set()
        THRESHOLD = 0.3
        
        for r, c in zip(row_ind, col_ind):
            if (1 - cost_matrix[r, c]) >= THRESHOLD:
                matched_data.append({
                    'filename': fname,
                    'Gold Standard': gs_meds[r],
                    'Prediction': pred_meds[c]
                })
                matched_indices_gs.add(r)
                matched_indices_pred.add(c)
        
        # Handle unmatched (Blanks)
        for i in range(n_gs):
            if i not in matched_indices_gs:
                matched_data.append({'filename': fname, 'Gold Standard': gs_meds[i], 'Prediction': ""})
                
        for j in range(n_pred):
            if j not in matched_indices_pred:
                matched_data.append({'filename': fname, 'Gold Standard': "", 'Prediction': pred_meds[j]})

    matched_data.sort(key=lambda x: natural_sort_key(x['filename']))
    return pd.DataFrame(matched_data)[['filename', 'Gold Standard', 'Prediction']]

if __name__ == "__main__":
    # Load Data
    df_gold = preprocess(pd.read_csv(os.path.join('Goldstandard_annotationen', gold_standard), sep=','))
    df_pred = preprocess(pd.read_csv(os.path.join('Model_annotationen', model_annotation), sep=';'))

    # Call the function
    result_df = get_best_alignment(df_gold, df_pred)

    # Save to CSV
    result_df.to_csv(os.path.join('Evaluation', output_filename), index=False, sep=';')
    