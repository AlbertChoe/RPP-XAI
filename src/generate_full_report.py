import pandas as pd
import numpy as np
import joblib
import os

from src.explain_lime import explain_instance_lime
from src.explain_shap import explain_instance_shap
from src.explain_global import explain_global_importance

from src.train_model import (
    EnhancedFeatureExtractor, FeatureExtractor, ColumnRemover, ImprovedImputer, OutlierHandler
)

def main():
    MODEL_PATH = './model/best_lgbm_pipeline.pkl'
    DATA_PATH = './data/train.csv' 
    BASE_REPORT_DIR = './reports'
    
    os.makedirs(BASE_REPORT_DIR, exist_ok=True)

    print("1. Loading Model & Data...")
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    
    X_raw = df.drop(columns=['coppaRisk'], errors='ignore')
    
    y_raw = df['coppaRisk']
    y_true = y_raw.map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
    
    if y_true.isnull().any():
        valid_idx = y_true.dropna().index
        X_raw = X_raw.loc[valid_idx]
        y_true = y_true.loc[valid_idx]

    print("2. Preparing Data Transformations...")
    preprocessor = pipeline[:-1]
    X_transformed = preprocessor.transform(X_raw)

    print("3. Selecting Interesting Cases...")
    preds = pipeline.predict(X_raw)
    
    indices_map = {}
    
    def get_idx(condition):
        res = np.where(condition)[0]
        return res[0] if len(res) > 0 else None

    indices_map['True_HighRisk'] = get_idx((y_true == 1) & (preds == 1))
    indices_map['True_LowRisk']  = get_idx((y_true == 0) & (preds == 0))
    # Error Cases
    indices_map['False_HighRisk'] = get_idx((y_true == 0) & (preds == 1))
    indices_map['False_LowRisk']  = get_idx((y_true == 1) & (preds == 0))
    
    if indices_map['False_HighRisk'] is None: indices_map['Random_Case_1'] = 5
    if indices_map['False_LowRisk'] is None: indices_map['Random_Case_2'] = 10

    print(f"   Selected Indices: {indices_map}")

    print("\n4. Generating XAI Reports...")
    
    for case_name, idx in indices_map.items():
        if idx is None: continue
        
        print(f"\n--- Processing Case: {case_name} (Index {idx}) ---")
        
        case_dir = os.path.join(BASE_REPORT_DIR, f"{case_name}_idx{idx}")
        os.makedirs(case_dir, exist_ok=True)
        
        # Info text
        with open(os.path.join(case_dir, 'info.txt'), 'w') as f:
            f.write(f"Case Type: {case_name}\nIndex: {idx}\nActual: {y_true.iloc[idx]}\nPredicted: {preds[idx]}\n")

        # A. Run LIME (Local)
        explain_instance_lime(pipeline, X_raw, X_transformed, idx, case_dir)
        
        # B. Run SHAP (Local)
        explain_instance_shap(pipeline, X_transformed, idx, case_dir)
        
        # C. Run Global Importance (Global Context)
        explain_global_importance(pipeline, X_raw, y_true, case_dir)
        
    print(f"\n[DONE] All reports generated in: {BASE_REPORT_DIR}")

if __name__ == "__main__":
    main()