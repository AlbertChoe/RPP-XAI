import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.train_model import (
    ColumnRemover,
    EnhancedFeatureExtractor,
    FeatureExtractor,
    ImprovedImputer,
    OutlierHandler,
)


def explain_global_importance(pipeline, X_raw, y_true, output_dir, top_n=10):
    """
    Generate Global Feature Importance using PERMUTATION IMPORTANCE.
    Ini adalah metode Post-hoc sejati karena kita mengacak fitur untuk melihat dampaknya.
    """
    print("   [GLOBAL] Calculating Permutation Importance (Model-Agnostic)...")

    result = permutation_importance(
        pipeline,
        X_raw,
        y_true,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="accuracy",
    )

    feature_names = X_raw.columns

    # Create DataFrame
    feature_imp_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": result.importances_mean,
                "Std": result.importances_std,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .head(top_n)
    )

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_imp_df,
        palette="viridis",
        hue="Feature",
        legend=False,
    )

    plt.title(f"Top {top_n} Global Features (Permutation Importance)")
    plt.xlabel("Decrease in Accuracy when Feature is Shuffled")
    plt.ylabel("Features")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "global_importance.png")
    plt.savefig(save_path)
    plt.close()

    print(f"   [GLOBAL] Saved to {save_path}")
