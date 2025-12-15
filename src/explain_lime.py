import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os

from src.train_model import (
    EnhancedFeatureExtractor, FeatureExtractor, ColumnRemover, ImprovedImputer, OutlierHandler
)

def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    if 'variance_selector' in pipeline.named_steps:
        selector = pipeline.named_steps['variance_selector']
        feature_names = np.array(feature_names)[selector.get_support()]
    return feature_names

def explain_instance_lime(pipeline, X_train, X_test_transformed, instance_index, output_dir):
    """
    Generate LIME explanation for a single instance.
    """
    print(f"   [LIME] Explaining instance {instance_index}...")
    
    
    preprocessor_pipe = pipeline[:-1]
    classifier = pipeline[-1]
    
    if isinstance(X_train, pd.DataFrame):
        X_train_trans = preprocessor_pipe.transform(X_train)
    else:
        X_train_trans = X_train

    feature_names = get_feature_names(pipeline)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        verbose=False,
        random_state=42
    )
    
    data_row = X_test_transformed[instance_index]
    
    exp = explainer.explain_instance(
        data_row=data_row,
        predict_fn=classifier.predict_proba,
        num_features=10
    )

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f"LIME Analysis (Index {instance_index})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lime_plot.png'))
    plt.close()

    exp.save_to_file(os.path.join(output_dir, 'lime_report.html'))