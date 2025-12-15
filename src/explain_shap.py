import shap
import matplotlib.pyplot as plt
import os

from src.train_model import (
    EnhancedFeatureExtractor, FeatureExtractor, ColumnRemover, ImprovedImputer, OutlierHandler
)

def explain_instance_shap(pipeline, X_test_transformed, instance_index, output_dir):
    """
    Generate SHAP Waterfall plot for a single instance.
    """
    print(f"   [SHAP] Explaining instance {instance_index}...")
    
    classifier = pipeline[-1]
    
    explainer = shap.TreeExplainer(classifier)
    
    instance_data = X_test_transformed[instance_index:instance_index+1]
    shap_values = explainer(instance_data)
    
    if shap_values.values.ndim == 3: 
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    
    shap.plots.waterfall(shap_values_class1[0], max_display=10, show=False)
    
    plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'), bbox_inches='tight')
    plt.close()
    print(f"   [SHAP] Saved to {output_dir}")