import shap
import matplotlib.pyplot as plt
import os

def explain_instance_shap(pipeline, X_test_transformed, instance_index, output_dir):
    """
    Generate SHAP Waterfall plot with CORRECT feature names mapping.
    """
    print(f"   [SHAP] Explaining instance {instance_index}...")
    
    try:
        ct_step = pipeline.named_steps['preprocessor']
        sel_step = pipeline.named_steps['variance_selector']
        classifier = pipeline.named_steps['classifier']
        
        all_feature_names = ct_step.get_feature_names_out()
        
        selected_mask = sel_step.get_support()
        final_feature_names = all_feature_names[selected_mask]

        clean_feature_names = [
            name.replace('num__', '').replace('cat__', '').replace('cat_high__', '') 
            for name in final_feature_names
        ]
        
    except Exception as e:
        print(f"   [Warning] Could not map feature names dynamically: {e}")
        feature_count = X_test_transformed.shape[1]
        clean_feature_names = [f"Feature {i}" for i in range(feature_count)]

    explainer = shap.TreeExplainer(classifier)

    instance_data = X_test_transformed[instance_index:instance_index+1]

    shap_values = explainer(instance_data)

    shap_values.feature_names = clean_feature_names

    if shap_values.values.ndim == 3: 
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    os.makedirs(output_dir, exist_ok=True)
    plt.figure()

    shap.plots.waterfall(shap_values_class1[0], max_display=12, show=False)
    
    plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'), bbox_inches='tight')
    plt.close()
    print(f"   [SHAP] Saved to {output_dir}")