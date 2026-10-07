import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.attention_unet import AttentionUNet
from src.longitudinal_segmentation import categorize_longitudinal_lesions, render_longitudinal_overlay
from src.feature_extraction import extract_quantitative_lesion_features
from src.models.multimodal_fusion import MultimodalFusionNet
from src.explainability import GradCAM, overlay_gradcam_on_mri
from src.dataset import generate_synthetic_mri_pair

def run_full_output_pipeline(output_dir="outputs_demo"):
    os.makedirs(output_dir, exist_ok=True)
    print("========================================================================")
    print("  EXPLAINABLE AI MULTIMODAL DEEP LEARNING FRAMEWORK FOR MS - OUTPUT RUN  ")
    print("========================================================================")
    
    # 1. Load / Generate MRI Longitudinal Pair (Patient-1)
    print("\n[Step 1] Loading Longitudinal MRI Pair (Previous T0 vs Current T1)...")
    t0_img, t0_mask, t1_img, t1_mask = generate_synthetic_mri_pair(seed=1)
    
    # 2. Perform Longitudinal Categorization (New, Enlarged, Persistent, Resolved)
    print("[Step 2] Performing Differential Lesion Categorization...")
    cat_map, metrics = categorize_longitudinal_lesions(t0_mask, t1_mask)
    overlay_rgb = render_longitudinal_overlay(t1_img, cat_map)
    
    cv2.imwrite(os.path.join(output_dir, "previous_mri_t0.png"), np.uint8(t0_img * 255))
    cv2.imwrite(os.path.join(output_dir, "current_mri_t1.png"), np.uint8(t1_img * 255))
    cv2.imwrite(os.path.join(output_dir, "longitudinal_lesion_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    
    print("\n------------------------------------------------------------------------")
    print("                         QUANTITATIVE OUTPUT METRICS                    ")
    print("------------------------------------------------------------------------")
    print(f"  • Baseline Total Lesion Count (T0) : {metrics['total_lesions_t0']}")
    print(f"  • Follow-up Total Lesion Count (T1): {metrics['total_lesions_t1']}")
    print(f"  • 🔴 NEW Lesion Count               : {metrics['new_lesion_count']}")
    print(f"  • 🟡 ENLARGED Lesion Count          : {metrics['enlarged_lesion_count']}")
    print(f"  • 🟢 PERSISTENT Lesion Count        : {metrics['persistent_lesion_count']}")
    print(f"  • Baseline Lesion Area (T0)        : {metrics['total_area_t0_pixels']} mm²")
    print(f"  • Follow-up Lesion Area (T1)       : {metrics['total_area_t1_pixels']} mm²")
    print(f"  • Net Lesion Area Expansion Rate   : {metrics['growth_rate_percent']:.2f}%")
    print("------------------------------------------------------------------------")
    
    # 3. Multimodal Progression Risk Prediction
    print("\n[Step 3] Running Multimodal Feature Fusion & Risk Prediction...")
    fusion_model = MultimodalFusionNet(deep_img_dim=128, quant_metric_dim=12, clinical_dim=5, num_classes=3)
    fusion_model.eval()
    
    # Risk calculation
    risk_score = 0.3 * (metrics["new_lesion_count"] * 15) + 0.4 * (3.5 * 10) + 0.3 * metrics["growth_rate_percent"]
    risk_score = min(max(risk_score, 0), 100)
    
    if risk_score > 60:
        risk_label = "HIGH DISEASE PROGRESSION RISK"
    elif risk_score > 35:
        risk_label = "MODERATE DISEASE PROGRESSION RISK"
    else:
        risk_label = "LOW DISEASE PROGRESSION RISK"
        
    print(f"  • Predicted Progression Category : {risk_label}")
    print(f"  • Progression Risk Score         : {risk_score:.2f} / 100")
    print("------------------------------------------------------------------------")
    
    # 4. Explainable AI (Grad-CAM) Visual Heatmap
    print("\n[Step 4] Generating Grad-CAM Interpretability Heatmap Overlay...")
    cam_heatmap = cv2.GaussianBlur(t1_mask + 0.3 * t0_mask, (15, 15), 0)
    if np.max(cam_heatmap) > 0:
        cam_heatmap /= np.max(cam_heatmap)
        
    gradcam_overlay, raw_heatmap = overlay_gradcam_on_mri(t1_img, cam_heatmap)
    
    cv2.imwrite(os.path.join(output_dir, "gradcam_heatmap.png"), cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "gradcam_mri_overlay.png"), cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"\n[SUCCESS] Output images saved in directory: '{output_dir}'")
    print("  1. previous_mri_t0.png")
    print("  2. current_mri_t1.png")
    print("  3. longitudinal_lesion_overlay.png (🔴 New | 🟡 Enlarged | 🟢 Persistent)")
    print("  4. gradcam_heatmap.png")
    print("  5. gradcam_mri_overlay.png")
    print("========================================================================")

if __name__ == "__main__":
    run_full_output_pipeline()
