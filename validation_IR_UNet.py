# -*- coding: utf-8 -*-
"""
Case Study Evaluation Pipeline for SAIRUNet Model
- GUI-based data and model selection
- Loads case study data from npz files (features.npz, labels.npz)
- Loads norm_params.json from model root directory
- All metrics calculated in mm/hr units
- Generates comprehensive JSON report with all metrics
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time
import json
import sys
from tkinter import filedialog, Tk
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import random
import gc
import warnings

# ============================================================
# MODEL IMPORTS
# ============================================================
try:
    from IR_UNET import SAIRUNet
    SAIRUNET_AVAILABLE = True
    print("SAIRUNet imported successfully")
except ImportError:
    SAIRUNET_AVAILABLE = False
    print("Warning: IR_UNET module not found. SAIRUNet will not be available.")

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CASE_STUDY_CONFIG = {
    'batch_size': 8,
    'thresholds_mm_per_hr': [1,  5, 15,  25, 50],
    'precipitation_threshold_mm_per_hr': 5,
    'plot_sample_predictions': True,
    'num_sample_plots': 5,
    'num_workers': 0,
    'epsilon': 1e-8,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU - inference may be slow")

# ============================================================
# METRICS CLASS (Enhanced for all required metrics)
# ============================================================
class PrecipitationMetrics:
    """Comprehensive precipitation metrics calculator including GSS and FSS."""
    
    def __init__(self, threshold=5.0, epsilon=1e-8, neighborhood_size=3):
        self.threshold = threshold
        self.epsilon = epsilon
        self.neighborhood_size = neighborhood_size
    
    def calculate_confusion_matrix(self, target, pred):
        """Calculate confusion matrix elements."""
        target_binary = (target >= self.threshold).astype(np.int32)
        pred_binary = (pred >= self.threshold).astype(np.int32)
        
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        tn = np.sum((pred_binary == 0) & (target_binary == 0))
        
        return int(tp), int(fp), int(fn), int(tn)
    
    def calculate_csi(self, tp, fp, fn):
        """Critical Success Index."""
        denominator = tp + fp + fn
        return float(tp / (denominator + self.epsilon)) if denominator > 0 else 0.0
    
    def calculate_pod(self, tp, fn):
        """Probability of Detection."""
        denominator = tp + fn
        return float(tp / (denominator + self.epsilon)) if denominator > 0 else 0.0
    
    def calculate_far(self, tp, fp):
        """False Alarm Ratio."""
        denominator = tp + fp
        return float(fp / (denominator + self.epsilon)) if denominator > 0 else 0.0
    
    def calculate_hss(self, tp, tn, fp, fn):
        """Heidke Skill Score."""
        numerator = 2 * (tp * tn - fp * fn)
        denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        return float(numerator / (denominator + self.epsilon)) if denominator != 0 else 0.0
    
    def calculate_gss(self, target, pred):
        """Gilbert's Skill Score (ETS)."""
        target_flat = target.flatten()
        pred_flat = pred.flatten()
        
        target_binary = (target_flat >= self.threshold).astype(np.int32)
        pred_binary = (pred_flat >= self.threshold).astype(np.int32)
        
        total_events = np.sum(target_binary)
        total_predicted = np.sum(pred_binary)
        total_area = target_binary.size
        
        hits = np.sum((pred_binary == 1) & (target_binary == 1))
        expected_hits = (total_events * total_predicted) / (total_area + self.epsilon)
        
        numerator = hits - expected_hits
        denominator = total_events + total_predicted - hits - expected_hits + self.epsilon
        
        return float(numerator / denominator)
    
    def calculate_fss(self, target, pred):
        """Fractions Skill Score using neighborhood approach."""
        try:
            from scipy.signal import convolve2d
            
            neighborhood_size = self.neighborhood_size
            
            # Handle multi-dimensional arrays
            if target.ndim > 2:
                fss_values = []
                n_samples = target.shape[0]
                for i in range(n_samples):
                    target_2d = target[i] if target.ndim == 3 else target[i, 0]
                    pred_2d = pred[i] if pred.ndim == 3 else pred[i, 0]
                    fss_val = self._calculate_fss_2d(target_2d, pred_2d, neighborhood_size)
                    fss_values.append(fss_val)
                return float(np.mean(fss_values))
            else:
                return self._calculate_fss_2d(target, pred, neighborhood_size)
                
        except Exception as e:
            print(f"Warning: FSS calculation failed: {e}")
            return 0.0
    
    def _calculate_fss_2d(self, target_2d, pred_2d, neighborhood_size):
        """Calculate FSS for 2D arrays."""
        try:
            from scipy.signal import convolve2d
            
            target_binary = (target_2d >= self.threshold).astype(np.float32)
            pred_binary = (pred_2d >= self.threshold).astype(np.float32)
            
            pad = neighborhood_size // 2
            target_padded = np.pad(target_binary, pad, mode='constant', constant_values=0)
            pred_padded = np.pad(pred_binary, pad, mode='constant', constant_values=0)
            
            kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)
            
            target_fraction = convolve2d(target_padded, kernel, mode='valid')
            pred_fraction = convolve2d(pred_padded, kernel, mode='valid')
            
            kernel_area = kernel.sum()
            target_fraction = target_fraction / kernel_area
            pred_fraction = pred_fraction / kernel_area
            
            mse = np.mean((pred_fraction - target_fraction) ** 2)
            ref_mse = np.mean(target_fraction ** 2) + np.mean(pred_fraction ** 2)
            
            if ref_mse == 0:
                return 1.0 if mse == 0 else 0.0
            
            fss = 1.0 - (mse / (ref_mse + self.epsilon))
            return float(max(0.0, min(1.0, fss)))
            
        except Exception as e:
            return 0.0
    
    def calculate_all_metrics(self, target, pred):
        """Calculate all metrics."""
        tp, fp, fn, tn = self.calculate_confusion_matrix(target, pred)
        
        return {
            'CSI': self.calculate_csi(tp, fp, fn),
            'POD': self.calculate_pod(tp, fn),
            'FAR': self.calculate_far(tp, fp),
            'HSS': self.calculate_hss(tp, tn, fp, fn),
            'GSS': self.calculate_gss(target, pred),
            'FSS': self.calculate_fss(target, pred),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }

# ============================================================
# NORMALIZER CLASS
# ============================================================
class DataNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.norm_params = None
    
    def load_params(self, load_path):
        with open(load_path, 'r') as f:
            self.norm_params = json.load(f)
        print(f"Normalization parameters loaded from: {load_path}")
        return self.norm_params
    
    def transform_features(self, features):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call load_params() first.")
        
        feat_range = self.norm_params['feat_max'] - self.norm_params['feat_min']
        features_norm = (features - self.norm_params['feat_min']) / (feat_range + self.epsilon)
        features_norm = features_norm * (self.norm_params['norm_max'] - self.norm_params['norm_min']) + self.norm_params['norm_min']
        return features_norm
    
    def inverse_transform_labels(self, labels_norm):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call load_params() first.")
        
        labels_norm = (labels_norm - self.norm_params['norm_min']) / (self.norm_params['norm_max'] - self.norm_params['norm_min'] + self.epsilon)
        label_range = self.norm_params['label_max'] - self.norm_params['label_min']
        labels_orig = labels_norm * label_range + self.norm_params['label_min']
        return labels_orig

# ============================================================
# DATASET CLASS
# ============================================================
class CaseStudyDataset(Dataset):
    """Dataset for loading case study npz files."""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        
        # Load features
        features_file = self.data_path / 'features.npz'
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        # Load labels
        labels_file = self.data_path / 'labels.npz'
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Load data
        self.features = np.load(features_file)['features']
        self.labels = np.load(labels_file)['labels']
        
        # Clean data
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = np.nan_to_num(self.labels, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = np.maximum(self.features, 0)
        self.labels = np.maximum(self.labels, 0)
        
        print(f"Case Study Dataset loaded:")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Label range: [{self.labels.min():.4f}, {self.labels.max():.4f}]")
        
        # Check for precipitation events
        positive_pixels = np.sum(self.labels > 0)
        total_pixels = self.labels.size
        print(f"  Positive precipitation pixels: {positive_pixels} / {total_pixels} ({100*positive_pixels/total_pixels:.2f}%)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.from_numpy(self.labels[idx]).float()

# ============================================================
# GUI SELECTORS
# ============================================================
def select_case_study_data_gui() -> Optional[Path]:
    """Open GUI to select case study data directory."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory(
        title="Select Case Study Data Directory (contains features.npz and labels.npz)"
    )
    root.destroy()
    return Path(selected_dir) if selected_dir else None

def select_model_directory_gui() -> Optional[Path]:
    """Open GUI to select model directory."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory(
        title="Select Model Root Directory (contains checkpoints/ and norm_params.json)"
    )
    root.destroy()
    return Path(selected_dir) if selected_dir else None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def generate_lead_time_labels(max_steps, interval_minutes=30):
    """Generate human-readable lead time labels."""
    if interval_minutes == 60:
        return [f"{i+1}h" for i in range(max_steps)]
    else:
        return [f"{(i+1)*interval_minutes}min" for i in range(max_steps)]

def calculate_regression_metrics(preds, targets):
    """Calculate regression metrics."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    mse = np.mean((preds_flat - targets_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    
    ss_res = np.sum((preds_flat - targets_flat) ** 2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999.0
    
    if len(preds_flat) > 1 and np.std(preds_flat) > 0 and np.std(targets_flat) > 0:
        correlation = np.corrcoef(preds_flat, targets_flat)[0, 1]
    else:
        correlation = 0.0
    
    alpha = np.std(preds_flat) / (np.std(targets_flat) + 1e-8)
    beta = np.mean(preds_flat) / (np.mean(targets_flat) + 1e-8)
    kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'NSE': float(nse),
        'KGE': float(kge),
        'Correlation': float(correlation)
    }

def plot_case_study_predictions(preds, targets, lead_times, out_dir, num=5, units="mm/hr"):
    """Generate sample prediction plots for case study."""
    n_samples = preds.shape[0]
    idx_list = random.sample(range(n_samples), min(num, n_samples))
    num_leads = preds.shape[1]
    
    for sample_idx in tqdm(idx_list, desc="Generating sample plots"):
        fig_width = max(12, num_leads * 4.5)
        fig_height = 10
        fig, axes = plt.subplots(2, num_leads, figsize=(fig_width, fig_height))
        if num_leads == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Case Study - Sample {sample_idx} - Predictions ({units})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        preds_clean = np.nan_to_num(preds, nan=0.0)
        targets_clean = np.nan_to_num(targets, nan=0.0)
        
        all_values = np.concatenate([preds_clean[sample_idx].flatten(), 
                                    targets_clean[sample_idx].flatten()])
        global_vmin = np.percentile(all_values, 2)
        global_vmax = np.percentile(all_values, 98)
        if global_vmin == global_vmax:
            global_vmin = all_values.min()
            global_vmax = all_values.max()
            if global_vmin == global_vmax:
                global_vmin = global_vmin - 0.01
                global_vmax = global_vmax + 0.01
        
        for lt_idx in range(num_leads):
            lead_label = lead_times[lt_idx] if lt_idx < len(lead_times) else f"Step {lt_idx+1}"
            
            ax_pred = axes[0, lt_idx]
            pred_img = preds_clean[sample_idx, lt_idx]
            im_pred = ax_pred.imshow(pred_img, cmap='turbo', vmin=global_vmin, vmax=global_vmax)
            ax_pred.set_title(f'Predicted - {lead_label}', fontsize=11, fontweight='bold')
            ax_pred.axis('off')
            cbar_pred = fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            cbar_pred.set_label(units, fontsize=8)
            
            ax_gt = axes[1, lt_idx]
            gt_img = targets_clean[sample_idx, lt_idx]
            im_gt = ax_gt.imshow(gt_img, cmap='turbo', vmin=global_vmin, vmax=global_vmax)
            ax_gt.set_title(f'Ground Truth - {lead_label}', fontsize=11, fontweight='bold')
            ax_gt.axis('off')
            cbar_gt = fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            cbar_gt.set_label(units, fontsize=8)
        
        fig.text(0.02, 0.75, 'PREDICTED', va='center', rotation='vertical', 
                fontsize=12, fontweight='bold', color='darkblue')
        fig.text(0.02, 0.25, 'GROUND TRUTH', va='center', rotation='vertical', 
                fontsize=12, fontweight='bold', color='darkgreen')
        
        plt.tight_layout(rect=[0.04, 0, 0.98, 0.97])
        plt.savefig(out_dir / f'case_sample_{sample_idx:03d}.png', dpi=150, 
                   bbox_inches='tight', facecolor='white')
        plt.close()

def load_sairunet_model(model_dir: Path, checkpoint_path: Path, device, config: dict):
    """Load SAIRUNet model from checkpoint."""
    if not SAIRUNET_AVAILABLE:
        raise ImportError("SAIRUNet is not available.")
    
    print("\n" + "="*60)
    print("LOADING SAIRUNet MODEL")
    print("="*60)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    in_ch = config.get('in_channels', 3)
    out_ch = config.get('out_channels', 6)
    n_f = config.get('n_filters', 64)
    d = config.get('dropout', 0.05)
    ks = config.get('kernel_size', 5)
    expand_ratio = config.get('expand_ratio', 6)
    
    print(f"Model Configuration:")
    print(f"  Architecture: SAIRUNet")
    print(f"  Input channels: {in_ch}")
    print(f"  Output channels: {out_ch}")
    print(f"  Base filters: {n_f}")
    print(f"  Kernel size: {ks}")
    print(f"  Expand ratio: {expand_ratio}")
    print(f"  Dropout: {d}")
    
    model = SAIRUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        n_filters=n_f,
        dropout=d,
        kernel_size=ks,
        expand_ratio=expand_ratio
    )
    
    state = ckpt.get('model_state_dict', ckpt)
    
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
        print("  Removed 'module.' prefix from state dict keys")
    
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    
    for k, v in state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"  Warning: Shape mismatch for {k}")
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)
    
    model.load_state_dict(filtered_state, strict=False)
    
    if skipped_keys:
        print(f"  Warning: Skipped {len(skipped_keys)} incompatible keys")
    
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    print("="*60)
    
    return model, total_params, trainable_params

def calculate_threshold_metrics_for_lead(metrics, preds, targets, thr_list, lead_idx):
    """Calculate threshold-based metrics for a specific lead time."""
    results = {}
    for thr in thr_list:
        thr_float = float(thr)
        metrics.threshold = thr_float
        
        # Extract lead time slice
        pred_slice = preds[:, lead_idx, ...]
        target_slice = targets[:, lead_idx, ...]
        
        # Calculate all metrics for this threshold
        all_metrics = metrics.calculate_all_metrics(target_slice, pred_slice)
        
        results[thr_float] = all_metrics
    
    return results

# ============================================================
# MAIN CASE STUDY EVALUATION FUNCTION
# ============================================================
def run_case_study_evaluation():
    """Main function to run case study evaluation."""
    print("\n" + "="*70)
    print("CASE STUDY EVALUATION PIPELINE FOR SAIRUNet")
    print("All metrics calculated in mm/hr")
    print("="*70)
    
    # Check if SAIRUNet is available
    if not SAIRUNET_AVAILABLE:
        print("\nERROR: SAIRUNet model is not available!")
        print("Please ensure your model file (IR_UNET.py) is in the correct location.")
        return
    
    # Select case study data directory
    print("\nSelect case study data directory...")
    print("(Should contain features.npz and labels.npz)")
    data_dir = select_case_study_data_gui()
    if data_dir is None:
        print("No directory selected. Exiting.")
        return
    
    print(f"Case study data: {data_dir}")
    
    # Check for required files
    features_file = data_dir / 'features.npz'
    labels_file = data_dir / 'labels.npz'
    
    if not features_file.exists() or not labels_file.exists():
        print(f"ERROR: Required files not found in {data_dir}")
        print("Expected: features.npz and labels.npz")
        return
    
    # Select model directory
    print("\nSelect model root directory...")
    print("(Should contain checkpoints/ and norm_params.json)")
    model_dir = select_model_directory_gui()
    if model_dir is None:
        print("No directory selected. Exiting.")
        return
    
    print(f"Model directory: {model_dir}")
    
    # Check for required model files
    checkpoint_dir = model_dir / 'checkpoints'
    norm_file = model_dir / 'norm_params.json'
    config_file = model_dir / 'config.json'
    
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoints folder not found in {model_dir}")
        return
    
    # Find best model checkpoint
    best_model = checkpoint_dir / 'best_model.pth'
    if not best_model.exists():
        best_model = checkpoint_dir / 'final_model.pth'
        if not best_model.exists():
            pth_files = list(checkpoint_dir.glob('*.pth'))
            if pth_files:
                best_model = pth_files[0]
                print(f"Using checkpoint: {best_model.name}")
            else:
                print("ERROR: No model checkpoint (.pth) found")
                return
    
    print(f"Checkpoint: {best_model.name}")
    
    # Load configuration
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Config loaded from: {config_file}")
    else:
        print("Warning: No config.json found. Using default parameters.")
        config = {
            'in_channels': 3,
            'out_channels': 6,
            'n_filters': 64,
            'dropout': 0.05,
            'kernel_size': 5,
            'expand_ratio': 6
        }
    
    # Load normalization parameters
    normalizer = None
    apply_denorm = False
    
    if norm_file.exists():
        normalizer = DataNormalizer()
        normalizer.load_params(norm_file)
        apply_denorm = True
        print("Denormalization will be applied to predictions")
        print(f"  Label range: [{normalizer.norm_params['label_min']:.4f}, "
              f"{normalizer.norm_params['label_max']:.4f}]")
    else:
        print("Warning: No normalization parameters found - using raw model outputs")
    
    # Determine interval from config
    interval_minutes = config.get('prediction_interval_minutes', 
                                  config.get('interval_minutes', 
                                           config.get('lead_time_interval_minutes', 30)))
    max_lead_steps = config.get('max_lead_steps', config.get('out_channels', 6))
    
    print(f"\nData Configuration:")
    print(f"  Original interval: {interval_minutes} minutes")
    print(f"  Lead steps: {max_lead_steps}")
    print(f"  Evaluation units: mm/hr")
    
    # Load case study data
    print("\nLoading case study data...")
    dataset = CaseStudyDataset(data_dir)
    
    # Apply normalization to features if needed
    if normalizer:
        print("\nNormalizing features using training parameters...")
        features = dataset.features
        original_shape = features.shape
        features_flat = features.reshape(-1, *original_shape[2:])
        features_norm = normalizer.transform_features(features_flat)
        dataset.features = features_norm.reshape(original_shape)
        print("Features normalized")
    
    # Get image dimensions
    h, w = dataset.features.shape[2], dataset.features.shape[3]
    print(f"\nImage dimensions: {h}x{w}")
    
    # Create dataloader
    print(f"Creating dataloader with {len(dataset)} samples...")
    loader = DataLoader(
        dataset,
        batch_size=CASE_STUDY_CONFIG['batch_size'],
        shuffle=False,
        num_workers=CASE_STUDY_CONFIG['num_workers']
    )
    
    # Load model
    model, total_params, trainable_params = load_sairunet_model(
        model_dir, best_model, DEVICE, config
    )
    
    # Create output directory
    out_dir = data_dir / 'case_study_evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {out_dir}")
    
    # Run inference
    print("\nRunning inference...")
    all_preds, all_targets = [], []
    start = time.perf_counter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Inference Progress"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            
            # Trim outputs if needed
            if outputs.shape[1] > max_lead_steps:
                outputs = outputs[:, :max_lead_steps, ...]
                targets = targets[:, :max_lead_steps, ...]
            
            all_preds.append(outputs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())
    
    inf_time = time.perf_counter() - start
    
    # Concatenate results
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Clean up predictions
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)
    
    # Denormalize predictions if needed
    if apply_denorm and normalizer:
        print("\nDenormalizing predictions...")
        original_shape = preds.shape
        preds_flat = preds.reshape(-1, *original_shape[2:])
        preds_denorm = normalizer.inverse_transform_labels(preds_flat)
        preds = preds_denorm.reshape(original_shape)
        print("Predictions denormalized")
        print(f"  Prediction range: [{preds.min():.2f}, {preds.max():.2f}]")
        print(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    
    # Convert to mm/hr
    print("\nConverting to mm/hr for evaluation...")
    if interval_minutes != 60:
        conversion_factor = 60.0 / interval_minutes
        preds_mmhr = preds * conversion_factor
        targets_mmhr = targets * conversion_factor
        print(f"Converted from mm/{interval_minutes}min to mm/hr (multiplied by {conversion_factor:.1f})")
    else:
        preds_mmhr = preds
        targets_mmhr = targets
        print("Data already in mm/hr units")
    
    print(f"  Prediction range (mm/hr): [{preds_mmhr.min():.2f}, {preds_mmhr.max():.2f}]")
    print(f"  Target range (mm/hr): [{targets_mmhr.min():.2f}, {targets_mmhr.max():.2f}]")
    
    actual_lead_steps = preds_mmhr.shape[1]
    lead_times = generate_lead_time_labels(actual_lead_steps, interval_minutes)
    
    print(f"\nInference completed in {inf_time:.2f}s")
    print(f"  Processed {len(preds_mmhr)} samples")
    print(f"  Lead times: {', '.join(lead_times)}")
    
    # Save predictions
    np.savez_compressed(out_dir / 'predictions_mmhr.npz', predictions=preds_mmhr)
    np.savez_compressed(out_dir / 'targets_mmhr.npz', targets=targets_mmhr)
    print("Predictions and targets saved")
    
    # Generate sample plots
    if CASE_STUDY_CONFIG['plot_sample_predictions'] and len(preds_mmhr) > 0:
        print("\nGenerating sample prediction plots...")
        plot_case_study_predictions(
            preds_mmhr, targets_mmhr, lead_times, out_dir,
            CASE_STUDY_CONFIG['num_sample_plots'], "mm/hr"
        )
        print("Sample plots saved")
    
    # Calculate regression metrics
    print("\n" + "="*60)
    print("REGRESSION METRICS (mm/hr)")
    print("="*60)
    
    regression_metrics = calculate_regression_metrics(preds_mmhr, targets_mmhr)
    
    print(f"  MSE:  {regression_metrics['MSE']:.6f}")
    print(f"  RMSE: {regression_metrics['RMSE']:.6f}")
    print(f"  MAE:  {regression_metrics['MAE']:.6f}")
    print(f"  NSE:  {regression_metrics['NSE']:.6f}")
    print(f"  KGE:  {regression_metrics['KGE']:.6f}")
    print(f"  Corr: {regression_metrics['Correlation']:.6f}")
    
    # Calculate threshold metrics
    print("\n" + "="*60)
    print("THRESHOLD METRICS (CSI, POD, FAR, HSS, GSS, FSS)")
    print(f"Thresholds: {CASE_STUDY_CONFIG['thresholds_mm_per_hr']} mm/hr")
    print("="*60)
    
    thr_list = CASE_STUDY_CONFIG['thresholds_mm_per_hr']
    threshold_metrics_by_lead = {}
    
    # Initialize metrics calculator
    metrics = PrecipitationMetrics(
        threshold=thr_list[0],
        epsilon=CASE_STUDY_CONFIG['epsilon']
    )
    
    for lead_idx in range(actual_lead_steps):
        lead_time = lead_times[lead_idx]
        print(f"\nProcessing lead time: {lead_time}")
        
        lead_metrics = calculate_threshold_metrics_for_lead(
            metrics, preds_mmhr, targets_mmhr, thr_list, lead_idx
        )
        threshold_metrics_by_lead[lead_time] = lead_metrics
        
        # Print summary for this lead time
        first_thr = thr_list[0]
        print(f"  Threshold {first_thr:.1f} mm/hr: "
              f"CSI={lead_metrics[first_thr]['CSI']:.4f}, "
              f"POD={lead_metrics[first_thr]['POD']:.4f}, "
              f"FAR={lead_metrics[first_thr]['FAR']:.4f}, "
              f"HSS={lead_metrics[first_thr]['HSS']:.4f}, "
              f"GSS={lead_metrics[first_thr]['GSS']:.4f}, "
              f"FSS={lead_metrics[first_thr]['FSS']:.4f}")
    
    # ============================================================
    # SAVE COMPREHENSIVE JSON REPORT
    # ============================================================
    print("\nSaving comprehensive report...")
    
    # Prepare results for JSON
    results = {
        'evaluation_type': 'case_study',
        'model_info': {
            'architecture': 'SAIRUNet',
            'checkpoint': str(best_model),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'in_channels': config.get('in_channels', 3),
            'out_channels': config.get('out_channels', 6),
            'n_filters': config.get('n_filters', 64),
            'kernel_size': config.get('kernel_size', 5),
            'expand_ratio': config.get('expand_ratio', 6),
            'dropout': config.get('dropout', 0.05),
        },
        'data_info': {
            'data_path': str(data_dir),
            'samples': int(len(dataset)),
            'image_size': list(preds_mmhr.shape[2:]),
            'lead_steps': int(actual_lead_steps),
            'lead_times': lead_times,
            'original_interval_minutes': interval_minutes,
            'converted_to_mmhr': interval_minutes != 60,
            'conversion_factor': 60.0 / interval_minutes if interval_minutes != 60 else 1.0,
            'normalization_applied': apply_denorm,
        },
        'performance': {
            'inference_time_seconds': float(inf_time),
            'inference_time_per_sample_ms': float(inf_time / len(dataset) * 1000),
        },
        'regression_metrics': regression_metrics,
        'threshold_metrics': threshold_metrics_by_lead,
        'thresholds_mm_per_hr': thr_list,
    }
    
    # Save main results JSON
    with open(out_dir / 'case_study_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_dir / 'case_study_results.json'}")
    
    # ============================================================
    # SAVE CSV FILES
    # ============================================================
    # Regression metrics CSV
    regression_df = pd.DataFrame([regression_metrics])
    regression_df.to_csv(out_dir / 'regression_metrics.csv', index=False)
    
    # Threshold metrics CSV
    threshold_rows = []
    for lead_time, thr_metrics in threshold_metrics_by_lead.items():
        for threshold, metrics_dict in thr_metrics.items():
            row = {
                'lead_time': lead_time,
                'threshold_mm_hr': threshold
            }
            # Add all metrics
            for key in ['CSI', 'POD', 'FAR', 'HSS', 'GSS', 'FSS', 'TP', 'FP', 'FN', 'TN']:
                row[key] = metrics_dict.get(key, 0)
            threshold_rows.append(row)
    
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(out_dir / 'threshold_metrics.csv', index=False)
    print(f"CSV files saved to: {out_dir}")
    
    # ============================================================
    # GENERATE SUMMARY REPORT
    # ============================================================
    print("\n" + "="*70)
    print("CASE STUDY EVALUATION SUMMARY")
    print("="*70)
    print(f"Data: {data_dir}")
    print(f"Samples: {len(dataset)}")
    print(f"Lead times: {', '.join(lead_times)}")
    print(f"Units: mm/hr")
    
    print(f"\nRegression Metrics:")
    print(f"  RMSE: {regression_metrics['RMSE']:.4f} mm/hr")
    print(f"  MAE:  {regression_metrics['MAE']:.4f} mm/hr")
    print(f"  NSE:  {regression_metrics['NSE']:.4f}")
    print(f"  KGE:  {regression_metrics['KGE']:.4f}")
    print(f"  Corr: {regression_metrics['Correlation']:.4f}")
    
    print(f"\nThreshold Metrics (Average across lead times):")
    print(f"{'Threshold':>12} {'CSI':>8} {'POD':>8} {'FAR':>8} {'HSS':>8} {'GSS':>8} {'FSS':>8}")
    print("-"*75)
    
    for threshold in thr_list:
        # Calculate average across lead times
        avg_metrics = {}
        for metric in ['CSI', 'POD', 'FAR', 'HSS', 'GSS', 'FSS']:
            values = []
            for lead_time, thr_metrics in threshold_metrics_by_lead.items():
                if threshold in thr_metrics:
                    values.append(thr_metrics[threshold].get(metric, 0))
            avg_metrics[metric] = np.mean(values) if values else 0
        
        print(f"{threshold:>12.1f} {avg_metrics['CSI']:>8.4f} {avg_metrics['POD']:>8.4f} "
              f"{avg_metrics['FAR']:>8.4f} {avg_metrics['HSS']:>8.4f} "
              f"{avg_metrics['GSS']:>8.4f} {avg_metrics['FSS']:>8.4f}")
    
    print(f"\nResults saved to: {out_dir}")
    print("="*70)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_case_study_evaluation()