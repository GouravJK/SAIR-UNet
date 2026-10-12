# -*- coding: utf-8 -*-
"""
SAIR-UNet Evaluation Pipeline

Expects:
    - data_dir/features.npz, data_dir/labels.npz
    - model_dir/checkpoints/best_model.pth
    - (optional) model_dir/norm_params.json or data_dir/norm_params.json

Outputs (saved under data_dir/evaluation/<model_name>/):
    - regression_metrics.csv
    - threshold_metrics.csv
    - overall_threshold_metrics.csv
    - evaluation_summary.json
    - sample_*.png (sample predictions)
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
import warnings
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import random
import os
from tkinter import filedialog, Tk

# Add the script's directory to sys.path so local modules can be found
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Direct import from scripts.IR_UNET
try:
    from scripts.IR_UNET import SAIRUNet
    print("  ✓ Successfully imported SAIRUNet from scripts.IR_UNET")
except ImportError as e:
    # If the above fails, try a fallback import from the current directory
    try:
        from IR_UNET import SAIRUNet
        print("  ✓ Successfully imported SAIRUNet from IR_UNET (fallback)")
    except ImportError as e2:
        raise ImportError(
            "Could not import SAIRUNet. Please ensure that:\n"
            "  - The file 'IR_UNET.py' exists in a subdirectory named 'scripts' (scripts/IR_UNET.py), or\n"
            "  - The file 'IR_UNET.py' is in the current directory.\n"
            f"Original errors: {e}, {e2}"
        )

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'batch_size': 1,
    'thresholds_mm_per_hr': [1, 5, 15, 25, 50],
    'epsilon': 1e-8,
    'num_workers': 0,
    'use_gpu_if_available': True,
    'save_visualizations': True,
    'num_sample_plots': 3,
    'prediction_interval_minutes': 30,          # default accumulation interval
    'use_all_thresholds': True,                 # if True, use all thresholds, else only 5 mm/hr
    'neighborhood_size': 5,
    'verbose': False,
    'cmap': 'turbo',
    'individual_plot_width_per_lead': 5.0,
    'individual_plot_height': 10.0,
    'individual_plot_dpi': 300,
}

# ============================================================
# METRICS CLASS (internal, no external dependency)
# ============================================================
class PrecipitationMetrics:
    """
    Compute categorical precipitation metrics (CSI, POD, FAR, HSS, GSS, FSS)
    for a given threshold.
    """
    def __init__(self, threshold: float = 5.0, epsilon: float = 1e-8, neighborhood_size: int = 3):
        self.threshold = threshold
        self.epsilon = epsilon
        self.neighborhood_size = neighborhood_size

    def calculate_confusion_matrix(self, target: np.ndarray, pred: np.ndarray):
        target_binary = (target >= self.threshold).astype(np.int32)
        pred_binary = (pred >= self.threshold).astype(np.int32)
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        tn = np.sum((pred_binary == 0) & (target_binary == 0))
        return int(tp), int(fp), int(fn), int(tn)

    def calculate_csi(self, tp, fp, fn):
        denom = tp + fp + fn
        return float(tp / (denom + self.epsilon)) if denom > 0 else 0.0

    def calculate_pod(self, tp, fn):
        denom = tp + fn
        return float(tp / (denom + self.epsilon)) if denom > 0 else 0.0

    def calculate_far(self, tp, fp):
        denom = tp + fp
        return float(fp / (denom + self.epsilon)) if denom > 0 else 0.0

    def calculate_hss(self, tp, tn, fp, fn):
        numerator = 2 * (tp * tn - fp * fn)
        denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        return float(numerator / (denominator + self.epsilon)) if denominator != 0 else 0.0

    def calculate_gss(self, target, pred):
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
        try:
            from scipy.signal import convolve2d
            nsize = self.neighborhood_size
            target_binary = (target >= self.threshold).astype(np.float32)
            pred_binary = (pred >= self.threshold).astype(np.float32)
            if target.ndim > 2:
                fss_values = []
                n_samples = target.shape[0]
                for i in range(n_samples):
                    t_2d = target[i] if target.ndim == 3 else target[i, 0]
                    p_2d = pred[i] if pred.ndim == 3 else pred[i, 0]
                    fss_val = self._fss_2d(t_2d, p_2d, nsize)
                    fss_values.append(fss_val)
                return float(np.mean(fss_values))
            else:
                return self._fss_2d(target, pred, nsize)
        except Exception:
            return 0.0

    def _fss_2d(self, target_2d, pred_2d, nsize):
        from scipy.signal import convolve2d
        target_binary = (target_2d >= self.threshold).astype(np.float32)
        pred_binary = (pred_2d >= self.threshold).astype(np.float32)
        pad = nsize // 2
        target_padded = np.pad(target_binary, pad, mode='constant', constant_values=0)
        pred_padded = np.pad(pred_binary, pad, mode='constant', constant_values=0)
        kernel = np.ones((nsize, nsize), dtype=np.float32)
        target_fraction = convolve2d(target_padded, kernel, mode='valid') / kernel.sum()
        pred_fraction = convolve2d(pred_padded, kernel, mode='valid') / kernel.sum()
        mse = np.mean((pred_fraction - target_fraction) ** 2)
        ref_mse = np.mean(target_fraction ** 2) + np.mean(pred_fraction ** 2)
        if ref_mse == 0:
            return 1.0 if mse == 0 else 0.0
        fss = 1.0 - (mse / (ref_mse + self.epsilon))
        return float(max(0.0, min(1.0, fss)))

    def calculate_all_metrics(self, target, pred):
        tp, fp, fn, tn = self.calculate_confusion_matrix(target, pred)
        return {
            'CSI': self.calculate_csi(tp, fp, fn),
            'POD': self.calculate_pod(tp, fn),
            'FAR': self.calculate_far(tp, fp),
            'HSS': self.calculate_hss(tp, tn, fp, fn),
            'GSS': self.calculate_gss(target, pred),
            'FSS': self.calculate_fss(target, pred),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
        }

# ============================================================
# DATASET CLASS
# ============================================================
class CaseStudyDataset(Dataset):
    def __init__(self, data_path: Path, normalizer=None, apply_norm=True):
        self.data_path = Path(data_path)
        features_file = self.data_path / 'features.npz'
        labels_file = self.data_path / 'labels.npz'
        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError("features.npz or labels.npz not found")
        self.features = np.load(features_file)['features']
        self.labels = np.load(labels_file)['labels']
        # Clean
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = np.nan_to_num(self.labels, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = np.maximum(self.features, 0)
        self.labels = np.maximum(self.labels, 0)

        if apply_norm and normalizer is not None:
            orig_shape = self.features.shape
            flat = self.features.reshape(-1, *orig_shape[2:])
            if hasattr(normalizer, 'transform_features'):
                norm_flat = normalizer.transform_features(flat)
            else:
                norm_flat = normalizer.transform(flat)
            self.features = norm_flat.reshape(orig_shape).astype(np.float32)
            print("  Features normalized")
        else:
            self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])

# ============================================================
# NORMALIZER LOADER
# ============================================================
def load_normalizer_from_file(norm_file: Path):
    if not norm_file.exists():
        return None
    with open(norm_file, 'r') as f:
        params = json.load(f)
    class Normalizer:
        def __init__(self, p):
            self.norm_params = p
            self.epsilon = 1e-8
        def transform_features(self, features):
            feat_min = self.norm_params['feat_min']
            feat_max = self.norm_params['feat_max']
            norm_min = self.norm_params.get('norm_min', 0)
            norm_max = self.norm_params.get('norm_max', 1)
            feat_range = feat_max - feat_min
            features_norm = (features - feat_min) / (feat_range + self.epsilon)
            features_norm = features_norm * (norm_max - norm_min) + norm_min
            return features_norm
        def inverse_transform_labels(self, labels_norm):
            norm_min = self.norm_params.get('norm_min', 0)
            norm_max = self.norm_params.get('norm_max', 1)
            label_min = self.norm_params['label_min']
            label_max = self.norm_params['label_max']
            labels_norm = (labels_norm - norm_min) / (norm_max - norm_min + self.epsilon)
            label_range = label_max - label_min
            return labels_norm * label_range + label_min
    return Normalizer(params)

def get_normalizer(data_dir: Path, model_dir: Path) -> Optional[object]:
    norm_file = data_dir / 'norm_params.json'
    if norm_file.exists():
        print(f"  Loading normalization from data directory: {norm_file}")
        return load_normalizer_from_file(norm_file)
    norm_file = model_dir / 'norm_params.json'
    if norm_file.exists():
        print(f"  Loading normalization from model directory: {norm_file}")
        return load_normalizer_from_file(norm_file)
    return None

# ============================================================
# SAIRUNet MODEL LOADER
# ============================================================
def load_sairunet_model(model_dir: Path, device: torch.device, config: dict) -> Tuple[torch.nn.Module, dict]:
    # Locate checkpoint
    checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / 'best_model.pth'
        if not checkpoint_path.exists():
            pth_files = list(model_dir.glob('**/*.pth'))
            if pth_files:
                checkpoint_path = pth_files[0]
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    # Load config if present
    config_file = model_dir / 'config.json'
    model_config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            model_config = json.load(f)

    # Model parameters
    in_ch = model_config.get('in_channels', 3)
    out_ch = model_config.get('out_channels', 6)
    n_f = model_config.get('n_filters', 64)
    d = model_config.get('dropout', 0.05)
    ks = model_config.get('kernel_size', 5)
    expand = model_config.get('expand_ratio', 6)

    # Instantiate model
    model = SAIRUNet(in_channels=in_ch, out_channels=out_ch,
                     n_filters=n_f, dropout=d, kernel_size=ks, expand_ratio=expand)

    # Load state dict
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    missing, _ = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {
        'model_type': 'SAIRUNet',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'in_channels': in_ch,
        'out_channels': out_ch,
        'checkpoint': str(checkpoint_path),
        'n_filters': n_f,
        'dropout': d,
        'kernel_size': ks,
        'expand_ratio': expand,
    }
    return model, model_info

# ============================================================
# INFERENCE AND METRICS
# ============================================================
def evaluate_model(model, dataloader, device, config: dict, interval_minutes: int = 30,
                   normalizer: Optional[object] = None):
    print("  Running inference...")
    all_preds, all_targets = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Inference", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.float().cpu().numpy())
            all_targets.append(targets.float().cpu().numpy())
    inf_time = time.perf_counter() - start

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    # Inverse-normalize predictions if a normalizer is provided
    if normalizer is not None and hasattr(normalizer, 'inverse_transform_labels'):
        orig_shape = preds.shape
        preds_flat = preds.reshape(-1, *orig_shape[2:])
        preds_flat = normalizer.inverse_transform_labels(preds_flat)
        preds = preds_flat.reshape(orig_shape)
        print("  Predictions inverse‑normalized to physical units")

    # Keep original units for plotting, convert to mm/hr for metrics
    preds_plot = preds.copy()
    targets_plot = targets.copy()
    if interval_minutes != 60:
        factor = 60.0 / interval_minutes
        preds_mmhr = preds * factor
        targets_mmhr = targets * factor
        plot_units = f"mm/{interval_minutes}min"
        print(f"  Metrics computed in mm/hr (×{factor:.1f}), plots in {plot_units}")
    else:
        preds_mmhr = preds
        targets_mmhr = targets
        plot_units = "mm/hr"
        print("  Using mm/hr for both metrics and plots")

    print(f"  Inference time: {inf_time:.2f}s for {len(preds_plot)} samples")

    # ---- Regression metrics (mm/hr) ----
    pred_flat = preds_mmhr.flatten()
    target_flat = targets_mmhr.flatten()
    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - target_flat))
    ss_res = np.sum((pred_flat - target_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999.0
    if len(pred_flat) > 1 and np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    else:
        corr = 0.0
    alpha = np.std(pred_flat) / (np.std(target_flat) + 1e-8)
    beta = np.mean(pred_flat) / (np.mean(target_flat) + 1e-8)
    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    regression_metrics = {
        'MSE': float(mse), 'RMSE': float(rmse), 'MAE': float(mae),
        'NSE': float(nse), 'KGE': float(kge), 'Correlation': float(corr),
        'units': 'mm/hr'
    }

    # ---- Threshold metrics (mm/hr) ----
    use_all = config.get('use_all_thresholds', False)
    thresholds = config['thresholds_mm_per_hr'] if use_all else [5.0]

    lead_steps = preds_mmhr.shape[1]
    threshold_metrics = {}
    lead_times = generate_lead_time_labels(lead_steps, interval_minutes)

    # For overall averages
    all_lead_metrics = {thr: {m: [] for m in ['CSI','POD','FAR','HSS','GSS','FSS']}
                        for thr in thresholds}

    window_size = config.get('neighborhood_size', 3)
    verbose = config.get('verbose', False)

    for lead_idx in tqdm(range(lead_steps), desc="Lead times", leave=False):
        lead_time = lead_times[lead_idx]
        pred_slice = preds_mmhr[:, lead_idx, ...]
        target_slice = targets_mmhr[:, lead_idx, ...]
        lead_metrics = {}
        for thr in thresholds:
            metrics_calc = PrecipitationMetrics(threshold=thr, epsilon=config['epsilon'],
                                                neighborhood_size=window_size)
            lead_metrics[thr] = metrics_calc.calculate_all_metrics(target_slice, pred_slice)
            for m in ['CSI','POD','FAR','HSS','GSS','FSS']:
                all_lead_metrics[thr][m].append(lead_metrics[thr][m])
        threshold_metrics[lead_time] = lead_metrics

        if verbose:
            first_thr = thresholds[0]
            m = lead_metrics[first_thr]
            print(f"  {lead_time}: CSI={m['CSI']:.3f}, POD={m['POD']:.3f}, FAR={m['FAR']:.3f}, "
                  f"HSS={m['HSS']:.3f}, GSS={m['GSS']:.3f}, FSS={m['FSS']:.3f}")

    overall_threshold_metrics = {}
    for thr in thresholds:
        overall_threshold_metrics[thr] = {}
        for m in ['CSI','POD','FAR','HSS','GSS','FSS']:
            vals = all_lead_metrics[thr][m]
            overall_threshold_metrics[thr][m] = float(np.mean(vals)) if vals else 0.0

    return {
        'regression_metrics': regression_metrics,
        'threshold_metrics': threshold_metrics,
        'overall_threshold_metrics': overall_threshold_metrics,
        'lead_times': lead_times,
        'inference_time': inf_time,
        'samples': len(preds_plot),
        'preds_mmhr': preds_plot,       # for plotting (original units)
        'targets_mmhr': targets_plot,
        'plot_units': plot_units,
        'metrics_units': 'mm/hr',
    }

def generate_lead_time_labels(max_steps, interval_minutes):
    if interval_minutes == 60:
        return [f"{i+1}h" for i in range(max_steps)]
    else:
        return [f"{(i+1)*interval_minutes}min" for i in range(max_steps)]

# ============================================================
# PLOTTING
# ============================================================
def get_safe_limits(data, percentile_low=2, percentile_high=98, fallback_min=0.0, fallback_max=1.0):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return float(fallback_min), float(fallback_max)
    vmin = np.percentile(finite_data, percentile_low)
    vmax = np.percentile(finite_data, percentile_high)
    if not np.isfinite(vmin):
        vmin = np.min(finite_data)
    if not np.isfinite(vmax):
        vmax = np.max(finite_data)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = np.min(finite_data)
        vmax = np.max(finite_data)
    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01
    if vmin >= vmax:
        vmin, vmax = fallback_min, fallback_max
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = fallback_min, fallback_max
    return float(vmin), float(vmax)

def plot_sample_predictions(preds, targets, lead_times, out_dir, num=3, units="mm/30min",
                            cmap='turbo', config=None):
    if config is None:
        config = CONFIG
    n_samples = preds.shape[0]
    idx_list = random.sample(range(n_samples), min(num, n_samples))
    num_leads = preds.shape[1]
    width = num_leads * config.get('individual_plot_width_per_lead', 5.0)
    height = config.get('individual_plot_height', 10.0)
    dpi = config.get('individual_plot_dpi', 150)

    for sample_idx in idx_list:
        fig, axes = plt.subplots(2, num_leads, figsize=(width, height), dpi=dpi)
        if num_leads == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f'Sample {sample_idx} - Predictions ({units})', fontsize=14, fontweight='bold')
        preds_clean = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        targets_clean = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        for lt_idx in range(num_leads):
            lead_label = lead_times[lt_idx] if lt_idx < len(lead_times) else f"Step {lt_idx+1}"
            col_data = np.concatenate([
                preds_clean[sample_idx, lt_idx].flatten(),
                targets_clean[sample_idx, lt_idx].flatten()
            ])
            vmin, vmax = get_safe_limits(col_data)

            ax_pred = axes[0, lt_idx]
            im_pred = ax_pred.imshow(preds_clean[sample_idx, lt_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f'Pred - {lead_label}', fontsize=10, fontweight='bold')
            ax_pred.axis('off')
            cbar_pred = fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            cbar_pred.set_label(units, fontsize=8)

            ax_gt = axes[1, lt_idx]
            im_gt = ax_gt.imshow(targets_clean[sample_idx, lt_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax_gt.set_title(f'GT - {lead_label}', fontsize=10, fontweight='bold')
            ax_gt.axis('off')
            cbar_gt = fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            cbar_gt.set_label(units, fontsize=8)

        fig.text(0.02, 0.75, 'PREDICTED', va='center', rotation='vertical',
                 fontweight='bold', fontsize=12)
        fig.text(0.02, 0.25, 'GROUND TRUTH', va='center', rotation='vertical',
                 fontweight='bold', fontsize=12)
        plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
        plt.savefig(out_dir / f'sample_{sample_idx:03d}.png', dpi=dpi,
                    bbox_inches='tight', facecolor='white')
        plt.close()

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def run_sairunet_evaluation(data_dir: Path, model_dir: Path, config: dict):
    device = torch.device('cuda' if config.get('use_gpu_if_available', True) and torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    interval_minutes = config.get('prediction_interval_minutes', 30)
    cmap = config.get('cmap', 'turbo')
    plot_units = f"mm/{interval_minutes}min" if interval_minutes != 60 else "mm/hr"

    print("\n" + "="*70)
    print("EVALUATING SAIR-UNet")
    print("="*70)

    # Load normalizer
    normalizer = get_normalizer(data_dir, model_dir)
    dataset = CaseStudyDataset(data_dir, normalizer=normalizer, apply_norm=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'])

    # Load model
    model, model_info = load_sairunet_model(model_dir, device, config)
    print(f"  Model parameters: {model_info['total_parameters']:,}")

    # Run inference and compute metrics
    result = evaluate_model(model, dataloader, device, config, interval_minutes,
                            normalizer=normalizer)

    # Create output directory
    model_name = model_dir.name
    out_dir = data_dir / 'evaluation' / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save regression metrics
    reg_df = pd.DataFrame([result['regression_metrics']])
    reg_df.to_csv(out_dir / 'regression_metrics.csv', index=False)

    # Save threshold metrics per lead time
    threshold_rows = []
    for lead_time, thr_metrics in result['threshold_metrics'].items():
        for thr, metrics_dict in thr_metrics.items():
            row = {'lead_time': lead_time, 'threshold_mm_hr': thr}
            row.update(metrics_dict)
            threshold_rows.append(row)
    thr_df = pd.DataFrame(threshold_rows)
    thr_df.to_csv(out_dir / 'threshold_metrics.csv', index=False)

    # Save overall threshold metrics (averaged over lead times)
    overall_rows = []
    for thr, metrics_dict in result['overall_threshold_metrics'].items():
        row = {'threshold_mm_hr': thr}
        row.update(metrics_dict)
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / 'overall_threshold_metrics.csv', index=False)

    # Save JSON summary
    summary = {
        'model_info': model_info,
        'data_info': {
            'data_path': str(data_dir),
            'samples': len(dataset),
            'image_size': list(result['preds_mmhr'].shape[2:]),
            'lead_steps': len(result['lead_times']),
            'lead_times': result['lead_times'],
            'interval_minutes': interval_minutes,
            'plot_units': result['plot_units'],
            'metrics_units': result['metrics_units'],
        },
        'performance': {
            'inference_time_seconds': result['inference_time'],
            'inference_time_per_sample_ms': result['inference_time'] / result['samples'] * 1000,
        },
        'regression_metrics': result['regression_metrics'],
        'threshold_metrics': result['threshold_metrics'],
        'overall_threshold_metrics': result['overall_threshold_metrics'],
        'thresholds_mm_per_hr': list(result['overall_threshold_metrics'].keys()),
    }
    with open(out_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Sample plots
    if config.get('save_visualizations', True) and result['samples'] > 0:
        print("  Generating sample plots...")
        plot_sample_predictions(result['preds_mmhr'], result['targets_mmhr'],
                                result['lead_times'], out_dir,
                                config.get('num_sample_plots', 3),
                                result['plot_units'], cmap=cmap, config=config)

    print(f"  Results saved to {out_dir}")
    print("="*70)
    return result

# ============================================================
# GUI SELECTORS
# ============================================================
def select_directory_gui(title: str = "Select Directory") -> Optional[Path]:
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(selected) if selected else None
    except:
        print("GUI selection failed. Please enter path manually.")
        return None

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("SAIR-UNet EVALUATION PIPELINE")
    print("="*70)
    print("Expected structure:")
    print("  model_dir/checkpoints/best_model.pth")
    print("  model_dir/norm_params.json (optional)")
    print("  data_dir/features.npz, data_dir/labels.npz")
    print("  Metrics computed in mm/hr, plots in original units (mm/30min by default)\n")

    data_dir = select_directory_gui("Select Case Study Data Directory")
    if data_dir is None:
        print("No data directory selected. Exiting.")
        return
    print(f"Data directory: {data_dir}")

    model_dir = select_directory_gui("Select SAIR-UNet Model Directory")
    if model_dir is None:
        print("No model directory selected. Exiting.")
        return
    print(f"Model directory: {model_dir}")

    # Optional: override config
    config = CONFIG.copy()
    # To evaluate only at 5 mm/hr, set use_all_thresholds=False
    # config['use_all_thresholds'] = False
    # config['verbose'] = True

    run_sairunet_evaluation(data_dir, model_dir, config)

if __name__ == "__main__":
    main()