# -*- coding: utf-8 -*-
"""
Validation pipeline for ablation study models (30-minute interval).

Supports UNet, DSC-UNet, SA-UNet, IR-UNet, and SA-IRUNet...
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Allow imports from the script's own directory and its parent
SCRIPT_DIR = Path(__file__).parent.absolute()
for p in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from ablation_models import UNet, DSCUNet, SAUNet, IRUNet, SAIRUNet
except ImportError as exc:
    print(f"Could not import models from ablation_models.py: {exc}")
    sys.exit(1)

MODEL_CLASSES = {
    "UNet": UNet,
    "DSC-UNet": DSCUNet,
    "SA-UNet": SAUNet,
    "IR-UNet": IRUNet,
    "SA-IRUNet": SAIRUNet,
}

# Accept a few aliases so older configs keep working.
MODEL_ALIASES = {
    "SAIRUNet": "SA-IRUNet",
    "IR_UNet": "IR-UNet",
    "DSC_UNet": "DSC-UNet",
    "SA_UNet": "SA-UNet",
}


# ============================================================
# CONFIGURATION
# ============================================================
VALIDATION_CONFIG = {
    "batch_size": 4,
    "thresholds_mm_per_hr": [1, 5, 15, 25, 50],
    "precipitation_threshold_mm_per_hr": 5,
    "epsilon": 1e-8,
    "num_workers": 0,
    "use_gpu_if_available": True,
    "save_visualizations": True,
    "num_sample_plots": 5,
    # Spacing between consecutive frames. Used only to label lead times
    # (e.g. "30min", "60min"). Values are not scaled by this.
    "lead_time_step_minutes": 30,
}

DEVICE = torch.device(
    "cuda" if VALIDATION_CONFIG["use_gpu_if_available"] and torch.cuda.is_available() else "cpu"
)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
else:
    print("Device: CPU (inference may be slow)")


# ============================================================
# METRICS
# ============================================================
class PrecipitationMetrics:
    def __init__(self, threshold=5.0, epsilon=1e-8, neighborhood_size=3):
        self.threshold = threshold
        self.epsilon = epsilon
        self.neighborhood_size = neighborhood_size

    def _confusion(self, target, pred):
        t = (target.flatten() >= self.threshold).astype(np.int32)
        p = (pred.flatten() >= self.threshold).astype(np.int32)
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        tn = int(np.sum((p == 0) & (t == 0)))
        return tp, fp, fn, tn

    def _csi(self, tp, fp, fn):
        denom = tp + fp + fn
        return float(tp / denom) if denom > 0 else 0.0

    def _pod(self, tp, fn):
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    def _far(self, tp, fp):
        denom = tp + fp
        return float(fp / denom) if denom > 0 else 0.0

    def _hss(self, tp, tn, fp, fn):
        num = 2 * (tp * tn - fp * fn)
        den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        return float(num / den) if den != 0 else 0.0

    def _gss(self, target, pred):
        t = (target.flatten() >= self.threshold).astype(np.int32)
        p = (pred.flatten() >= self.threshold).astype(np.int32)
        hits = np.sum((p == 1) & (t == 1))
        total_events = np.sum(t)
        total_pred = np.sum(p)
        total = t.size
        expected = (total_events * total_pred) / (total + self.epsilon)
        num = hits - expected
        den = total_events + total_pred - hits - expected + self.epsilon
        return float(num / den) if den != 0 else 0.0

    def _fss_2d(self, target_2d, pred_2d, nsize):
        from scipy.signal import convolve2d

        t = (target_2d >= self.threshold).astype(np.float32)
        p = (pred_2d >= self.threshold).astype(np.float32)

        pad = nsize // 2
        kernel = np.ones((nsize, nsize), dtype=np.float32)

        t = np.pad(t, pad, mode="constant")
        p = np.pad(p, pad, mode="constant")

        t_frac = convolve2d(t, kernel, mode="valid") / kernel.sum()
        p_frac = convolve2d(p, kernel, mode="valid") / kernel.sum()

        mse = np.mean((p_frac - t_frac) ** 2)
        ref = np.mean(t_frac ** 2) + np.mean(p_frac ** 2)

        if ref == 0:
            return 1.0 if mse == 0 else 0.0
        return float(max(0.0, min(1.0, 1.0 - mse / (ref + self.epsilon))))

    def _fss(self, target, pred):
        nsize = self.neighborhood_size
        if target.ndim > 2:
            values = [
                self._fss_2d(
                    target[i] if target.ndim == 3 else target[i, 0],
                    pred[i] if pred.ndim == 3 else pred[i, 0],
                    nsize,
                )
                for i in range(target.shape[0])
            ]
            return float(np.mean(values)) if values else 0.0
        return self._fss_2d(target, pred, nsize)

    def calculate_all(self, target, pred):
        tp, fp, fn, tn = self._confusion(target, pred)
        return {
            "CSI": self._csi(tp, fp, fn),
            "POD": self._pod(tp, fn),
            "FAR": self._far(tp, fp),
            "HSS": self._hss(tp, tn, fp, fn),
            "GSS": self._gss(target, pred),
            "FSS": self._fss(target, pred),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }


# ============================================================
# NORMALIZATION
# ============================================================
class DataNormalizer:
    """Min-max normalizer. Labels are always kept in their original scale."""

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.params = None

    def load(self, path):
        with open(path) as f:
            self.params = json.load(f)
        print(f"Normalization parameters loaded from {path}")
        return self.params

    def transform_features(self, features):
        p = self.params
        feat_range = p["feat_max"] - p["feat_min"]
        scaled = (features - p["feat_min"]) / (feat_range + self.epsilon)
        return scaled * (p["norm_max"] - p["norm_min"]) + p["norm_min"]

    def inverse_transform_predictions(self, preds):
        p = self.params
        clipped = np.clip(preds, p["norm_min"], p["norm_max"])
        scaled = (clipped - p["norm_min"]) / (p["norm_max"] - p["norm_min"] + self.epsilon)
        label_range = p["label_max"] - p["label_min"]
        return scaled * label_range + p["label_min"]


def load_normalizer(path):
    if not Path(path).exists():
        return None
    normalizer = DataNormalizer()
    normalizer.load(path)
    return normalizer


# ============================================================
# DATASET
# ============================================================
class ValidationDataset(Dataset):
    """Loads features (normalized) and labels (original scale, mm/hr) from NPZ files."""

    def __init__(self, data_path, normalizer=None):
        data_path = Path(data_path)
        features_file = data_path / "features.npz"
        labels_file = data_path / "labels.npz"

        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"features.npz or labels.npz not found in {data_path}")

        features = np.load(features_file)["features"]
        labels = np.load(labels_file)["labels"]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.maximum(features, 0)
        labels = np.maximum(labels, 0)

        if normalizer is not None:
            features = normalizer.transform_features(features)

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

        print(f"Loaded {len(self.features)} samples")
        print(f"  Features: {self.features.shape}  Labels: {self.labels.shape}")
        print(f"  Label range: [{self.labels.min():.4f}, {self.labels.max():.4f}] mm/hr")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])


# ============================================================
# HELPERS
# ============================================================
def lead_time_labels(max_steps, step_minutes=30):
    """Human-readable labels for successive prediction steps."""
    if step_minutes == 30:
        return [f"{(i + 1) * 30}min" for i in range(max_steps)]
    return [f"{(i + 1) * step_minutes}min" for i in range(max_steps)]


def regression_metrics(preds, targets):
    p = preds.flatten()
    t = targets.flatten()

    mse = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - t)))

    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else -999.0

    if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 0.0

    alpha = float(np.std(p) / (np.std(t) + 1e-8))
    beta = float(np.mean(p) / (np.mean(t) + 1e-8))
    kge = float(1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "NSE": nse, "KGE": kge, "Correlation": corr}


def pick_directory(title):
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(selected) if selected else None
    except Exception:
        return None


# ============================================================
# MODEL LOADING
# ============================================================
def resolve_model_type(model_dir, config, checkpoint):
    """Pick the model type from config, checkpoint, or folder name."""
    model_type = config.get("model_type") or checkpoint.get("model_type")

    if model_type is None:
        name = model_dir.name
        if "SA_IRUNet" in name or "SAIRUNet" in name:
            model_type = "SA-IRUNet"
        elif "IR_UNet" in name or "IRUNet" in name:
            model_type = "IR-UNet"
        elif "SA_UNet" in name or "SAUNet" in name:
            model_type = "SA-UNet"
        elif "DSC_UNet" in name or "DSCUNet" in name:
            model_type = "DSC-UNet"
        else:
            model_type = "UNet"

    return MODEL_ALIASES.get(model_type, model_type)


def build_model(class_name, config):
    params = {
        "in_channels": config.get("in_channels", 3),
        "out_channels": config.get("out_channels", 6),
        "n_filters": config.get("n_filters", 16),
        "dropout": config.get("dropout", 0.05),
        "kernel_size": config.get("kernel_size", 3),
    }
    if class_name in ("IR-UNet", "SA-IRUNet"):
        params["expand_ratio"] = config.get("expand_ratio", 4)

    return MODEL_CLASSES[class_name](**params)


def load_model(model_dir, checkpoint_path, device, config):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = resolve_model_type(model_dir, config, checkpoint)
    class_name = MODEL_ALIASES.get(model_type, model_type)

    if class_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model class: {class_name}")

    print(f"Model type: {class_name}")
    model = build_model(class_name, config)

    state = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    skipped = len(state) - len(compatible)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"  Skipped {skipped} incompatible keys")
    if missing:
        print(f"  {len(missing)} missing keys (default init)")
    if unexpected:
        print(f"  {len(unexpected)} unexpected keys (ignored)")

    model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}  ({total * 4 / 1024 / 1024:.2f} MB)")

    return model, model_type, total


# ============================================================
# VALIDATION
# ============================================================
def run_inference(model, dataloader, device):
    preds, targets = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference", leave=False):
            outputs = model(inputs.to(device))
            preds.append(outputs.float().cpu().numpy())
            targets.append(labels.float().cpu().numpy())

    elapsed = time.perf_counter() - start

    preds = np.nan_to_num(np.concatenate(preds, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(np.concatenate(targets, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    return np.maximum(preds, 0), np.maximum(targets, 0), elapsed


def validate(model, dataloader, device, config, lead_time_step_minutes=30, normalizer=None):
    print("Running inference...")
    preds, targets, inference_time = run_inference(model, dataloader, device)

    print(f"  Inference: {inference_time:.2f}s for {len(preds)} samples")
    print(f"  Raw prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Raw target range:     [{targets.min():.4f}, {targets.max():.4f}]")

    if normalizer is not None:
        preds = normalizer.inverse_transform_predictions(preds)
        print(f"  Predictions inverse-transformed: [{preds.min():.4f}, {preds.max():.4f}]")

    # Values are already in mm/hr. No scaling is applied.
    preds_mmhr = preds
    targets_mmhr = targets

    print(f"  Prediction (mm/hr): [{preds_mmhr.min():.2f}, {preds_mmhr.max():.2f}]")
    print(f"  Target (mm/hr):     [{targets_mmhr.min():.2f}, {targets_mmhr.max():.2f}]")

    reg = regression_metrics(preds_mmhr, targets_mmhr)
    print(f"  RMSE={reg['RMSE']:.4f}  MAE={reg['MAE']:.4f}")

    lead_steps = preds_mmhr.shape[1]
    thresholds = config["thresholds_mm_per_hr"]
    lead_times = lead_time_labels(lead_steps, lead_time_step_minutes)
    primary_thr = config["precipitation_threshold_mm_per_hr"]

    threshold_metrics = {}
    lead_results = []

    for lead_idx, lead_time in enumerate(lead_times):
        pred_slice = preds_mmhr[:, lead_idx, ...]
        target_slice = targets_mmhr[:, lead_idx, ...]

        per_threshold = {
            thr: PrecipitationMetrics(threshold=thr, epsilon=config["epsilon"])
                       .calculate_all(target_slice, pred_slice)
            for thr in thresholds
        }
        threshold_metrics[lead_time] = per_threshold

        ref_thr = primary_thr if primary_thr in per_threshold else thresholds[0]
        ref = per_threshold[ref_thr]
        rmse_lead = float(np.sqrt(np.mean((pred_slice.flatten() - target_slice.flatten()) ** 2)))

        lead_results.append({
            "lead_time": lead_time,
            "metrics": {
                "CSI": ref["CSI"],
                "POD": ref["POD"],
                "FAR": ref["FAR"],
                "HSS": ref["HSS"],
                "GSS": ref["GSS"],
                "FSS": ref["FSS"],
                "RMSE": rmse_lead,
            },
        })

    print(f"Processed {lead_steps} lead times across {len(thresholds)} thresholds")

    return {
        "regression_metrics": reg,
        "threshold_metrics": threshold_metrics,
        "lead_times": lead_times,
        "lead_results": lead_results,
        "inference_time": inference_time,
        "samples": len(preds_mmhr),
        "preds_mmhr": preds_mmhr,
        "targets_mmhr": targets_mmhr,
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_sample_predictions(preds, targets, lead_times, out_dir, model_name, num=5, units="mm/hr"):
    n_samples = preds.shape[0]
    if n_samples == 0:
        return

    idx_list = random.sample(range(n_samples), min(num, n_samples))
    num_leads = preds.shape[1]

    preds_clean = np.nan_to_num(preds, nan=0.0)
    targets_clean = np.nan_to_num(targets, nan=0.0)

    for sample_idx in tqdm(idx_list, desc="Generating sample plots"):
        fig, axes = plt.subplots(2, num_leads, figsize=(max(12, num_leads * 4.5), 10))
        if num_leads == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"{model_name} - Sample {sample_idx} ({units})",
                     fontsize=16, fontweight="bold", y=0.98)

        all_values = np.concatenate([preds_clean[sample_idx].flatten(),
                                     targets_clean[sample_idx].flatten()])
        vmin, vmax = np.percentile(all_values, 2), np.percentile(all_values, 98)
        if vmin == vmax:
            vmin, vmax = all_values.min() - 0.01, all_values.max() + 0.01

        for lt_idx in range(num_leads):
            label = lead_times[lt_idx] if lt_idx < len(lead_times) else f"Step {lt_idx + 1}"

            im = axes[0, lt_idx].imshow(preds_clean[sample_idx, lt_idx],
                                        cmap="turbo", vmin=vmin, vmax=vmax)
            axes[0, lt_idx].set_title(f"Predicted - {label}", fontsize=11, fontweight="bold")
            axes[0, lt_idx].axis("off")
            fig.colorbar(im, ax=axes[0, lt_idx], fraction=0.046, pad=0.04).set_label(units, fontsize=8)

            im = axes[1, lt_idx].imshow(targets_clean[sample_idx, lt_idx],
                                        cmap="turbo", vmin=vmin, vmax=vmax)
            axes[1, lt_idx].set_title(f"Ground Truth - {label}", fontsize=11, fontweight="bold")
            axes[1, lt_idx].axis("off")
            fig.colorbar(im, ax=axes[1, lt_idx], fraction=0.046, pad=0.04).set_label(units, fontsize=8)

        fig.text(0.02, 0.75, "PREDICTED", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold", color="darkblue")
        fig.text(0.02, 0.25, "GROUND TRUTH", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold", color="darkgreen")

        plt.tight_layout(rect=[0.04, 0, 0.98, 0.97])
        plt.savefig(out_dir / f"sample_{sample_idx:03d}.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()


def plot_lead_time_metrics(lead_results, out_dir, model_name):
    if not lead_results:
        return

    lead_times = [r["lead_time"] for r in lead_results]
    metrics = ["CSI", "POD", "FAR", "HSS", "GSS", "FSS"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, metric in zip(axes.flatten(), metrics):
        values = [r["metrics"].get(metric, 0) for r in lead_results]
        bars = ax.bar(lead_times, values, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_title(f"{metric} by Lead Time", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xlabel("Lead Time")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(f"{model_name} - Lead Time Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "lead_time_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_values = [r["metrics"].get("RMSE", 0) for r in lead_results]
    bars = ax.bar(lead_times, rmse_values, color="coral", alpha=0.7, edgecolor="black")
    ax.set_title(f"{model_name} - RMSE by Lead Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("RMSE (mm/hr)")
    ax.set_xlabel("Lead Time")
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_by_lead_time.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# TOP-LEVEL VALIDATION
# ============================================================
def save_results(results, out_dir, model_type, checkpoint_path, model_params,
                 data_dir, lead_time_step_minutes, normalizer):
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results["regression_metrics"]]).to_csv(
        out_dir / "regression_metrics.csv", index=False)

    threshold_rows = []
    for lead_time, thr_metrics in results["threshold_metrics"].items():
        for thr, m in thr_metrics.items():
            threshold_rows.append({"lead_time": lead_time, "threshold_mm_hr": thr, **m})
    pd.DataFrame(threshold_rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    pd.DataFrame(results["lead_results"]).to_csv(out_dir / "lead_wise_metrics.csv", index=False)

    summary = {
        "validation_settings": {
            "units": "mm/hr",
            "lead_time_step_minutes": lead_time_step_minutes,
            "thresholds_mm_per_hr": VALIDATION_CONFIG["thresholds_mm_per_hr"],
            "precipitation_threshold_mm_per_hr": VALIDATION_CONFIG["precipitation_threshold_mm_per_hr"],
            "unit_conversion_applied": False,
        },
        "model_info": {
            "model_type": model_type,
            "checkpoint": str(checkpoint_path),
            **model_params,
        },
        "data_info": {
            "data_path": str(data_dir),
            "samples": results["samples"],
            "image_size": list(results["preds_mmhr"].shape[2:]),
            "lead_steps": len(results["lead_times"]),
            "lead_times": results["lead_times"],
            "normalization_applied": normalizer is not None,
            "labels_normalized": False,
        },
        "performance": {
            "inference_time_seconds": results["inference_time"],
            "inference_time_per_sample_ms": results["inference_time"] / results["samples"] * 1000,
        },
        "regression_metrics": results["regression_metrics"],
        "threshold_metrics": results["threshold_metrics"],
        "lead_wise_metrics": results["lead_results"],
    }

    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def print_summary(results, model_type):
    print("\n" + "=" * 60)
    print(f"Validation results - {model_type}")
    print("=" * 60)
    print("Regression metrics (mm/hr):")
    for k, v in results["regression_metrics"].items():
        print(f"  {k}: {v:.6f}")

    print("\nThreshold metrics (threshold = 5 mm/hr):")
    print(f"{'Lead Time':<12} {'CSI':>10} {'POD':>10} {'FAR':>10} {'HSS':>10}")
    print("-" * 60)
    for lead_time, thr_metrics in results["threshold_metrics"].items():
        if 5 in thr_metrics:
            m = thr_metrics[5]
            print(f"{lead_time:<12} {m['CSI']:>10.4f} {m['POD']:>10.4f} "
                  f"{m['FAR']:>10.4f} {m['HSS']:>10.4f}")
    print("=" * 60)


def run_validation(data_dir, model_dir, config):
    print("=" * 70)
    print("Ablation study model validation")
    print("=" * 70)

    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            model_config = json.load(f)
        print(f"Model config: {config_file}")
    else:
        print("No config.json found; using defaults.")
        model_config = {
            "model_type": "UNet",
            "in_channels": 3,
            "out_channels": 6,
            "n_filters": 16,
            "dropout": 0.05,
            "kernel_size": 3,
            "expand_ratio": 4,
        }

    checkpoint_path = None
    for candidate in (
        model_dir / "checkpoints" / "best_model.pth",
        model_dir / "checkpoints" / "final_model.pth",
        model_dir / "best_model.pth",
        model_dir / "final_model.pth",
    ):
        if candidate.exists():
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    print(f"Checkpoint: {checkpoint_path.name}")

    normalizer = load_normalizer(model_dir / "norm_params.json") \
        or load_normalizer(data_dir / "norm_params.json")
    if normalizer is None:
        print("No normalization parameters found; using raw data.")

    dataset = ValidationDataset(data_dir, normalizer=normalizer)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])

    model, model_type, total_params = load_model(model_dir, checkpoint_path, DEVICE, model_config)

    results = validate(model, dataloader, DEVICE, config,
                       lead_time_step_minutes=config["lead_time_step_minutes"],
                       normalizer=normalizer)

    out_dir = model_dir / "validation_results_30min"
    model_params = {
        "total_parameters": total_params,
        "in_channels": model_config.get("in_channels", 3),
        "out_channels": model_config.get("out_channels", 6),
        "n_filters": model_config.get("n_filters", 16),
        "kernel_size": model_config.get("kernel_size", 3),
        "expand_ratio": model_config.get("expand_ratio", 4),
        "dropout": model_config.get("dropout", 0.05),
    }

    save_results(results, out_dir, model_type, checkpoint_path, model_params,
                 data_dir, config["lead_time_step_minutes"], normalizer)

    if config.get("save_visualizations", True) and results["samples"] > 0:
        print("Generating sample plots...")
        plot_sample_predictions(results["preds_mmhr"], results["targets_mmhr"],
                                results["lead_times"], out_dir, model_type,
                                config.get("num_sample_plots", 5), "mm/hr")
        plot_lead_time_metrics(results["lead_results"], out_dir, model_type)

    print_summary(results, model_type)
    print(f"\nResults saved to: {out_dir}")

    return results


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    print("=" * 70)
    print("Ablation study validation pipeline")
    print(f"  Batch size: {VALIDATION_CONFIG['batch_size']}")
    print(f"  Thresholds: {VALIDATION_CONFIG['thresholds_mm_per_hr']} mm/hr")
    print("=" * 70)

    data_dir = pick_directory("Select validation data directory")
    if data_dir is None:
        print("No data directory selected. Exiting.")
        return

    model_dir = pick_directory("Select model directory (contains checkpoints/ and config.json)")
    if model_dir is None:
        print("No model directory selected. Exiting.")
        return

    print(f"Data directory:  {data_dir}")
    print(f"Model directory: {model_dir}")

    run_validation(data_dir, model_dir, VALIDATION_CONFIG.copy())


if __name__ == "__main__":
    main()