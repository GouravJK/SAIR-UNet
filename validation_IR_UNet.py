# -*- coding: utf-8 -*-
"""
Case study evaluation pipeline for SAIRUNet.

GUI-based selection of case study data and model directory. Loads features
and labels from NPZ files, applies the training-time normalizer to features,
denormalizes predictions, and reports metrics...
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from IR_UNET import SAIRUNet
    SAIRUNET_AVAILABLE = True
except ImportError:
    SAIRUNET_AVAILABLE = False
    print("IR_UNET module not found; SAIRUNet will not be available.")


# ============================================================
# CONFIGURATION
# ============================================================
CASE_STUDY_CONFIG = {
    "batch_size": 8,
    "thresholds_mm_per_hr": [1, 5, 15, 25, 50],
    "precipitation_threshold_mm_per_hr": 5,
    "plot_sample_predictions": True,
    "num_sample_plots": 5,
    "num_workers": 0,
    "epsilon": 1e-8,
    # Spacing between consecutive prediction frames. Used only to label
    # lead times (e.g. "30min", "60min"). Values are never scaled by this.
    "lead_time_step_minutes": 30,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
else:
    print("Device: CPU (inference may be slow)")


# ============================================================
# METRICS
# ============================================================
class PrecipitationMetrics:
    """Threshold-based precipitation metrics."""

    def __init__(self, threshold=5.0, epsilon=1e-8, neighborhood_size=3):
        self.threshold = threshold
        self.epsilon = epsilon
        self.neighborhood_size = neighborhood_size

    def _confusion(self, target, pred):
        t = (target >= self.threshold).astype(np.int32)
        p = (pred >= self.threshold).astype(np.int32)
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        tn = int(np.sum((p == 0) & (t == 0)))
        return tp, fp, fn, tn

    def _csi(self, tp, fp, fn):
        den = tp + fp + fn
        return float(tp / den) if den > 0 else 0.0

    def _pod(self, tp, fn):
        den = tp + fn
        return float(tp / den) if den > 0 else 0.0

    def _far(self, tp, fp):
        den = tp + fp
        return float(fp / den) if den > 0 else 0.0

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

    def _fss_2d(self, target_2d, pred_2d):
        from scipy.signal import convolve2d

        n = self.neighborhood_size
        t = (target_2d >= self.threshold).astype(np.float32)
        p = (pred_2d >= self.threshold).astype(np.float32)

        pad = n // 2
        kernel = np.ones((n, n), dtype=np.float32)

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
        if target.ndim > 2:
            values = [
                self._fss_2d(
                    target[i] if target.ndim == 3 else target[i, 0],
                    pred[i] if pred.ndim == 3 else pred[i, 0],
                )
                for i in range(target.shape[0])
            ]
            return float(np.mean(values)) if values else 0.0
        return self._fss_2d(target, pred)

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
    """Min-max normalizer. Only predictions are inverse-transformed."""

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
        rng = p["feat_max"] - p["feat_min"]
        scaled = (features - p["feat_min"]) / (rng + self.epsilon)
        return scaled * (p["norm_max"] - p["norm_min"]) + p["norm_min"]

    def inverse_transform_predictions(self, preds):
        p = self.params
        clipped = np.clip(preds, p["norm_min"], p["norm_max"])
        scaled = (clipped - p["norm_min"]) / (p["norm_max"] - p["norm_min"] + self.epsilon)
        label_range = p["label_max"] - p["label_min"]
        return scaled * label_range + p["label_min"]


# ============================================================
# DATASET
# ============================================================
class CaseStudyDataset(Dataset):
    """Loads case-study features and labels from NPZ files."""

    def __init__(self, data_path):
        data_path = Path(data_path)
        features_file = data_path / "features.npz"
        labels_file = data_path / "labels.npz"

        if not features_file.exists():
            raise FileNotFoundError(f"Missing file: {features_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Missing file: {labels_file}")

        features = np.load(features_file)["features"]
        labels = np.load(labels_file)["labels"]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.maximum(features, 0)
        labels = np.maximum(labels, 0)

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

        positive = int(np.sum(self.labels > 0))
        total = self.labels.size

        print("Case study dataset")
        print(f"  Features: {self.features.shape}")
        print(f"  Labels:   {self.labels.shape}")
        print(f"  Label range: [{self.labels.min():.4f}, {self.labels.max():.4f}] mm/hr")
        print(f"  Positive pixels: {positive}/{total} ({100 * positive / total:.2f}%)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]).float(),
                torch.from_numpy(self.labels[idx]).float())


# ============================================================
# GUI HELPERS
# ============================================================
def pick_directory(title):
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(selected) if selected else None
    except Exception:
        return None


# ============================================================
# UTILITIES
# ============================================================
def lead_time_labels(max_steps, step_minutes=30):
    if step_minutes == 60:
        return [f"{i + 1}h" for i in range(max_steps)]
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

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "NSE": nse,
            "KGE": kge, "Correlation": corr}


def find_checkpoint(checkpoint_dir):
    for name in ("best_model.pth", "final_model.pth"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate

    pth_files = sorted(checkpoint_dir.glob("*.pth"))
    return pth_files[0] if pth_files else None


# ============================================================
# MODEL LOADING
# ============================================================
def load_sairunet(checkpoint_path, device, config):
    if not SAIRUNET_AVAILABLE:
        raise ImportError("SAIRUNet is not available.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    in_ch = config.get("in_channels", 3)
    out_ch = config.get("out_channels", 6)
    n_f = config.get("n_filters", 64)
    d = config.get("dropout", 0.05)
    ks = config.get("kernel_size", 5)
    expand = config.get("expand_ratio", 6)

    model = SAIRUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        n_filters=n_f,
        dropout=d,
        kernel_size=ks,
        expand_ratio=expand,
    )

    state = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    skipped = len(state) - len(compatible)

    model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"  Skipped {skipped} incompatible keys")

    model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"SAIRUNet loaded ({in_ch} in, {out_ch} out, filters={n_f}, "
          f"kernel={ks}, expand={expand}, dropout={d})")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model, total, trainable


# ============================================================
# PLOTTING
# ============================================================
def plot_sample_predictions(preds, targets, lead_times, out_dir, num=5, units="mm/hr"):
    n_samples = preds.shape[0]
    if n_samples == 0:
        return

    idx_list = random.sample(range(n_samples), min(num, n_samples))
    num_leads = preds.shape[1]

    preds_clean = np.nan_to_num(preds, nan=0.0)
    targets_clean = np.nan_to_num(targets, nan=0.0)

    for sample_idx in tqdm(idx_list, desc="Sample plots"):
        fig, axes = plt.subplots(2, num_leads, figsize=(max(12, num_leads * 4.5), 10))
        if num_leads == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"Case study - Sample {sample_idx} ({units})",
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
            fig.colorbar(im, ax=axes[0, lt_idx], fraction=0.046, pad=0.04) \
                .set_label(units, fontsize=8)

            im = axes[1, lt_idx].imshow(targets_clean[sample_idx, lt_idx],
                                        cmap="turbo", vmin=vmin, vmax=vmax)
            axes[1, lt_idx].set_title(f"Ground truth - {label}", fontsize=11, fontweight="bold")
            axes[1, lt_idx].axis("off")
            fig.colorbar(im, ax=axes[1, lt_idx], fraction=0.046, pad=0.04) \
                .set_label(units, fontsize=8)

        fig.text(0.02, 0.75, "PREDICTED", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold", color="darkblue")
        fig.text(0.02, 0.25, "GROUND TRUTH", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold", color="darkgreen")

        plt.tight_layout(rect=[0.04, 0, 0.98, 0.97])
        plt.savefig(out_dir / f"case_sample_{sample_idx:03d}.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================
def run_case_study_evaluation():
    print("=" * 70)
    print("Case study evaluation - SAIRUNet")
    print("=" * 70)

    if not SAIRUNET_AVAILABLE:
        print("SAIRUNet is not available. Ensure IR_UNET.py is importable.")
        return

    data_dir = pick_directory(
        "Select case study data directory (contains features.npz and labels.npz)")
    if data_dir is None:
        print("No data directory selected. Exiting.")
        return

    model_dir = pick_directory(
        "Select model root directory (contains checkpoints/ and norm_params.json)")
    if model_dir is None:
        print("No model directory selected. Exiting.")
        return

    print(f"Data directory:  {data_dir}")
    print(f"Model directory: {model_dir}")

    # ----- data -----
    dataset = CaseStudyDataset(data_dir)

    # ----- model config -----
    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Config loaded from {config_file}")
    else:
        print("No config.json found; using defaults.")
        config = {
            "in_channels": 3,
            "out_channels": 6,
            "n_filters": 64,
            "dropout": 0.05,
            "kernel_size": 5,
            "expand_ratio": 6,
        }

    # ----- normalizer -----
    normalizer = None
    norm_file = model_dir / "norm_params.json"
    if norm_file.exists():
        normalizer = DataNormalizer()
        normalizer.load(norm_file)
        print(f"Denormalization will be applied to predictions "
              f"(label range: [{normalizer.params['label_min']:.4f}, "
              f"{normalizer.params['label_max']:.4f}])")
    else:
        print("No normalization parameters found; predictions used as-is.")

    # ----- checkpoint -----
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"checkpoints/ not found in {model_dir}")
        return

    checkpoint = find_checkpoint(checkpoint_dir)
    if checkpoint is None:
        print("No checkpoint (.pth) found.")
        return
    print(f"Checkpoint: {checkpoint.name}")

    max_lead_steps = config.get("out_channels", 6)

    # ----- feature normalization -----
    if normalizer is not None:
        original_shape = dataset.features.shape
        flat = dataset.features.reshape(-1, *original_shape[2:])
        dataset.features = normalizer.transform_features(flat).reshape(original_shape)
        print("Features normalized using training parameters.")

    loader = DataLoader(
        dataset,
        batch_size=CASE_STUDY_CONFIG["batch_size"],
        shuffle=False,
        num_workers=CASE_STUDY_CONFIG["num_workers"],
    )

    # ----- model -----
    model, total_params, trainable_params = load_sairunet(checkpoint, DEVICE, config)

    out_dir = data_dir / "case_study_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ----- inference -----
    print("Running inference...")
    preds_list, targets_list = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Inference", leave=False):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)

            if outputs.shape[1] > max_lead_steps:
                outputs = outputs[:, :max_lead_steps, ...]
                targets = targets[:, :max_lead_steps, ...]

            preds_list.append(outputs.float().cpu().numpy())
            targets_list.append(targets.float().cpu().numpy())

    inference_time = time.perf_counter() - start

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    # ----- denormalize predictions only -----
    if normalizer is not None:
        original_shape = preds.shape
        flat = preds.reshape(-1, *original_shape[2:])
        preds = normalizer.inverse_transform_predictions(flat).reshape(original_shape)
        print("Predictions denormalized to original label scale.")

    # Values are already mm/hr; nothing is scaled here.
    preds_mmhr = preds
    targets_mmhr = targets

    print(f"  Inference: {inference_time:.2f}s for {len(preds_mmhr)} samples")
    print(f"  Prediction range (mm/hr): [{preds_mmhr.min():.2f}, {preds_mmhr.max():.2f}]")
    print(f"  Target range (mm/hr):     [{targets_mmhr.min():.2f}, {targets_mmhr.max():.2f}]")

    actual_lead_steps = preds_mmhr.shape[1]
    lead_times = lead_time_labels(actual_lead_steps,
                                  CASE_STUDY_CONFIG["lead_time_step_minutes"])

    # ----- save raw arrays -----
    np.savez_compressed(out_dir / "predictions.npz", predictions=preds_mmhr)
    np.savez_compressed(out_dir / "targets.npz", targets=targets_mmhr)

    # ----- sample plots -----
    if CASE_STUDY_CONFIG["plot_sample_predictions"] and len(preds_mmhr) > 0:
        plot_sample_predictions(preds_mmhr, targets_mmhr, lead_times, out_dir,
                                CASE_STUDY_CONFIG["num_sample_plots"], "mm/hr")

    # ----- regression metrics -----
    reg = regression_metrics(preds_mmhr, targets_mmhr)
    print("\nRegression metrics (mm/hr)")
    for k, v in reg.items():
        print(f"  {k}: {v:.6f}")

    # ----- threshold metrics -----
    thr_list = CASE_STUDY_CONFIG["thresholds_mm_per_hr"]
    threshold_metrics_by_lead = {}

    metrics_calc = PrecipitationMetrics(epsilon=CASE_STUDY_CONFIG["epsilon"])

    for lead_idx, lead_time in enumerate(lead_times):
        pred_slice = preds_mmhr[:, lead_idx, ...]
        target_slice = targets_mmhr[:, lead_idx, ...]

        per_threshold = {}
        for thr in thr_list:
            metrics_calc.threshold = float(thr)
            per_threshold[float(thr)] = metrics_calc.calculate_all(target_slice, pred_slice)
        threshold_metrics_by_lead[lead_time] = per_threshold

    print("\nThreshold metrics (averaged across lead times)")
    header = f"{'Threshold':>10} {'CSI':>8} {'POD':>8} {'FAR':>8} {'HSS':>8} {'GSS':>8} {'FSS':>8}"
    print(header)
    print("-" * len(header))
    for thr in thr_list:
        averages = {}
        for metric in ("CSI", "POD", "FAR", "HSS", "GSS", "FSS"):
            values = [threshold_metrics_by_lead[lt][thr][metric]
                      for lt in threshold_metrics_by_lead]
            averages[metric] = float(np.mean(values)) if values else 0.0
        print(f"{thr:>10.1f} {averages['CSI']:>8.4f} {averages['POD']:>8.4f} "
              f"{averages['FAR']:>8.4f} {averages['HSS']:>8.4f} "
              f"{averages['GSS']:>8.4f} {averages['FSS']:>8.4f}")

    # ----- JSON report -----
    report = {
        "evaluation_type": "case_study",
        "model_info": {
            "architecture": "SAIRUNet",
            "checkpoint": str(checkpoint),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "in_channels": config.get("in_channels", 3),
            "out_channels": config.get("out_channels", 6),
            "n_filters": config.get("n_filters", 64),
            "kernel_size": config.get("kernel_size", 5),
            "expand_ratio": config.get("expand_ratio", 6),
            "dropout": config.get("dropout", 0.05),
        },
        "data_info": {
            "data_path": str(data_dir),
            "samples": int(len(dataset)),
            "image_size": list(preds_mmhr.shape[2:]),
            "lead_steps": int(actual_lead_steps),
            "lead_times": lead_times,
            "lead_time_step_minutes": CASE_STUDY_CONFIG["lead_time_step_minutes"],
            "units": "mm/hr",
            "unit_conversion_applied": False,
            "normalization_applied": normalizer is not None,
        },
        "performance": {
            "inference_time_seconds": float(inference_time),
            "inference_time_per_sample_ms": float(inference_time / len(dataset) * 1000),
        },
        "regression_metrics": reg,
        "threshold_metrics": threshold_metrics_by_lead,
        "thresholds_mm_per_hr": thr_list,
    }

    with open(out_dir / "case_study_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ----- CSV outputs -----
    pd.DataFrame([reg]).to_csv(out_dir / "regression_metrics.csv", index=False)

    rows = []
    for lead_time, per_thr in threshold_metrics_by_lead.items():
        for thr, m in per_thr.items():
            rows.append({
                "lead_time": lead_time,
                "threshold_mm_hr": thr,
                **{k: m.get(k, 0) for k in
                   ("CSI", "POD", "FAR", "HSS", "GSS", "FSS", "TP", "FP", "FN", "TN")},
            })
    pd.DataFrame(rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    run_case_study_evaluation()
