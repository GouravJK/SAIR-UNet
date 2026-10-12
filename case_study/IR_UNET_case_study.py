# -*- coding: utf-8 -*-
"""
SAIR-UNet evaluation pipeline.

Expected layout:
    data_dir/features.npz
    data_dir/labels.npz
    model_dir/checkpoints/best_model.pth
    model_dir/config.json             (optional)
    model_dir/norm_params.json        (optional)
    data_dir/norm_params.json         (optional)

Outputs go to data_dir/evaluation/<model_name>/:
    regression_metrics.csv
    threshold_metrics.csv
    overall_threshold_metrics.csv
    evaluation_summary.json
    sample_*.png

All values are already in mm/hr. No unit conversion is applied.
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

# Allow imports of local modules regardless of the working directory.
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.IR_UNET import SAIRUNet
except ImportError:
    try:
        from IR_UNET import SAIRUNet
    except ImportError as exc:
        raise ImportError(
            "Could not import SAIRUNet. Place IR_UNET.py either in a "
            "'scripts' subdirectory or alongside this file."
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "batch_size": 1,
    "thresholds_mm_per_hr": [1, 5, 15, 25, 50],
    "epsilon": 1e-8,
    "num_workers": 0,
    "use_gpu_if_available": True,
    "save_visualizations": True,
    "num_sample_plots": 3,
    "use_all_thresholds": True,
    "neighborhood_size": 5,
    "verbose": False,
    "cmap": "turbo",
    "plot_width_per_lead": 5.0,
    "plot_height": 10.0,
    "plot_dpi": 300,
    # Spacing between prediction frames. Used only to label lead times.
    "lead_time_step_minutes": 30,
}


# ============================================================
# METRICS
# ============================================================
class PrecipitationMetrics:
    """Threshold-based categorical precipitation metrics."""

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

    def __init__(self, params, epsilon=1e-8):
        self.params = params
        self.epsilon = epsilon

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


def load_normalizer(data_dir, model_dir):
    """Look for norm_params.json in the data directory, then the model directory."""
    for candidate in (Path(data_dir) / "norm_params.json",
                      Path(model_dir) / "norm_params.json"):
        if candidate.exists():
            with open(candidate) as f:
                params = json.load(f)
            print(f"Normalization parameters loaded from {candidate}")
            return DataNormalizer(params)
    print("No normalization parameters found; features used as-is.")
    return None


# ============================================================
# DATASET
# ============================================================
class CaseStudyDataset(Dataset):
    """Loads case-study features and labels from NPZ files."""

    def __init__(self, data_path, normalizer=None):
        data_path = Path(data_path)
        features_file = data_path / "features.npz"
        labels_file = data_path / "labels.npz"

        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"features.npz or labels.npz missing in {data_path}"
            )

        features = np.load(features_file)["features"]
        labels = np.load(labels_file)["labels"]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.maximum(features, 0)
        labels = np.maximum(labels, 0)

        if normalizer is not None:
            shape = features.shape
            flat = features.reshape(-1, *shape[2:])
            features = normalizer.transform_features(flat).reshape(shape)
            print("Features normalized using training parameters.")

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.from_numpy(self.labels[idx]))


# ============================================================
# MODEL LOADING
# ============================================================
def find_checkpoint(model_dir):
    candidates = [
        model_dir / "checkpoints" / "best_model.pth",
        model_dir / "checkpoints" / "final_model.pth",
        model_dir / "best_model.pth",
        model_dir / "final_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pth_files = sorted(model_dir.glob("**/*.pth"))
    if pth_files:
        return pth_files[0]

    raise FileNotFoundError(f"No checkpoint (.pth) found under {model_dir}")


def load_sairunet(model_dir, device):
    config_file = model_dir / "config.json"
    model_config = {}
    if config_file.exists():
        with open(config_file) as f:
            model_config = json.load(f)
        print(f"Model config loaded from {config_file}")
    else:
        print("No config.json found; using default architecture parameters.")

    in_ch = model_config.get("in_channels", 3)
    out_ch = model_config.get("out_channels", 6)
    n_filters = model_config.get("n_filters", 64)
    dropout = model_config.get("dropout", 0.05)
    kernel_size = model_config.get("kernel_size", 5)
    expand_ratio = model_config.get("expand_ratio", 6)

    model = SAIRUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        n_filters=n_filters,
        dropout=dropout,
        kernel_size=kernel_size,
        expand_ratio=expand_ratio,
    )

    checkpoint_path = find_checkpoint(model_dir)
    print(f"Checkpoint: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    missing = len(model_state) - len(compatible)
    if missing:
        print(f"Warning: {missing} model keys not initialized from checkpoint.")

    model.load_state_dict(compatible, strict=False)
    model.to(device).eval()

    info = {
        "model_type": "SAIRUNet",
        "checkpoint": str(checkpoint_path),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "in_channels": in_ch,
        "out_channels": out_ch,
        "n_filters": n_filters,
        "dropout": dropout,
        "kernel_size": kernel_size,
        "expand_ratio": expand_ratio,
    }
    print(f"Model parameters: {info['total_parameters']:,}")
    return model, info


# ============================================================
# INFERENCE
# ============================================================
def run_inference(model, dataloader, device, normalizer=None):
    print("Running inference...")
    preds_list, targets_list = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Inference", leave=False):
            outputs = model(inputs.to(device))
            preds_list.append(outputs.float().cpu().numpy())
            targets_list.append(targets.float().cpu().numpy())

    elapsed = time.perf_counter() - start

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    if normalizer is not None:
        shape = preds.shape
        flat = preds.reshape(-1, *shape[2:])
        preds = normalizer.inverse_transform_predictions(flat).reshape(shape)
        print("Predictions inverse-transformed to physical units.")

    return preds, targets, elapsed


# ============================================================
# METRIC AGGREGATION
# ============================================================
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

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "NSE": nse,
        "KGE": kge,
        "Correlation": corr,
        "units": "mm/hr",
    }


def lead_time_labels(max_steps, step_minutes=30):
    if step_minutes == 60:
        return [f"{i + 1}h" for i in range(max_steps)]
    return [f"{(i + 1) * step_minutes}min" for i in range(max_steps)]


def threshold_metrics_by_lead(preds, targets, thresholds, epsilon, neighborhood_size,
                              step_minutes, verbose=False):
    n_leads = preds.shape[1]
    lead_times = lead_time_labels(n_leads, step_minutes)
    per_lead = {}
    accumulated = {thr: {m: [] for m in ("CSI", "POD", "FAR", "HSS", "GSS", "FSS")}
                   for thr in thresholds}

    for idx, lead_time in enumerate(lead_times):
        pred_slice = preds[:, idx, ...]
        target_slice = targets[:, idx, ...]

        per_threshold = {}
        for thr in thresholds:
            calc = PrecipitationMetrics(
                threshold=thr, epsilon=epsilon, neighborhood_size=neighborhood_size
            )
            metrics = calc.calculate_all(target_slice, pred_slice)
            per_threshold[thr] = metrics
            for m in accumulated[thr]:
                accumulated[thr][m].append(metrics[m])
        per_lead[lead_time] = per_threshold

        if verbose:
            m = per_threshold[thresholds[0]]
            print(f"  {lead_time}: CSI={m['CSI']:.3f}  POD={m['POD']:.3f}  "
                  f"FAR={m['FAR']:.3f}  HSS={m['HSS']:.3f}  "
                  f"GSS={m['GSS']:.3f}  FSS={m['FSS']:.3f}")

    overall = {
        thr: {m: float(np.mean(values)) if values else 0.0
              for m, values in accumulated[thr].items()}
        for thr in thresholds
    }
    return per_lead, overall, lead_times


# ============================================================
# PLOTTING
# ============================================================
def compute_limits(values, low=2, high=98):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, low), np.percentile(finite, high)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = finite.min(), finite.max()
    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01
    return float(vmin), float(vmax)


def plot_sample_predictions(preds, targets, lead_times, out_dir, num, units,
                            cmap, config):
    n_samples = preds.shape[0]
    if n_samples == 0:
        return

    idx_list = random.sample(range(n_samples), min(num, n_samples))
    n_leads = preds.shape[1]
    width = max(10, n_leads * config["plot_width_per_lead"])
    height = config["plot_height"]
    dpi = config["plot_dpi"]

    preds_clean = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targets_clean = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    for sample_idx in idx_list:
        fig, axes = plt.subplots(2, n_leads, figsize=(width, height), dpi=dpi)
        if n_leads == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"Sample {sample_idx} ({units})",
                     fontsize=14, fontweight="bold")

        for lt_idx in range(n_leads):
            label = lead_times[lt_idx] if lt_idx < len(lead_times) else f"Step {lt_idx + 1}"
            column = np.concatenate([
                preds_clean[sample_idx, lt_idx].flatten(),
                targets_clean[sample_idx, lt_idx].flatten(),
            ])
            vmin, vmax = compute_limits(column)

            ax = axes[0, lt_idx]
            im = ax.imshow(preds_clean[sample_idx, lt_idx],
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Pred - {label}", fontsize=10, fontweight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(units, fontsize=8)

            ax = axes[1, lt_idx]
            im = ax.imshow(targets_clean[sample_idx, lt_idx],
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"GT - {label}", fontsize=10, fontweight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(units, fontsize=8)

        fig.text(0.02, 0.75, "PREDICTED", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold")
        fig.text(0.02, 0.25, "GROUND TRUTH", va="center", rotation="vertical",
                 fontsize=12, fontweight="bold")

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
        plt.savefig(out_dir / f"sample_{sample_idx:03d}.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close()


# ============================================================
# OUTPUT
# ============================================================
def save_outputs(out_dir, result, model_info, data_dir, n_samples, step_minutes):
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result["regression_metrics"]]).to_csv(
        out_dir / "regression_metrics.csv", index=False)

    rows = []
    for lead_time, per_thr in result["threshold_metrics"].items():
        for thr, m in per_thr.items():
            rows.append({"lead_time": lead_time, "threshold_mm_hr": thr, **m})
    pd.DataFrame(rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    overall_rows = []
    for thr, m in result["overall_threshold_metrics"].items():
        overall_rows.append({"threshold_mm_hr": thr, **m})
    pd.DataFrame(overall_rows).to_csv(
        out_dir / "overall_threshold_metrics.csv", index=False)

    summary = {
        "model_info": model_info,
        "data_info": {
            "data_path": str(data_dir),
            "samples": n_samples,
            "image_size": list(result["preds"].shape[2:]),
            "lead_steps": len(result["lead_times"]),
            "lead_times": result["lead_times"],
            "lead_time_step_minutes": step_minutes,
            "units": "mm/hr",
            "unit_conversion_applied": False,
        },
        "performance": {
            "inference_time_seconds": result["inference_time"],
            "inference_time_per_sample_ms":
                result["inference_time"] / max(n_samples, 1) * 1000,
        },
        "regression_metrics": result["regression_metrics"],
        "threshold_metrics": result["threshold_metrics"],
        "overall_threshold_metrics": result["overall_threshold_metrics"],
        "thresholds_mm_per_hr": list(result["overall_threshold_metrics"].keys()),
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate(data_dir, model_dir, config):
    device = torch.device(
        "cuda" if config["use_gpu_if_available"] and torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    normalizer = load_normalizer(data_dir, model_dir)
    dataset = CaseStudyDataset(data_dir, normalizer=normalizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model, model_info = load_sairunet(model_dir, device)

    preds, targets, inference_time = run_inference(
        model, dataloader, device, normalizer
    )

    # Values are already mm/hr; no conversion applied.
    reg = regression_metrics(preds, targets)
    print("Regression metrics (mm/hr):")
    for k, v in reg.items():
        if k != "units":
            print(f"  {k}: {v:.6f}")

    thresholds = (
        config["thresholds_mm_per_hr"] if config["use_all_thresholds"] else [5.0]
    )

    per_lead, overall, lead_times = threshold_metrics_by_lead(
        preds, targets,
        thresholds=thresholds,
        epsilon=config["epsilon"],
        neighborhood_size=config["neighborhood_size"],
        step_minutes=config["lead_time_step_minutes"],
        verbose=config["verbose"],
    )

    result = {
        "regression_metrics": reg,
        "threshold_metrics": per_lead,
        "overall_threshold_metrics": overall,
        "lead_times": lead_times,
        "inference_time": inference_time,
        "preds": preds,
        "targets": targets,
    }

    out_dir = Path(data_dir) / "evaluation" / model_dir.name
    save_outputs(out_dir, result, model_info, data_dir, len(dataset),
                 config["lead_time_step_minutes"])

    if config["save_visualizations"] and len(preds) > 0:
        print("Generating sample plots...")
        plot_sample_predictions(
            preds, targets, lead_times, out_dir,
            num=config["num_sample_plots"],
            units="mm/hr",
            cmap=config["cmap"],
            config=config,
        )

    print(f"Results saved to {out_dir}")
    return result


# ============================================================
# ENTRY POINT
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


def main():
    print("SAIR-UNet evaluation pipeline")
    print("All metrics are reported in mm/hr; no unit conversion is applied.")

    data_dir = pick_directory("Select case study data directory")
    if data_dir is None:
        print("No data directory selected. Exiting.")
        return

    model_dir = pick_directory("Select SAIR-UNet model directory")
    if model_dir is None:
        print("No model directory selected. Exiting.")
        return

    print(f"Data directory:  {data_dir}")
    print(f"Model directory: {model_dir}")

    evaluate(data_dir, model_dir, CONFIG.copy())


if __name__ == "__main__":
    main()
