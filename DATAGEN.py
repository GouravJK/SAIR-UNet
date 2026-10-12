# -*- coding: utf-8 -*-
"""
Data generation for LR_UNET (3 inputs, 6 outputs).

Loads INSAT-3D HEM GeoTIFFs, resamples to the target grid, builds sliding
windows (3 input frames -> 6 output frames), and writes NPZ files for
train/val/test splits.

Data are used as-is. No outlier correction and no upper-bound clipping
are applied; the maximum value reflects the original data.
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
import xarray as xr
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Paths
    "base_path": r"path_for_dataset",
    "output_dir": r"output_path/Data/128_128",

    # Temporal selection
    "years": [2020, 2021, 2022, 2023],
    "months": [6, 7, 8],
    "temporal_resolution_minutes": 30,

    # Spatial configuration
    "target_spatial_resolution_km": None,   # If set, target_size is computed
    "target_size": (128, 128),
    "resample_method": "bilinear",          # bilinear | nearest | cubic

    # Sequence configuration
    "input_frames": 3,
    "output_frames": 6,

    # Split ratios
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 42,

    # Data range: only a lower floor is applied (negatives -> 0).
    "min_valid_precipitation": 0.0,

    # Output options
    "save_metadata": True,

    # Advanced
    "skip_time_gaps": True,
    "verbose": True,
    "filter_by_time_range": None,
    "file_extension": ".tif",
    "search_recursively": True,
}

assert CONFIG["train_ratio"] + CONFIG["val_ratio"] + CONFIG["test_ratio"] == 1.0
assert CONFIG["input_frames"] > 0 and CONFIG["output_frames"] > 0
if CONFIG["target_spatial_resolution_km"] is not None:
    CONFIG["target_size"] = None


# ============================================================
# FILENAME PARSING
# ============================================================
_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def extract_datetime_from_filename(filename):
    """Parse datetime from common INSAT-style filenames."""
    name = Path(filename).stem
    parts = name.split("_")

    # Pattern: 3RIMG_01JUN2020_0415_...
    if len(parts) >= 3:
        try:
            date_str, time_str = parts[1], parts[2]
            day = int(date_str[:2])
            month = _MONTH_ABBR.get(date_str[2:5].upper())
            year = int(date_str[5:])
            if month is not None:
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                return datetime(year, month, day, hour, minute)
        except (ValueError, IndexError):
            pass

    # YYYYMMDD_HHMM or YYYYMMDD_HHMMSS
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})?", name)
    if m:
        y, mo, d, h, mi, _ = m.groups()
        return datetime(int(y), int(mo), int(d), int(h), int(mi))

    # YYYY-MM-DD_HHMM
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})", name)
    if m:
        y, mo, d, h, mi = m.groups()
        return datetime(int(y), int(mo), int(d), int(h), int(mi))

    # Date only, default to midnight
    m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
    if m:
        y, mo, d = m.groups()
        return datetime(int(y), int(mo), int(d), 0, 0)

    raise ValueError(f"Cannot parse datetime from {filename}")


# ============================================================
# GEOTIFF HELPERS
# ============================================================
def get_image_metadata(img_path):
    """Read spatial metadata from a GeoTIFF."""
    with rasterio.open(img_path) as src:
        transform = src.transform
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        bounds = src.bounds
        return {
            "original_shape": (src.height, src.width),
            "pixel_width_degrees": pixel_width,
            "pixel_height_degrees": pixel_height,
            "spatial_resolution_approx_km": pixel_width * 111.0,
            "crs": src.crs.to_string() if src.crs else "Unknown",
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
        }


def calculate_target_size(original_shape, original_res_km, target_res_km):
    """Target dimensions for a requested spatial resolution."""
    h, w = original_shape
    scale = original_res_km / target_res_km
    return (int(h * scale), int(w * scale))


def load_geotiff(filepath, target_size, config):
    """Load a single GeoTIFF, apply the lower floor, and resample if needed."""
    try:
        dt = extract_datetime_from_filename(filepath)
    except ValueError:
        return None, None

    if dt.month not in config["months"]:
        return None, None

    with rasterio.open(filepath) as src:
        img = src.read(1).astype(np.float32)
        if src.nodata is not None:
            img = np.where(img == src.nodata, 0.0, img)

    img = np.nan_to_num(img, nan=0.0)
    img = np.maximum(img, config["min_valid_precipitation"])

    if img.shape != tuple(target_size):
        order = {"nearest": 0, "bilinear": 1, "cubic": 3}.get(config["resample_method"], 1)
        factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        img = zoom(img, factors, order=order)

    return img, dt


def find_geotiffs(base_path, years, months, config):
    """Find all GeoTIFFs for the given years and months."""
    extensions = [config["file_extension"]]
    if config["file_extension"].lower() == ".tif":
        extensions += [".tiff", ".TIF", ".TIFF"]

    all_files = set()
    for year in years:
        year_path = Path(base_path) / str(year)
        if not year_path.exists():
            print(f"Warning: {year_path} not found")
            continue

        for ext in extensions:
            if config["search_recursively"]:
                all_files.update(year_path.rglob(f"*{ext}"))
            else:
                all_files.update(year_path.glob(f"*{ext}"))
                for month in months:
                    for fmt in ("numeric", "short", "long"):
                        sub = year_path / get_month_folder_name(month, fmt)
                        if sub.exists():
                            all_files.update(sub.glob(f"*{ext}"))

    filtered = []
    for f in all_files:
        try:
            dt = extract_datetime_from_filename(f)
            if dt.month in months:
                filtered.append((f, dt))
        except ValueError:
            continue

    filtered.sort(key=lambda x: x[1])
    print(f"Found {len(filtered)} files for months {months}")
    return filtered


def get_month_folder_name(month, fmt):
    if fmt == "numeric":
        return f"{month:02d}"
    if fmt == "short":
        return datetime(2000, month, 1).strftime("%b").upper()
    if fmt == "long":
        return datetime(2000, month, 1).strftime("%B")
    return f"{month:02d}"


def load_all_years(base_path, years, config):
    """Load and stack all selected GeoTIFFs."""
    files = find_geotiffs(base_path, years, config["months"], config)
    if not files:
        raise RuntimeError(f"No GeoTIFFs found under {base_path}")

    geo_meta = get_image_metadata(files[0][0])
    target_size = config["target_size"]

    if config["target_spatial_resolution_km"] is not None:
        target_size = calculate_target_size(
            geo_meta["original_shape"],
            geo_meta["spatial_resolution_approx_km"],
            config["target_spatial_resolution_km"],
        )
        print(f"Computed target size: {target_size[0]} x {target_size[1]}")

    images, times = [], []
    for filepath, dt in files:
        img, _ = load_geotiff(filepath, target_size, config)
        if img is not None:
            images.append(img)
            times.append(dt)

    if not images:
        raise RuntimeError("No images loaded.")

    data = np.stack(images, axis=0)
    times = np.array(times)
    print(f"Loaded {len(images)} images, shape {data.shape}")
    print(f"Time range: {times[0]} to {times[-1]}")

    return data, times, geo_meta, target_size


# ============================================================
# WINDOW / SPLIT PROCESSING
# ============================================================
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.input_frames = config["input_frames"]
        self.output_frames = config["output_frames"]
        self.window_size = self.input_frames + self.output_frames

    def create_windows(self, data, times):
        """Slice the time series into (input, output) pairs."""
        n_windows = data.shape[0] - self.window_size + 1
        breaks = self._find_time_breaks(times) if self.config["skip_time_gaps"] else []

        features, labels, meta = [], [], []
        skipped = 0

        for i in range(n_windows):
            if any(i <= b < i + self.window_size - 1 for b in breaks):
                skipped += 1
                continue

            window = data[i:i + self.window_size]
            if window.shape[0] != self.window_size:
                continue

            features.append(window[:self.input_frames])
            labels.append(window[self.input_frames:])
            meta.append({
                "start_idx": i,
                "end_idx": i + self.window_size - 1,
                "start_time": str(times[i]),
                "end_time": str(times[i + self.window_size - 1]),
            })

        features = np.array(features)
        labels = np.array(labels)
        print(f"Created {len(features)} windows (skipped {skipped})")
        print(f"Features: {features.shape}  Labels: {labels.shape}")
        return features, labels, meta

    def _find_time_breaks(self, times):
        """Indices where the time delta deviates from the expected resolution."""
        expected = self.config["temporal_resolution_minutes"]
        diffs = np.array([
            (times[i] - times[i - 1]).total_seconds() / 60.0
            for i in range(1, len(times))
        ])
        return np.where(np.abs(diffs - expected) > 1)[0]

    def split(self, features, labels, meta):
        n = len(features)
        idx = np.arange(n)

        train_idx, temp_idx = train_test_split(
            idx,
            test_size=self.config["val_ratio"] + self.config["test_ratio"],
            random_state=self.config["random_seed"],
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=self.config["test_ratio"] / (self.config["val_ratio"] + self.config["test_ratio"]),
            random_state=self.config["random_seed"],
            shuffle=True,
        )

        print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

        def pack(idxs):
            return {
                "features": features[idxs],
                "labels": labels[idxs],
                "metadata": [meta[i] for i in idxs],
                "indices": idxs,
            }

        return {"train": pack(train_idx), "val": pack(val_idx), "test": pack(test_idx)}

    def save(self, splits, output_dir, geo_meta, original_shape, spatial_resolution):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, split in splits.items():
            split_dir = out / name
            split_dir.mkdir(exist_ok=True)

            np.savez_compressed(split_dir / "features.npz", features=split["features"])
            np.savez_compressed(split_dir / "labels.npz", labels=split["labels"])

            if self.config["save_metadata"]:
                with open(split_dir / "metadata.json", "w") as f:
                    json.dump({
                        "split_name": name,
                        "total_windows": len(split["features"]),
                        "input_frames": self.input_frames,
                        "output_frames": self.output_frames,
                        "feature_shape": list(split["features"].shape),
                        "label_shape": list(split["labels"].shape),
                        "sample_indices": split["indices"].tolist(),
                        "original_shape": original_shape,
                        "spatial_resolution": spatial_resolution,
                    }, f, indent=2)

                with open(split_dir / "window_metadata.json", "w") as f:
                    json.dump(split["metadata"], f, indent=2)

            print(f"{name}: features {split['features'].shape}, labels {split['labels'].shape}")

        with open(out / "complete_config.json", "w") as f:
            json.dump(self.config, f, indent=4)


# ============================================================
# TIME-OF-DAY FILTER
# ============================================================
def filter_by_time_of_day(data, times, config):
    start_hour, end_hour = config["filter_by_time_range"]
    keep = [
        i for i, t in enumerate(times)
        if hasattr(t, "hour") and start_hour <= t.hour < end_hour
    ]
    if not keep:
        print(f"No samples in time range {start_hour}:00-{end_hour}:00")
        return data, times
    print(f"Time filter kept {len(keep)}/{len(data)} samples")
    return data[keep], times[keep]


# ============================================================
# PYTORCH DATASET
# ============================================================
class PrecipitationDataset(Dataset):
    """Loads preprocessed precipitation windows from NPZ files."""

    def __init__(self, data_dir, split="train"):
        split_path = Path(data_dir) / split
        features_file = split_path / "features.npz"
        labels_file = split_path / "labels.npz"

        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"NPZ files not found in {split_path}")

        self.features = np.load(features_file)["features"].astype(np.float32)
        self.labels = np.load(labels_file)["labels"].astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])


# ============================================================
# DIAGNOSTICS
# ============================================================
def diagnose_directory_structure(base_path, years, months):
    """Print a summary of the directory layout to help with configuration."""
    print("=" * 60)
    print("DIAGNOSING DIRECTORY STRUCTURE")
    print("=" * 60)

    base = Path(base_path)
    if not base.exists():
        print(f"Base path not found: {base_path}")
        return

    month_formats = {
        "numeric": [f"{m:02d}" for m in months],
        "short": [datetime(2000, m, 1).strftime("%b").upper() for m in months],
        "long": [datetime(2000, m, 1).strftime("%B") for m in months],
    }

    found_any = False
    for year in years:
        year_path = base / str(year)
        if not year_path.exists():
            print(f"Year {year}: not found")
            continue

        print(f"Year {year}: found")
        for fmt_name, names in month_formats.items():
            for name in names:
                sub = year_path / name
                if sub.exists():
                    tifs = list(sub.glob("*.tif")) + list(sub.glob("*.tiff"))
                    if tifs:
                        print(f"  {fmt_name} '{name}': {len(tifs)} files")
                        found_any = True

    if not found_any:
        print("\nNo month subdirectories with GeoTIFFs detected.")
        print("Check base_path, folder names, or try search_recursively=True.")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    config = CONFIG.copy()

    print("=" * 60)
    print("DATA GENERATION PIPELINE")
    print("=" * 60)
    print(f"Input frames: {config['input_frames']}  Output frames: {config['output_frames']}")
    print(f"Years: {config['years']}  Months: {config['months']}")
    print("=" * 60)

    data, times, geo_meta, target_size = load_all_years(
        config["base_path"], config["years"], config
    )

    if config["target_spatial_resolution_km"] is not None:
        config["target_size"] = target_size

    original_shape = geo_meta["original_shape"]
    spatial_resolution = {
        "pixel_width_degrees": geo_meta["pixel_width_degrees"],
        "pixel_height_degrees": geo_meta["pixel_height_degrees"],
        "approx_km_per_pixel": geo_meta["spatial_resolution_approx_km"],
        "crs": geo_meta["crs"],
        "bounds": geo_meta["bounds"],
    }

    print("\nSpatial specifications")
    print(f"  Original: {original_shape[0]} x {original_shape[1]} at "
          f"{spatial_resolution['approx_km_per_pixel']:.2f} km/pixel")
    print(f"  Target:   {config['target_size'][0]} x {config['target_size'][1]}")

    if config["filter_by_time_range"] is not None:
        data, times = filter_by_time_of_day(data, times, config)

    print("\nDataset statistics")
    data_min = float(data.min())
    data_max = float(data.max())
    data_mean = float(data.mean())
    data_std = float(data.std())
    percentiles = {p: float(np.percentile(data, p)) for p in (90, 95, 99)}
    print(f"  Range: [{data_min:.4f}, {data_max:.4f}]")
    print(f"  Mean: {data_mean:.4f}  Std: {data_std:.4f}")
    for p, v in percentiles.items():
        print(f"  {p}th percentile: {v:.4f} mm")

    processor = DataProcessor(config)
    features, labels, meta = processor.create_windows(data, times)
    splits = processor.split(features, labels, meta)
    processor.save(splits, config["output_dir"], geo_meta, original_shape, spatial_resolution)

    stats = {
        "original_image_specs": {
            "shape": original_shape,
            "pixel_size_degrees": spatial_resolution["pixel_width_degrees"],
            "pixel_size_km": spatial_resolution["approx_km_per_pixel"],
            "crs": spatial_resolution["crs"],
            "bounds": spatial_resolution["bounds"],
        },
        "resampled_specs": {
            "target_shape": config["target_size"],
            "resampling_factor": original_shape[0] / config["target_size"][0],
            "new_pixel_size_degrees": spatial_resolution["pixel_width_degrees"]
                                     * (original_shape[0] / config["target_size"][0]),
            "new_pixel_size_km": spatial_resolution["approx_km_per_pixel"]
                                 * (original_shape[0] / config["target_size"][0]),
        },
        "data_statistics": {
            "min_mm": data_min,
            "max_mm": data_max,
            "mean_mm": data_mean,
            "std_mm": data_std,
            "percentiles_mm": percentiles,
        },
        "window_configuration": {
            "input_frames": config["input_frames"],
            "output_frames": config["output_frames"],
            "window_size": config["input_frames"] + config["output_frames"],
            "total_windows_created": len(features),
        },
        "total_time_steps": data.shape[0],
        "temporal_resolution_minutes": config["temporal_resolution_minutes"],
        "years_used": config["years"],
        "months_used": config["months"],
    }

    stats_path = Path(config["output_dir"]) / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"\nStatistics saved to {stats_path}")
    print(f"Data written to {config['output_dir']}")

    return processor, splits


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    diagnose_directory_structure(CONFIG["base_path"], CONFIG["years"], CONFIG["months"])

    print("\nStarting data generation...")
    try:
        main()

        print("\nVerifying dataset loader...")
        train_ds = PrecipitationDataset(CONFIG["output_dir"], split="train")
        print(f"Train samples: {len(train_ds)}")

        if len(train_ds) > 0:
            x, y = train_ds[0]
            print(f"Sample input shape:  {x.shape}")
            print(f"Sample output shape: {y.shape}")
            print(f"Input range:  [{x.min():.2f}, {x.max():.2f}]")
            print(f"Output range: [{y.min():.2f}, {y.max():.2f}]")
            print(f"Input non-zero pixels:  {(x > 0).sum().item() / x.numel() * 100:.2f}%")
            print(f"Output non-zero pixels: {(y > 0).sum().item() / y.numel() * 100:.2f}%")

    except Exception as e:
        print(f"\nError: {e}")
        print("Check the directory diagnostics above, then verify:")
        print("  - base_path and month folder naming")
        print("  - file extension")
        print("  - search_recursively setting")
        raise