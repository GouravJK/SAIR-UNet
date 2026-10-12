# -*- coding: utf-8 -*-
"""
Case Study Data Generation Pipeline for LR_UNET (3 inputs, 2 outputs)
- Loads INSAT-3D HEM GeoTIFFs for a specific continuous time period
- Ensures temporal continuity and identifies gaps
- Resamples to user-specified spatial resolution
- Creates sliding windows: 3 input frames → 2 output frames
- Maintains chronological order (no shuffling/splitting)
- Designed for validation/case study scenarios
- Detects and replaces outlier values with local mean of neighbors
- Records original image shape and spatial resolution
"""

import numpy as np
import xarray as xr
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
import warnings
import json
from datetime import datetime, timedelta
import rasterio
import re
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION DICTIONARY - USER SETTINGS
# ============================================================

CASE_STUDY_CONFIG = {
    # ===== DATA PATHS =====
    'base_path': r'D:\PhD Work III\INSAT 3DR HEM',
    'output_dir': r'D:\PhD Work III\INSAT 3DR HEM/case_study/cs_13_jun_24',
    
    # ===== TEMPORAL SELECTION =====
    'start_datetime': '2024-06-13 18:30',
    'end_datetime': '2024-06-15 18:30',
    'temporal_resolution_minutes': 30,
    
    # ===== SPATIAL CONFIGURATION =====
    'target_spatial_resolution_km': None,
    'target_size': (128, 128),
    'resample_method': 'bilinear',
    
    # ===== SEQUENCE CONFIGURATION =====
    'input_frames': 3,
    'output_frames': 6,
    
    # ===== OUTLIER HANDLING =====
    'fix_outliers': False,
    'outlier_threshold': 100.0,
    'adaptive_threshold': False,
    'outlier_percentile': 99.5,
    
    # ===== CONTINUITY CHECKING =====
    'max_gap_minutes': 60,
    'flag_continuity_breaks': True,
    'create_discontinuity_report': True,
    
    # ===== DATA QUALITY =====
    'min_valid_precipitation': 0.0,
    'max_valid_precipitation': 500.0,
    
    # ===== OUTPUT OPTIONS =====
    'save_npz': True,
    'save_pt': False,
    'save_binary_versions': False,
    'save_metadata': True,
    'generate_report': True,
    
    # ===== BINARY THRESHOLD =====
    'binary_threshold_percentile': 95,
    
    # ===== VISUALIZATION =====
    'create_visualization': True,
    'num_viz_samples': 10,
    'viz_cmap': 'turbo',
    'rain_threshold_mm': 5,
    
    # ===== ADVANCED OPTIONS =====
    'verbose': True,
    'preserve_original_nodata': False,
}

assert CASE_STUDY_CONFIG['input_frames'] > 0 and CASE_STUDY_CONFIG['output_frames'] > 0, "Input and output frames must be positive"
window_size = CASE_STUDY_CONFIG['input_frames'] + CASE_STUDY_CONFIG['output_frames']


# ============================================================
# 0. UTILITY FUNCTIONS
# ============================================================

def convert_to_json_serializable(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M')
    else:
        return obj

def parse_datetime(datetime_str):
    """Parse datetime string to datetime object."""
    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')

def extract_datetime_from_filename(filename):
    """Parse datetime from INSAT-3D filename format."""
    name = Path(filename).stem
    parts = name.split('_')
    if len(parts) >= 3:
        date_str = parts[1]
        time_str = parts[2]
        day = int(date_str[:2])
        month_abbr = date_str[2:5]
        year = int(date_str[5:])
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month = month_map.get(month_abbr.upper())
        if month is None:
            raise ValueError(f"Unknown month: {month_abbr}")
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        return datetime(year, month, day, hour, minute)
    else:
        raise ValueError(f"Cannot parse datetime from {filename}")

def get_image_metadata(img_path):
    """Extract metadata from a GeoTIFF file."""
    with rasterio.open(img_path) as src:
        transform = src.transform
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        crs = src.crs.to_string() if src.crs else "Unknown"
        bounds = src.bounds
        
        metadata = {
            'original_shape': (int(src.height), int(src.width)),
            'pixel_width_degrees': float(pixel_width),
            'pixel_height_degrees': float(pixel_height),
            'spatial_resolution_approx_km': float(pixel_width * 111),
            'crs': crs,
            'bounds': {
                'left': float(bounds.left),
                'bottom': float(bounds.bottom),
                'right': float(bounds.right),
                'top': float(bounds.top)
            },
            'nodata_value': float(src.nodata) if src.nodata is not None else None
        }
        return metadata

def replace_outliers_with_local_mean(image, threshold=100.0, kernel_size=3):
    """Replace values > threshold with the local mean of neighboring pixels."""
    from scipy.signal import convolve2d
    
    image = image.astype(np.float32)
    outlier_mask = image > threshold
    num_outliers = int(np.sum(outlier_mask))
    
    if num_outliers == 0:
        return image, outlier_mask, 0
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_sum = convolve2d(image, kernel, mode='same', boundary='symm')
    kernel_count = convolve2d(np.ones_like(image), kernel, mode='same', boundary='symm')
    local_mean = local_sum / kernel_count
    
    corrected_image = image.copy()
    corrected_image[outlier_mask] = local_mean[outlier_mask]
    
    return corrected_image, outlier_mask, num_outliers

def replace_outliers_with_adaptive_threshold(image, percentile_threshold=99.5, kernel_size=3):
    """Replace extreme outliers using adaptive threshold based on percentile."""
    from scipy.signal import convolve2d
    
    image = image.astype(np.float32)
    positive_values = image[image > 0]
    if len(positive_values) > 0:
        threshold = float(np.percentile(positive_values, percentile_threshold))
    else:
        threshold = 500.0
    
    outlier_mask = image > threshold
    num_outliers = int(np.sum(outlier_mask))
    
    if num_outliers == 0:
        return image, outlier_mask, 0
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_sum = convolve2d(image, kernel, mode='same', boundary='symm')
    kernel_count = convolve2d(np.ones_like(image), kernel, mode='same', boundary='symm')
    local_mean = local_sum / kernel_count
    
    corrected_image = image.copy()
    corrected_image[outlier_mask] = local_mean[outlier_mask]
    
    return corrected_image, outlier_mask, num_outliers

def check_temporal_continuity(times, expected_interval_minutes, max_gap_minutes):
    """Check temporal continuity of time series."""
    if len(times) < 2:
        return True, [], {}
    
    times_sorted = sorted(times)
    gaps = []
    total_gaps = 0
    total_missing_timesteps = 0
    
    for i in range(1, len(times_sorted)):
        delta = (times_sorted[i] - times_sorted[i-1]).total_seconds() / 60
        expected_delta = expected_interval_minutes
        
        if abs(delta - expected_delta) > 1:
            missing_steps = int(round(delta / expected_delta)) - 1
            gap_info = {
                'index': int(i),
                'start_time': times_sorted[i-1].strftime('%Y-%m-%d %H:%M'),
                'end_time': times_sorted[i].strftime('%Y-%m-%d %H:%M'),
                'gap_minutes': float(delta),
                'expected_interval_minutes': float(expected_delta),
                'missing_timesteps': int(max(0, missing_steps)),
                'is_significant': bool(delta > max_gap_minutes)
            }
            gaps.append(gap_info)
            total_gaps += 1
            total_missing_timesteps += missing_steps
    
    continuity_report = {
        'total_timesteps': int(len(times)),
        'expected_interval_minutes': float(expected_interval_minutes),
        'time_range': {
            'start': times_sorted[0].strftime('%Y-%m-%d %H:%M'),
            'end': times_sorted[-1].strftime('%Y-%m-%d %H:%M'),
            'duration_hours': float((times_sorted[-1] - times_sorted[0]).total_seconds() / 3600)
        },
        'gaps_found': int(total_gaps),
        'total_missing_timesteps': int(total_missing_timesteps),
        'is_fully_continuous': bool(len(gaps) == 0),
        'has_significant_gaps': bool(any(g['is_significant'] for g in gaps)),
        'gaps': gaps
    }
    
    return len(gaps) == 0, gaps, continuity_report

def calculate_target_size_from_resolution(original_shape, original_resolution_km, target_resolution_km):
    """Calculate target image dimensions based on desired spatial resolution."""
    original_height, original_width = original_shape
    scale_factor = original_resolution_km / target_resolution_km
    target_height = int(original_height * scale_factor)
    target_width = int(original_width * scale_factor)
    return (target_height, target_width)


# ============================================================
# 1. CASE STUDY DATA GENERATOR
# ============================================================
class CaseStudyDataGenerator:
    """Generator for creating continuous case study datasets from GeoTIFF sequences."""
    
    def __init__(self, config):
        self.config = config
        self.start_dt = parse_datetime(config['start_datetime'])
        self.end_dt = parse_datetime(config['end_datetime'])
        self.window_size = config['input_frames'] + config['output_frames']
        self.input_frames = config['input_frames']
        self.output_frames = config['output_frames']
        
        self.outlier_stats = {
            'total_fixed': 0,
            'files_with_outliers': 0,
            'outlier_details': []
        }
        self.processing_stats = {}
        
    def find_and_load_files(self):
        """Find and load all GeoTIFF files within the specified time range."""
        base_path = Path(self.config['base_path'])
        
        if self.config['verbose']:
            print("\n" + "="*60)
            print("SEARCHING FOR DATA FILES")
            print("="*60)
            print(f"Time range: {self.config['start_datetime']} to {self.config['end_datetime']}")
        
        current = self.start_dt.replace(day=1, hour=0, minute=0)
        search_months = []
        while current <= self.end_dt:
            search_months.append((current.year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        if self.config['verbose']:
            print(f"Searching in months: {search_months}")
        
        all_files = []
        for year, month in search_months:
            month_name = datetime(year, month, 1).strftime('%b').upper()
            year_path = base_path / str(year) / month_name
            
            if not year_path.exists():
                if self.config['verbose']:
                    print(f"  Warning: {year_path} not found")
                continue
            
            tif_files = sorted(year_path.glob('*.tif'))
            for fpath in tif_files:
                try:
                    dt = extract_datetime_from_filename(fpath)
                    if self.start_dt <= dt <= self.end_dt:
                        all_files.append((dt, fpath))
                except Exception as e:
                    if self.config['verbose']:
                        print(f"  Skipping {fpath.name}: {e}")
                    continue
        
        if not all_files:
            raise ValueError(f"No files found in the specified time range")
        
        all_files.sort(key=lambda x: x[0])
        
        if self.config['verbose']:
            print(f"\nFound {len(all_files)} files in time range")
            print(f"First file: {all_files[0][1].name} ({all_files[0][0]})")
            print(f"Last file:  {all_files[-1][1].name} ({all_files[-1][0]})")
        
        return all_files
    
    def load_and_process_images(self, file_list):
        """Load and process all images, handling outliers and resampling."""
        if self.config['verbose']:
            print("\n" + "="*60)
            print("LOADING AND PROCESSING IMAGES")
            print("="*60)
        
        first_file = file_list[0][1]
        geo_metadata = get_image_metadata(first_file)
        
        if self.config['verbose']:
            print(f"Original image shape: {geo_metadata['original_shape']}")
            print(f"Spatial resolution: {geo_metadata['pixel_width_degrees']:.6f}° "
                  f"(≈ {geo_metadata['spatial_resolution_approx_km']:.2f} km/pixel)")
        
        target_size = self.config['target_size']
        if self.config['target_spatial_resolution_km'] is not None:
            target_size = calculate_target_size_from_resolution(
                geo_metadata['original_shape'],
                geo_metadata['spatial_resolution_approx_km'],
                self.config['target_spatial_resolution_km']
            )
        
        times = []
        data_list = []
        load_errors = []
        
        for i, (dt, fpath) in enumerate(file_list):
            try:
                with rasterio.open(fpath) as src:
                    img = src.read(1).astype(np.float32)
                    
                    if src.nodata is not None and not self.config['preserve_original_nodata']:
                        img = np.where(img == src.nodata, 0.0, img)
                    
                    img = np.nan_to_num(img, nan=0.0)
                    img = np.clip(img, self.config['min_valid_precipitation'], 
                                 self.config['max_valid_precipitation'])
                    
                    if self.config['fix_outliers']:
                        if self.config['adaptive_threshold']:
                            img_corrected, mask, num_fixed = replace_outliers_with_adaptive_threshold(
                                img, self.config['outlier_percentile'], kernel_size=3
                            )
                        else:
                            img_corrected, mask, num_fixed = replace_outliers_with_local_mean(
                                img, self.config['outlier_threshold'], kernel_size=3
                            )
                        
                        if num_fixed > 0:
                            self.outlier_stats['total_fixed'] += num_fixed
                            self.outlier_stats['files_with_outliers'] += 1
                            self.outlier_stats['outlier_details'].append({
                                'file': fpath.name,
                                'datetime': str(dt),
                                'outliers_fixed': int(num_fixed),
                                'percentage': float(num_fixed / img.size * 100)
                            })
                            img = img_corrected
                    
                    H, W = img.shape
                    th, tw = target_size
                    if (H, W) != (th, tw):
                        method_map = {'bilinear': 1, 'nearest': 0, 'cubic': 3}
                        zoom_factors = (th/H, tw/W)
                        img = zoom(img, zoom_factors, order=method_map.get(self.config['resample_method'], 1))
                    
                    data_list.append(img)
                    times.append(dt)
                    
                    if self.config['verbose'] and (i % 10 == 0 or i == len(file_list) - 1):
                        print(f"  Processed {i+1}/{len(file_list)}: {fpath.name}")
                        
            except Exception as e:
                load_errors.append({'file': fpath.name, 'error': str(e)})
                if self.config['verbose']:
                    print(f"  Error loading {fpath.name}: {e}")
        
        if not data_list:
            raise RuntimeError("No images could be loaded successfully")
        
        data = np.stack(data_list, axis=0)
        times = np.array(times)
        
        if self.config['verbose']:
            print(f"\nSuccessfully loaded {len(data_list)} images")
            print(f"Data shape: {data.shape}")
            if self.outlier_stats['total_fixed'] > 0:
                print(f"Outliers fixed: {self.outlier_stats['total_fixed']} in {self.outlier_stats['files_with_outliers']} files")
        
        self.processing_stats['load_errors'] = load_errors
        self.processing_stats['total_files_loaded'] = len(data_list)
        self.processing_stats['total_files_found'] = len(file_list)
        
        return data, times, geo_metadata, target_size
    
    def check_continuity_and_report(self, times):
        """Check temporal continuity and generate report."""
        if self.config['verbose']:
            print("\n" + "="*60)
            print("TEMPORAL CONTINUITY ANALYSIS")
            print("="*60)
        
        is_continuous, gaps, continuity_report = check_temporal_continuity(
            times, 
            self.config['temporal_resolution_minutes'],
            self.config['max_gap_minutes']
        )
        
        if self.config['verbose']:
            print(f"Data is {'fully continuous' if is_continuous else 'NOT fully continuous'}")
            print(f"Time range: {continuity_report['time_range']['start']} to {continuity_report['time_range']['end']}")
            print(f"Duration: {continuity_report['time_range']['duration_hours']:.1f} hours")
            print(f"Timesteps: {continuity_report['total_timesteps']}")
            print(f"Gaps found: {continuity_report['gaps_found']}")
            
            if gaps:
                print("\nDetailed gaps:")
                for gap in gaps:
                    significance = "SIGNIFICANT" if gap['is_significant'] else "Minor"
                    print(f"  [{significance}] {gap['start_time']} → {gap['end_time']}: "
                          f"{gap['gap_minutes']:.0f} min gap ({gap['missing_timesteps']} missing steps)")
        
        return is_continuous, continuity_report
    
    def create_continuous_windows(self, data, times, continuity_report):
        """Create sliding windows, respecting temporal continuity."""
        if self.config['verbose']:
            print("\n" + "="*60)
            print("CREATING CONTINUOUS WINDOWS")
            print("="*60)
        
        T = data.shape[0]
        n_windows = T - self.window_size + 1
        
        if n_windows <= 0:
            raise ValueError(f"Not enough timesteps ({T}) to create windows (need {self.window_size})")
        
        break_points = set()
        if not continuity_report.get('is_fully_continuous', False):
            for gap in continuity_report.get('gaps', []):
                if gap['is_significant'] or self.config['flag_continuity_breaks']:
                    break_points.add(gap['index'])
        
        features = []
        labels = []
        metadata = []
        skipped_windows = []
        window_id = 0
        
        for i in range(n_windows):
            window_end = i + self.window_size
            
            crosses_break = False
            for bp in break_points:
                if i < bp <= window_end:
                    crosses_break = True
                    break
            
            if crosses_break:
                skipped_windows.append({
                    'window_start_idx': int(i),
                    'window_end_idx': int(window_end - 1),
                    'start_time': str(times[i]),
                    'end_time': str(times[window_end - 1]),
                    'reason': 'Crosses temporal gap'
                })
                continue
            
            window = data[i:window_end]
            if window.shape[0] != self.window_size:
                continue
            
            features.append(window[:self.input_frames])
            labels.append(window[self.input_frames:])
            
            meta = {
                'window_id': int(window_id),
                'start_idx': int(i),
                'end_idx': int(window_end - 1),
                'input_times': [str(times[t]) for t in range(i, i + self.input_frames)],
                'output_times': [str(times[t]) for t in range(i + self.input_frames, window_end)]
            }
            metadata.append(meta)
            window_id += 1
        
        features = np.array(features) if features else np.array([])
        labels = np.array(labels) if labels else np.array([])
        
        if self.config['verbose']:
            print(f"Created {len(features)} continuous windows")
            print(f"Skipped {len(skipped_windows)} windows (crossing temporal gaps)")
            if len(features) > 0:
                print(f"Features shape: {features.shape}")
                print(f"Labels shape: {labels.shape}")
        
        self.processing_stats['total_windows_created'] = int(len(features))
        self.processing_stats['skipped_windows'] = skipped_windows
        
        return features, labels, metadata
    
    def create_binary_versions(self, features, labels, threshold_mm):
        """Threshold precipitation values (>= threshold → 1, else 0)."""
        features_bin = (features >= threshold_mm).astype(np.float32)
        labels_bin = (labels >= threshold_mm).astype(np.float32)
        return features_bin, labels_bin
    
    def visualize_sample_windows(self, features, labels, times_metadata, output_dir, num_samples=5):
        """Create visualizations of sample windows with precipitation maps."""
        if len(features) == 0:
            print("No windows to visualize")
            return
        
        viz_dir = Path(output_dir) / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        cmap = self.config.get('viz_cmap', 'Spectral')
        
        n_total = len(features)
        if n_total <= num_samples:
            sample_indices = list(range(n_total))
        else:
            sample_indices = np.linspace(0, n_total - 1, num_samples, dtype=int)
        
        for idx in sample_indices:
            window_meta = times_metadata[idx]
            input_times = window_meta['input_times']
            output_times = window_meta['output_times']
            
            n_cols = max(self.input_frames, self.output_frames)
            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
            
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            
            vmin = float(min(features[idx].min(), labels[idx].min()))
            vmax = float(max(features[idx].max(), labels[idx].max()))
            if vmax == vmin:
                vmax = vmin + 0.1
            
            for t in range(self.input_frames):
                im = axes[0, t].imshow(features[idx, t], cmap=cmap, vmin=vmin, vmax=vmax)
                time_offset = t - (self.input_frames - 1)
                if time_offset == 0:
                    time_label = 't'
                else:
                    time_label = f't{time_offset}' if time_offset < 0 else f't+{time_offset}'
                axes[0, t].set_title(f'{time_label}\n{input_times[t][:16]}', fontsize=10)
                axes[0, t].axis('off')
            
            for t in range(self.input_frames, n_cols):
                axes[0, t].axis('off')
            
            for t in range(self.output_frames):
                im = axes[1, t].imshow(labels[idx, t], cmap=cmap, vmin=vmin, vmax=vmax)
                time_label = f't+{t + 1}'
                axes[1, t].set_title(f'{time_label}\n{output_times[t][:16]}', fontsize=10)
                axes[1, t].axis('off')
            
            for t in range(self.output_frames, n_cols):
                axes[1, t].axis('off')
            
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Precipitation (mm)', fontsize=12)
            
            fig.text(0.02, 0.75, 'Input\nFrames', va='center', ha='center', 
                    fontsize=12, fontweight='bold', rotation=90)
            fig.text(0.02, 0.25, 'Output\nFrames', va='center', ha='center', 
                    fontsize=12, fontweight='bold', rotation=90)
            
            plt.suptitle(f'Window {window_meta["window_id"]}\n'
                        f'Input: {input_times[0][:16]} → {input_times[-1][:16]} | '
                        f'Output: {output_times[0][:16]} → {output_times[-1][:16]}', 
                        fontsize=12, y=0.98)
            
            plt.tight_layout(rect=[0.05, 0, 0.90, 0.95])
            
            viz_path = viz_dir / f'window_{window_meta["window_id"]:04d}.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.config['verbose']:
                print(f"  Saved visualization: {viz_path.name}")
        
        # Create summary line plots
        self._create_summary_visualization(features, labels, times_metadata, viz_dir)
    
    def _create_summary_visualization(self, features, labels, times_metadata, viz_dir):
        """Create separate summary line plots showing statistics across all windows."""
        if len(features) == 0:
            return
        
        # Get user-defined rain threshold from config
        rain_threshold = self.config.get('rain_threshold_mm', 0.1)
        
        # Parse times for x-axis
        window_times = []
        for meta in times_metadata:
            try:
                dt = datetime.strptime(meta['input_times'][0][:16], '%Y-%m-%d %H:%M')
                window_times.append(dt)
            except:
                window_times.append(len(window_times))
        
        # ============================================================
        # 1. MEAN PRECIPITATION OVER TIME
        # ============================================================
        fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
        
        window_means_input = [features[i].mean() for i in range(len(features))]
        window_means_output = [labels[i].mean() for i in range(len(labels))]
        
        ax1.plot(window_times, window_means_input, 'b-', linewidth=1, alpha=0.7, label='Input Mean')
        ax1.plot(window_times, window_means_output, 'r-', linewidth=1, alpha=0.7, label='Output Mean')
        ax1.scatter(window_times, window_means_input, c='blue', s=20, alpha=0.5)
        ax1.scatter(window_times, window_means_output, c='red', s=20, alpha=0.5)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Mean Precipitation (mm)', fontsize=12)
        ax1.set_title(f'Mean Precipitation per Window\n({len(features)} windows)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        if len(window_times) > 0 and isinstance(window_times[0], datetime):
            ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M'))
            ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add vertical lines at day boundaries
            unique_days = set([dt.date() for dt in window_times])
            for day in unique_days:
                day_start = datetime.combine(day, datetime.min.time())
                ax1.axvline(x=day_start, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        mean_path = viz_dir / 'summary_mean_precipitation.png'
        plt.savefig(mean_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config['verbose']:
            print(f"  Saved mean precipitation plot: {mean_path.name}")
        
        # ============================================================
        # 2. MAXIMUM PRECIPITATION OVER TIME
        # ============================================================
        fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
        
        window_max_input = [features[i].max() for i in range(len(features))]
        window_max_output = [labels[i].max() for i in range(len(labels))]
        
        ax2.plot(window_times, window_max_input, 'b-', linewidth=1, alpha=0.7, label='Input Max')
        ax2.plot(window_times, window_max_output, 'r-', linewidth=1, alpha=0.7, label='Output Max')
        ax2.scatter(window_times, window_max_input, c='blue', s=20, alpha=0.5)
        ax2.scatter(window_times, window_max_output, c='red', s=20, alpha=0.5)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Maximum Precipitation (mm)', fontsize=12)
        ax2.set_title(f'Maximum Precipitation per Window\n({len(features)} windows)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        if len(window_times) > 0 and isinstance(window_times[0], datetime):
            ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M'))
            ax2.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add vertical lines at day boundaries
            unique_days = set([dt.date() for dt in window_times])
            for day in unique_days:
                day_start = datetime.combine(day, datetime.min.time())
                ax2.axvline(x=day_start, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        max_path = viz_dir / 'summary_max_precipitation.png'
        plt.savefig(max_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config['verbose']:
            print(f"  Saved max precipitation plot: {max_path.name}")
        
        # ============================================================
        # 3. RAIN AREA FRACTION OVER TIME
        # ============================================================
        fig3, ax3 = plt.subplots(1, 1, figsize=(16, 6))
        
        wet_fractions_input = [(features[i] > rain_threshold).mean() * 100 for i in range(len(features))]
        wet_fractions_output = [(labels[i] > rain_threshold).mean() * 100 for i in range(len(labels))]
        
        ax3.plot(window_times, wet_fractions_input, 'b-', linewidth=1, alpha=0.7, label=f'Input (> {rain_threshold} mm)')
        ax3.plot(window_times, wet_fractions_output, 'r-', linewidth=1, alpha=0.7, label=f'Output (> {rain_threshold} mm)')
        ax3.scatter(window_times, wet_fractions_input, c='blue', s=20, alpha=0.5)
        ax3.scatter(window_times, wet_fractions_output, c='red', s=20, alpha=0.5)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel(f'Rain Area (%)\n(Precipitation > {rain_threshold} mm)', fontsize=12)
        ax3.set_title(f'Fraction of Rainy Pixels per Window\n({len(features)} windows)', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        if len(window_times) > 0 and isinstance(window_times[0], datetime):
            ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M'))
            ax3.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add vertical lines at day boundaries
            unique_days = set([dt.date() for dt in window_times])
            for day in unique_days:
                day_start = datetime.combine(day, datetime.min.time())
                ax3.axvline(x=day_start, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        rain_area_path = viz_dir / 'summary_rain_area.png'
        plt.savefig(rain_area_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config['verbose']:
            print(f"  Saved rain area plot: {rain_area_path.name}")
        
        print(f"\nCreated 3 summary line plots in {viz_dir}/")
    
    def save_dataset(self, features, labels, metadata, output_dir, threshold_mm, 
                    geo_metadata, original_shape, spatial_resolution):
        """Save the case study dataset."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        if self.config['verbose']:
            print("\n" + "="*60)
            print("SAVING CASE STUDY DATASET")
            print("="*60)
        
        if self.config['save_binary_versions'] and len(features) > 0:
            features_bin, labels_bin = self.create_binary_versions(features, labels, threshold_mm)
        
        if self.config['save_npz'] and len(features) > 0:
            np.savez_compressed(out / 'features.npz', features=features)
            np.savez_compressed(out / 'labels.npz', labels=labels)
            if self.config['verbose']:
                print(f"Saved NPZ files to {out}/")
            if self.config['save_binary_versions']:
                np.savez_compressed(out / 'features_binary.npz', features=features_bin)
                np.savez_compressed(out / 'labels_binary.npz', labels=labels_bin)
        
        if self.config['save_pt'] and len(features) > 0:
            torch.save(torch.tensor(features), out / 'features.pt')
            torch.save(torch.tensor(labels), out / 'labels.pt')
            if self.config['verbose']:
                print(f"Saved PyTorch files to {out}/")
            if self.config['save_binary_versions']:
                torch.save(torch.tensor(features_bin), out / 'features_binary.pt')
                torch.save(torch.tensor(labels_bin), out / 'labels_binary.pt')
        
        if self.config['save_metadata']:
            metadata_dict = {
                'case_study_config': {
                    'start_datetime': self.config['start_datetime'],
                    'end_datetime': self.config['end_datetime'],
                    'temporal_resolution_minutes': int(self.config['temporal_resolution_minutes']),
                    'input_frames': int(self.input_frames),
                    'output_frames': int(self.output_frames),
                    'window_size': int(self.window_size)
                },
                'data_specifications': {
                    'total_windows': int(len(features)),
                    'feature_shape': [int(s) for s in features.shape] if len(features) > 0 else [],
                    'label_shape': [int(s) for s in labels.shape] if len(features) > 0 else [],
                    'spatial_resolution': {
                        'original_km_per_pixel': float(spatial_resolution['approx_km_per_pixel']),
                        'target_shape': [int(s) for s in self.config['target_size']],
                        'crs': str(spatial_resolution['crs'])
                    },
                    'value_range': {
                        'min_mm': float(features.min()) if len(features) > 0 else None,
                        'max_mm': float(features.max()) if len(features) > 0 else None,
                        'mean_mm': float(features.mean()) if len(features) > 0 else None
                    }
                },
                'binary_threshold': {
                    'method': f"{self.config['binary_threshold_percentile']}th_percentile",
                    'value_mm': float(threshold_mm)
                },
                'outlier_handling': {
                    'enabled': bool(self.config['fix_outliers']),
                    'method': 'adaptive_percentile' if self.config['adaptive_threshold'] else 'fixed_threshold',
                    'threshold': float(self.config['outlier_threshold']) if not self.config['adaptive_threshold'] 
                                else f"{self.config['outlier_percentile']}th_percentile",
                    'total_outliers_fixed': int(self.outlier_stats['total_fixed'])
                },
                'processing_stats': convert_to_json_serializable(self.processing_stats)
            }
            
            metadata_dict = convert_to_json_serializable(metadata_dict)
            
            with open(out / 'case_study_metadata.json', 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            serializable_metadata = convert_to_json_serializable(metadata)
            with open(out / 'window_metadata.json', 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            
            if self.config['verbose']:
                print(f"Saved metadata files to {out}/")
        
        if self.config['generate_report']:
            continuity_report = self.processing_stats.get('continuity_report', {})
            serializable_report = convert_to_json_serializable(continuity_report)
            with open(out / 'continuity_report.json', 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            serializable_outliers = convert_to_json_serializable(self.outlier_stats)
            with open(out / 'outlier_statistics.json', 'w') as f:
                json.dump(serializable_outliers, f, indent=2)
        
        if self.config['create_visualization'] and len(features) > 0:
            print("\nCreating visualizations...")
            self.visualize_sample_windows(
                features, labels, metadata, output_dir, 
                self.config['num_viz_samples']
            )
        
        if self.config['verbose']:
            print(f"\nAll data saved to {output_dir}/")
            print("Files created:")
            for f in out.glob('*'):
                print(f"  {f.name}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("CASE STUDY GENERATION SUMMARY")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Time period: {self.config['start_datetime']} to {self.config['end_datetime']}")
        print(f"  Input frames: {self.input_frames} ({(self.input_frames * self.config['temporal_resolution_minutes']) / 60:.1f} hours)")
        print(f"  Output frames: {self.output_frames} ({(self.output_frames * self.config['temporal_resolution_minutes']) / 60:.1f} hours)")
        print(f"  Spatial resolution: {self.config['target_size']} pixels")
        
        print(f"\nData Loading:")
        print(f"  Files found: {self.processing_stats.get('total_files_found', 0)}")
        print(f"  Files loaded: {self.processing_stats.get('total_files_loaded', 0)}")
        print(f"  Load errors: {len(self.processing_stats.get('load_errors', []))}")
        
        print(f"\nTemporal Continuity:")
        continuity = self.processing_stats.get('continuity_report', {})
        print(f"  Fully continuous: {continuity.get('is_fully_continuous', 'Unknown')}")
        print(f"  Gaps found: {continuity.get('gaps_found', 0)}")
        print(f"  Missing timesteps: {continuity.get('total_missing_timesteps', 0)}")
        
        print(f"\nWindows:")
        print(f"  Created: {self.processing_stats.get('total_windows_created', 0)}")
        print(f"  Skipped: {len(self.processing_stats.get('skipped_windows', []))}")
        
        print(f"\nOutliers:")
        print(f"  Total fixed: {self.outlier_stats['total_fixed']}")
        print(f"  Files affected: {self.outlier_stats['files_with_outliers']}")
        
        print(f"\nOutput Location:")
        print(f"  {self.config['output_dir']}")


# ============================================================
# 2. PYTORCH DATASET CLASS FOR CASE STUDY
# ============================================================
class CaseStudyDataset(Dataset):
    """PyTorch Dataset for case study precipitation windows (chronological order)."""
    
    def __init__(self, data_dir, binary=False):
        data_path = Path(data_dir)
        
        if binary:
            features_file = data_path / 'features_binary.npz'
            labels_file = data_path / 'labels_binary.npz'
        else:
            features_file = data_path / 'features.npz'
            labels_file = data_path / 'labels.npz'
        
        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"Data files not found in {data_dir}")
        
        self.features = np.load(features_file)['features'].astype(np.float32)
        self.labels = np.load(labels_file)['labels'].astype(np.float32)
        
        metadata_file = data_path / 'window_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.from_numpy(self.labels[idx])
        
        if self.metadata:
            meta = self.metadata[idx]
        else:
            meta = {'window_id': idx}
        
        return x, y, meta


# ============================================================
# 3. MAIN EXECUTION FUNCTION
# ============================================================
def generate_case_study_dataset(config=None):
    """Main function to generate a case study dataset."""
    if config is None:
        config = CASE_STUDY_CONFIG
    
    print("\n" + "="*60)
    print("CASE STUDY DATA GENERATION PIPELINE")
    print("="*60)
    print(f"Study period: {config['start_datetime']} to {config['end_datetime']}")
    print(f"Window: {config['input_frames']} input → {config['output_frames']} output frames")
    print(f"Temporal resolution: {config['temporal_resolution_minutes']} minutes")
    print(f"Target size: {config['target_size']}")
    
    generator = CaseStudyDataGenerator(config)
    
    file_list = generator.find_and_load_files()
    if len(file_list) == 0:
        raise ValueError("No files found in the specified time range")
    
    data, times, geo_metadata, target_size = generator.load_and_process_images(file_list)
    
    is_continuous, continuity_report = generator.check_continuity_and_report(times)
    generator.processing_stats['continuity_report'] = continuity_report
    
    config['target_size'] = target_size
    
    print("\n" + "="*60)
    print("COMPUTING DATASET STATISTICS")
    print("="*60)
    
    data_min = float(data.min())
    data_max = float(data.max())
    data_mean = float(data.mean())
    data_std = float(data.std())
    
    if len(data[data > 0]) > 0:
        binary_threshold = float(np.percentile(data[data > 0], config['binary_threshold_percentile']))
    else:
        binary_threshold = 0.0
    
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
    print(f"Mean: {data_mean:.4f}, Std: {data_std:.4f}")
    print(f"Binary threshold ({config['binary_threshold_percentile']}th percentile): {binary_threshold:.4f} mm")
    
    features, labels, window_metadata = generator.create_continuous_windows(
        data, times, continuity_report
    )
    
    if len(features) == 0:
        print("\n" + "!"*60)
        print("WARNING: No windows could be created!")
        print("!"*60)
        return generator, features, labels, window_metadata
    
    original_shape = geo_metadata['original_shape']
    spatial_resolution = {
        'pixel_width_degrees': float(geo_metadata['pixel_width_degrees']),
        'pixel_height_degrees': float(geo_metadata['pixel_height_degrees']),
        'approx_km_per_pixel': float(geo_metadata['spatial_resolution_approx_km']),
        'crs': str(geo_metadata['crs']),
        'bounds': geo_metadata['bounds']
    }
    
    generator.save_dataset(
        features, labels, window_metadata, 
        config['output_dir'], binary_threshold,
        geo_metadata, original_shape, spatial_resolution
    )
    
    generator.generate_summary_report()
    
    return generator, features, labels, window_metadata


# ============================================================
# 4. UTILITY FUNCTION TO QUICKLY LOAD A SAVED CASE STUDY
# ============================================================
def load_case_study(data_dir, binary=False, return_metadata=True):
    """Quick loader for a previously generated case study dataset."""
    dataset = CaseStudyDataset(data_dir, binary=binary)
    
    if return_metadata:
        metadata_path = Path(data_dir) / 'case_study_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        return dataset, metadata
    
    return dataset


# ============================================================
# 5. EXECUTION
# ============================================================
if __name__ == "__main__":
    generator, features, labels, metadata = generate_case_study_dataset()
    
    print("\n" + "="*60)
    print("TESTING CASE STUDY DATASET LOADER")
    print("="*60)
    
    try:
        case_dataset, case_metadata = load_case_study(CASE_STUDY_CONFIG['output_dir'])
        
        print(f"Dataset size: {len(case_dataset)} windows")
        
        if len(case_dataset) > 0:
            sample_x, sample_y, sample_meta = case_dataset[0]
            print(f"Sample input shape: {sample_x.shape}")
            print(f"Sample output shape: {sample_y.shape}")
            print(f"Sample input range: [{sample_x.min():.2f}, {sample_x.max():.2f}]")
            print(f"Sample output range: [{sample_y.min():.2f}, {sample_y.max():.2f}]")
            print(f"Sample window times: {sample_meta['input_times'][0]} → {sample_meta['output_times'][-1]}")
            
            dataloader = DataLoader(case_dataset, batch_size=4, shuffle=False)
            batch_x, batch_y, batch_meta = next(iter(dataloader))
            print(f"\nBatch input shape: {batch_x.shape}")
            print(f"Batch output shape: {batch_y.shape}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    print("\n" + "="*60)
    print("CASE STUDY GENERATION COMPLETE!")
    print("="*60)