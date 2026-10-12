# -*- coding: utf-8 -*-
"""
Refined Training Pipeline for U-Net Ablation Study Models
Supports dynamic input/output channel configuration
Supports all 5 models: UNet, DSC-UNet, SA-UNet, IR-UNet, SA-IRUNet

Improvements:
- Saves normalization parameters per model and in checkpoint files
- Detailed logging with min_delta_met flag
- Consistent summary JSON
- Correct mixed precision API (torch.cuda.amp)
- Proper time tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import gc
import os
import csv
import sys
import re
from datetime import datetime
from thop import profile

# Import all models from ablation_models.py
try:
    from ablation_models import (
        UNet, DSCUNet, SAUNet, IRUNet, SAIRUNet,
        count_parameters, print_model_metrics, save_model_metrics,
        calculate_model_metrics
    )
except ImportError as e:
    print(f"Error: Could not import from ablation_models.py: {e}")
    print("Please ensure ablation_models.py exists with all model definitions.")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # ========== MODEL SELECTION ==========
    'model_list': [ 'SA-IRUNet'],  #  model_list: ['UNet', 'DSC-UNet', 'SA-UNet', 'IR-UNet', 'SA-IRUNet']

    # ========== DATA PATHS ==========
    'data_dir': 'path/to/Data/128_128',
    'train_dir': 'path/to/ablation_study',

    # ========== TRAINING PARAMETERS ==========
    'batch_size': 4,
    'epochs': 200,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'loss_function': 'mse',  # 'mse', 'mae', or 'huber'
    'huber_delta': 1.0,

    # ========== MODEL ARCHITECTURE PARAMETERS ==========
    'in_channels': None,       # Auto-detect from data
    'out_channels': None,      # Auto-detect from data
    'n_filters': 16,
    'dropout': 0.05,
    'kernel_size': 3,
    'expand_ratio': 4,         # Only for IR-UNet and SA-IRUNet

    # ========== SCHEDULER AND EARLY STOPPING ==========
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_delta': 1e-5,
    'early_stop_patience': 15,
    'min_lr': 1e-7,

    # ========== REGULARIZATION ==========
    'depthwise_lambda': 1e-4,
    'use_unified_lambda': True,

    # ========== NORMALIZATION ==========
    'normalize_data': True,
    'normalization_range': (0, 1),
    'epsilon': 1e-8,

    # ========== MIXED PRECISION ==========
    'use_mixed_precision': True,

    # ========== DATALOADER SETTINGS ==========
    'num_workers': 0,
    'pin_memory': True,
    'prefetch_factor': 2,

    # ========== AUGMENTATION ==========
    'use_augmentation': False,
    'augmentation_factor': 4,
    'augmentation_ops': ['hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
    'augment_validation': False,

    # ========== OUTPUT CHANNELS ==========
    'max_lead_steps': 6,

    # ========== CHECKPOINT SETTINGS ==========
    'save_all_checkpoints': False,
}

# ============================================================
# AVAILABLE MODELS
# ============================================================
AVAILABLE_MODELS = {
    'UNet': UNet,
    'DSC-UNet': DSCUNet,
    'SA-UNet': SAUNet,
    'IR-UNet': IRUNet,
    'SA-IRUNet': SAIRUNet
}

def get_models_to_train(config):
    models = config.get('model_list', ['UNet'])
    invalid_models = [m for m in models if m not in AVAILABLE_MODELS]
    if invalid_models:
        print(f"Warning: Invalid models in model_list: {invalid_models}")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    valid_models = [m for m in models if m in AVAILABLE_MODELS]
    if not valid_models:
        print("No valid models to train. Defaulting to UNet.")
        valid_models = ['UNet']
    return valid_models

def get_model_display_name(model_type):
    display_names = {
        'UNet': 'Standard U-Net (Baseline)',
        'DSC-UNet': 'Depthwise Separable Convolution U-Net',
        'SA-UNet': 'Shuffle Attention U-Net',
        'IR-UNet': 'Inverted Residual U-Net',
        'SA-IRUNet': 'Shuffle Attention Inverted Residual U-Net'
    }
    return display_names.get(model_type, model_type)

def get_model_description(model_type):
    descriptions = {
        'UNet': 'Regular 2D convolutions, bilinear upsampling',
        'DSC-UNet': 'Depthwise separable convolutions (parameter efficient)',
        'SA-UNet': 'Depthwise separable convolutions + Shuffle Attention blocks',
        'IR-UNet': 'Inverted residual blocks, no attention',
        'SA-IRUNet': 'Inverted residual blocks with Shuffle Attention'
    }
    return descriptions.get(model_type, '')

def get_model_short_name(model_type):
    short_names = {
        'UNet': 'UNet',
        'DSC-UNet': 'DSC_UNet',
        'SA-UNet': 'SA_UNet',
        'IR-UNet': 'IR_UNet',
        'SA-IRUNet': 'SA_IRUNet'
    }
    return short_names.get(model_type, model_type)

# ============================================================
# MODEL FACTORY
# ============================================================
def create_model(model_type, config):
    model_params = {
        'in_channels': config['in_channels'],
        'out_channels': config['out_channels'],
        'n_filters': config['n_filters'],
        'dropout': config['dropout'],
        'kernel_size': config['kernel_size']
    }
    if model_type in ['IR-UNet', 'SA-IRUNet']:
        model_params['expand_ratio'] = config['expand_ratio']

    if model_type == 'UNet':
        return UNet(**model_params)
    elif model_type == 'DSC-UNet':
        return DSCUNet(**model_params)
    elif model_type == 'SA-UNet':
        return SAUNet(**model_params)
    elif model_type == 'IR-UNet':
        return IRUNet(**model_params)
    elif model_type == 'SA-IRUNet':
        return SAIRUNet(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================
class DataNormalizer:
    def __init__(self, norm_range=(0, 1), epsilon=1e-8):
        self.norm_range = norm_range
        self.epsilon = epsilon
        self.norm_params = None

    def fit(self, features, labels):
        self.norm_params = {
            'feat_min': float(features.min()),
            'feat_max': float(features.max()),
            'label_min': float(labels.min()),
            'label_max': float(labels.max()),
            'norm_min': self.norm_range[0],
            'norm_max': self.norm_range[1]
        }
        print("\n" + "="*60)
        print("NORMALIZATION PARAMETERS (Computed from Training Data)")
        print("="*60)
        print(f"  Features - Min: {self.norm_params['feat_min']:.6f}, Max: {self.norm_params['feat_max']:.6f}")
        print(f"  Labels   - Min: {self.norm_params['label_min']:.6f}, Max: {self.norm_params['label_max']:.6f}")
        print(f"  Normalization Range: [{self.norm_params['norm_min']}, {self.norm_params['norm_max']}]")
        print("="*60 + "\n")
        return self.norm_params

    def transform_features(self, features):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        feat_range = self.norm_params['feat_max'] - self.norm_params['feat_min']
        features_norm = (features - self.norm_params['feat_min']) / (feat_range + self.epsilon)
        features_norm = features_norm * (self.norm_params['norm_max'] - self.norm_params['norm_min']) + self.norm_params['norm_min']
        return features_norm

    def transform_labels(self, labels):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        label_range = self.norm_params['label_max'] - self.norm_params['label_min']
        labels_norm = (labels - self.norm_params['label_min']) / (label_range + self.epsilon)
        labels_norm = labels_norm * (self.norm_params['norm_max'] - self.norm_params['norm_min']) + self.norm_params['norm_min']
        return labels_norm

    def transform(self, features, labels=None):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        features_norm = self.transform_features(features)
        if labels is not None:
            labels_norm = self.transform_labels(labels)
            return features_norm, labels_norm
        return features_norm

    def inverse_transform_labels(self, labels_norm):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        labels_norm = (labels_norm - self.norm_params['norm_min']) / (self.norm_params['norm_max'] - self.norm_params['norm_min'] + self.epsilon)
        label_range = self.norm_params['label_max'] - self.norm_params['label_min']
        labels_orig = labels_norm * label_range + self.norm_params['label_min']
        return labels_orig

    def inverse_transform_features(self, features_norm):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        features_norm = (features_norm - self.norm_params['norm_min']) / (self.norm_params['norm_max'] - self.norm_params['norm_min'] + self.epsilon)
        feat_range = self.norm_params['feat_max'] - self.norm_params['feat_min']
        features_orig = features_norm * feat_range + self.norm_params['feat_min']
        return features_orig

    def save_params(self, save_path):
        if self.norm_params:
            with open(save_path, 'w') as f:
                json.dump(self.norm_params, f, indent=4)
            print(f"Normalization parameters saved to: {save_path}")

    def load_params(self, load_path):
        with open(load_path, 'r') as f:
            self.norm_params = json.load(f)
        print(f"Normalization parameters loaded from: {load_path}")

# ============================================================
# HUBER LOSS
# ============================================================
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        residual = pred - target
        abs_residual = torch.abs(residual)
        quadratic = torch.clamp(abs_residual, max=self.delta)
        linear = abs_residual - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================================
# AUGMENTATION
# ============================================================
def apply_augmentation(features, labels, aug_type):
    if aug_type == 'hflip':
        return np.flip(features, axis=-1).copy(), np.flip(labels, axis=-1).copy()
    elif aug_type == 'vflip':
        return np.flip(features, axis=-2).copy(), np.flip(labels, axis=-2).copy()
    elif aug_type == 'rot90':
        return np.rot90(features, k=1, axes=(-2, -1)).copy(), np.rot90(labels, k=1, axes=(-2, -1)).copy()
    elif aug_type == 'rot180':
        return np.rot90(features, k=2, axes=(-2, -1)).copy(), np.rot90(labels, k=2, axes=(-2, -1)).copy()
    elif aug_type == 'rot270':
        return np.rot90(features, k=3, axes=(-2, -1)).copy(), np.rot90(labels, k=3, axes=(-2, -1)).copy()
    else:
        return features, labels

def expand_dataset_with_augmentation(features, labels, augmentation_factor=4, augmentation_ops=None):
    if augmentation_ops is None:
        augmentation_ops = ['hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    n_samples = len(features)
    total_size = n_samples * (augmentation_factor + 1)
    original_shape = features.shape
    expanded_features = np.zeros((total_size, *original_shape[1:]), dtype=features.dtype)
    expanded_labels = np.zeros((total_size, *labels.shape[1:]), dtype=labels.dtype)
    expanded_features[:n_samples] = features
    expanded_labels[:n_samples] = labels
    idx = n_samples
    for i in range(n_samples):
        for aug_idx in range(augmentation_factor):
            aug_type = augmentation_ops[aug_idx % len(augmentation_ops)]
            aug_feat, aug_label = apply_augmentation(features[i], labels[i], aug_type)
            expanded_features[idx] = aug_feat
            expanded_labels[idx] = aug_label
            idx += 1
    return expanded_features, expanded_labels

# ============================================================
# DATASET
# ============================================================
class PrecipitationDataset(Dataset):
    def __init__(self, features, labels, normalizer=None, is_training=False,
                 normalize_features=True, normalize_labels=True, verbose=False):
        self.normalizer = normalizer
        self.is_training = is_training

        if normalizer is not None and (normalize_features or normalize_labels):
            if normalize_features and normalize_labels:
                features, labels = normalizer.transform(features, labels)
                if verbose:
                    print(f"  Features and labels normalized for {'training' if is_training else 'validation/test'} data")
            elif normalize_features:
                features = normalizer.transform_features(features)
                if verbose:
                    print(f"  Only features normalized for {'training' if is_training else 'validation/test'} data")
            elif normalize_labels:
                labels = normalizer.transform_labels(labels)
                if verbose:
                    print(f"  Only labels normalized for {'training' if is_training else 'validation/test'} data")

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

        if normalizer is not None and normalize_features:
            self.features = np.clip(self.features, 0, 1)
        if normalizer is not None and normalize_labels:
            self.labels = np.clip(self.labels, 0, 1)

        self.in_channels = self.features.shape[1]
        self.out_channels = self.labels.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])

# ============================================================
# REGULARIZATION
# ============================================================
def compute_regularization_loss(model, config):
    reg_loss = 0.0
    depthwise_lambda = config.get('depthwise_lambda', 1e-4)
    use_unified = config.get('use_unified_lambda', True)
    if use_unified:
        unified_lambda = depthwise_lambda

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'depthwise.weight' in name or ('shortcut.weight' in name and 'bn_shortcut' not in name):
            lambda_reg = unified_lambda if use_unified else depthwise_lambda
            reg_loss += lambda_reg * torch.norm(param, 2) ** 2
    return reg_loss

# ============================================================
# LOAD DATA
# ============================================================
def load_split_data(split_dir, load_binary=False):
    split_path = Path(split_dir)
    features = np.load(split_path / 'features.npz')['features']
    labels = np.load(split_path / 'labels.npz')['labels']
    print(f"  Loaded features shape: {features.shape}")
    print(f"  Loaded labels shape: {labels.shape}")

    if load_binary:
        features_binary = np.load(split_path / 'features_binary.npz')['features']
        labels_binary = np.load(split_path / 'labels_binary.npz')['labels']
    else:
        features_binary, labels_binary = None, None

    with open(split_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    return {
        'features': features,
        'labels': labels,
        'features_binary': features_binary,
        'labels_binary': labels_binary,
        'metadata': metadata
    }

def get_image_shape_from_data(data_dict):
    feature_shape = data_dict['features'].shape
    label_shape = data_dict['labels'].shape
    if len(feature_shape) == 4:
        n_samples, in_channels, height, width = feature_shape
        n_labels, out_channels, label_height, label_width = label_shape
        return {
            'height': height,
            'width': width,
            'input_channels': in_channels,
            'output_channels': out_channels,
            'input_shape': (in_channels, height, width),
            'output_shape': (out_channels, height, width),
            'resolution': f"{height}x{width}"
        }
    else:
        raise ValueError(f"Unexpected data shape: {feature_shape}. Expected (N, C, H, W)")

# ============================================================
# DATALOADER CREATION
# ============================================================
def create_dataloaders(train_data, val_data, test_data, config, normalizer=None,
                       normalize_features=None, normalize_labels=None, verbose=False):
    if normalizer is None:
        normalize_features = False if normalize_features is None else normalize_features
        normalize_labels = False if normalize_labels is None else normalize_labels
    else:
        normalize_features = True if normalize_features is None else normalize_features
        normalize_labels = True if normalize_labels is None else normalize_labels

    import sys
    if sys.platform == 'win32':
        num_workers = 0
    else:
        num_workers = config.get('num_workers', 0)

    loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': config.get('pin_memory', False) if torch.cuda.is_available() else False,
    }
    if num_workers > 0 and config.get('prefetch_factor'):
        loader_kwargs['prefetch_factor'] = config['prefetch_factor']

    train_loader = DataLoader(
        PrecipitationDataset(
            train_data['features'], train_data['labels'],
            normalizer=normalizer,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            is_training=True,
            verbose=verbose
        ),
        shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        PrecipitationDataset(
            val_data['features'], val_data['labels'],
            normalizer=normalizer,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            is_training=False,
            verbose=verbose
        ),
        shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        PrecipitationDataset(
            test_data['features'], test_data['labels'],
            normalizer=normalizer,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            is_training=False,
            verbose=verbose
        ),
        shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader

# ============================================================
# SAVE FUNCTIONS
# ============================================================
def save_config(config, save_path):
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

def get_base_train_dir(config):
    base_path = Path(config['train_dir'])
    if config.get('use_augmentation', False):
        aug_folder = 'augmented'
    else:
        aug_folder = 'non_augmented'
    resolution = config.get('resolution', 'unknown')
    return base_path / aug_folder / resolution

def get_model_unique_name(config, model_type):
    base_name = get_model_short_name(model_type)
    loss_func = config['loss_function'].upper()
    if loss_func == 'HUBER':
        loss_func = f"HUBER_delta{config.get('huber_delta', 1.0)}"
    else:
        loss_func = config['loss_function'].upper()

    if config.get('use_augmentation', False):
        aug_suffix = f"_aug{config['augmentation_factor']}x"
    else:
        aug_suffix = ""

    norm_suffix = "_norm" if config.get('normalize_data', False) else ""
    in_ch = config.get('in_channels', 'auto')
    out_ch = config.get('out_channels', 'auto')
    channel_info = f"_in{in_ch}_out{out_ch}"
    folder_name = f"{base_name}_{loss_func}{channel_info}{aug_suffix}{norm_suffix}"
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
    return folder_name

def save_training_log(log_dir, epoch, train_loss, val_loss, learning_rate, best_val_loss,
                      lr_reduction_count, no_improve_count, loss_improvement, min_delta_met):
    log_file = log_dir / 'training_log.csv'
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'best_val_loss',
                           'lr_reduction_count', 'no_improve_count', 'loss_improvement',
                           'min_delta_met', 'timestamp'])
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{learning_rate:.6e}", f"{best_val_loss:.6f}",
            lr_reduction_count, no_improve_count, f"{loss_improvement:.6e}",
            min_delta_met,
            datetime.now().isoformat()
        ])

def plot_training_curves(log_dir, train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("Warning: No loss data to plot")
        return

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    train_losses = np.nan_to_num(train_losses, nan=1e6, posinf=1e6, neginf=1e6)
    val_losses = np.nan_to_num(val_losses, nan=1e6, posinf=1e6, neginf=1e6)

    if np.max(train_losses) / (np.min(train_losses) + 1e-8) > 1000:
        plt.yscale('log')
        ylabel = 'Loss (log scale)'
    else:
        ylabel = 'Loss'

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    plt.plot(best_val_idx + 1, best_val_loss, 'ro', markersize=8,
             label=f'Best Val Loss: {best_val_loss:.6f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Training vs Validation Loss Curves - {model_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = log_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  GPU: Not available (using CPU)")

# ============================================================
# FIND EXISTING MODEL
# ============================================================
def find_existing_model_with_same_config(base_train_dir, current_config, model_type):
    base_path = Path(base_train_dir)
    if not base_path.exists():
        return None, None

    compare_keys = [
        'data_dir', 'batch_size', 'epochs', 'learning_rate', 'weight_decay',
        'loss_function', 'huber_delta', 'num_workers', 'pin_memory', 'prefetch_factor',
        'scheduler_patience', 'scheduler_factor', 'scheduler_min_delta', 'early_stop_patience', 'min_lr',
        'in_channels', 'out_channels', 'n_filters', 'dropout', 'kernel_size', 'expand_ratio',
        'use_augmentation', 'augmentation_factor', 'augment_validation',
        'depthwise_lambda', 'use_unified_lambda',
        'normalize_data', 'normalization_range'
    ]

    current_comparable = {k: current_config[k] for k in compare_keys if k in current_config}
    current_comparable['model_type'] = model_type

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        config_file = model_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            existing_comparable = {k: existing_config[k] for k in compare_keys if k in existing_config}
            existing_comparable['model_type'] = existing_config.get('model_type', '')
            if existing_comparable == current_comparable:
                return model_dir, existing_config
    return None, None

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, config, scaler=None, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]",
                leave=True, position=0, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                task_loss = criterion(out, y)
                reg_loss = compute_regularization_loss(model, config)
                loss = task_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            task_loss = criterion(out, y)
            reg_loss = compute_regularization_loss(model, config)
            loss = task_loss + reg_loss
            loss.backward()
            optimizer.step()

        running_loss += task_loss.item()
        pbar.set_postfix({
            'loss': f'{task_loss.item():.6f}',
            'avg': f'{running_loss/(pbar.n+1):.6f}'
        })

    return running_loss / len(loader)

def validate_epoch(model, loader, criterion, device, epoch=None, total_epochs=None):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]",
                leave=True, position=1, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg': f'{running_loss/(pbar.n+1):.6f}'
            })

    return running_loss / len(loader)

# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
def train_single_model(model_type, config, train_data_raw, val_data_raw, test_data_raw,
                       normalizer, image_info, base_train_dir):
    """Train a single model with the given configuration"""

    print("\n" + "="*80)
    print(f"TRAINING MODEL: {model_type}")
    print("="*80)
    print(f"  Description: {get_model_description(model_type)}")
    print("="*80)

    # Get model directory
    unique_model_name = get_model_unique_name(config, model_type)
    model_train_dir = base_train_dir / unique_model_name
    checkpoints_dir = model_train_dir / 'checkpoints'
    logs_dir = model_train_dir / 'logs'

    # Check if model already exists
    existing_model_dir, existing_config = find_existing_model_with_same_config(
        base_train_dir, config, model_type
    )

    if existing_model_dir is not None:
        print("\n" + "="*60)
        print("WARNING: Trained model with same parameters already exists!")
        print(f"  Existing model: {existing_model_dir.name}")
        print(f"  Location: {existing_model_dir}")
        print("="*60)
        response = input(f"Continue training {model_type} anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print(f"Skipping {model_type} training.")
            return None

    # Create directories
    model_train_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save config with model type
    config_copy = config.copy()
    config_copy['model_type'] = model_type
    save_config(config_copy, model_train_dir / 'config.json')

    # Save normalization parameters to model directory (if available)
    if normalizer and normalizer.norm_params:
        model_norm_path = model_train_dir / 'norm_params.json'
        normalizer.save_params(model_norm_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel: {unique_model_name}")
    print(f"Device: {device}")
    print(f"Directory: {model_train_dir}")
    print_gpu_info()

    if normalizer:
        print(f"\nNormalization Settings:")
        print(f"  Type: Min-Max Normalization")
        print(f"  Range: [{config['normalization_range'][0]}, {config['normalization_range'][1]}]")

    print(f"\nModel Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Description: {get_model_description(model_type)}")
    print(f"  Input Channels: {config['in_channels']}")
    print(f"  Output Channels: {config['out_channels']}")
    print(f"  Base Filters: {config['n_filters']}")
    print(f"  Kernel Size: {config['kernel_size']}")
    if model_type in ['IR-UNet', 'SA-IRUNet']:
        print(f"  Expand Ratio: {config['expand_ratio']}")
    print(f"  Dropout: {config['dropout']}")

    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, config).to(device)

    # Calculate model metrics
    try:
        print("\nCalculating model metrics...")
        input_shape = (config['in_channels'], config['image_height'], config['image_width'])
        model_metrics = calculate_model_metrics(model, input_shape, device)
        print_model_metrics(model_metrics, get_model_display_name(model_type))
        save_model_metrics(model_metrics, model_train_dir / 'model_metrics.txt', get_model_display_name(model_type))

        with open(model_train_dir / 'model_metrics.json', 'w') as f:
            metrics_json = {k: v for k, v in model_metrics.items() if k not in ['layer_breakdown', 'block_summary']}
            metrics_json['layer_breakdown'] = [
                {k: v for k, v in layer.items()} for layer in model_metrics['layer_breakdown']
            ]
            json.dump(metrics_json, f, indent=4)
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

    # Prepare data
    if normalizer:
        train_features, train_labels = normalizer.transform(train_data_raw['features'], train_data_raw['labels'])
        val_features, val_labels = normalizer.transform(val_data_raw['features'], val_data_raw['labels'])
        test_features, test_labels = normalizer.transform(test_data_raw['features'], test_data_raw['labels'])
        train_data = {'features': train_features, 'labels': train_labels}
        val_data = {'features': val_features, 'labels': val_labels}
        test_data = {'features': test_features, 'labels': test_labels}
    else:
        train_data = train_data_raw
        val_data = val_data_raw
        test_data = test_data_raw

    original_train_size = len(train_data['features'])
    original_val_size = len(val_data['features'])
    test_size = len(test_data['features'])

    # Apply augmentation
    if config['use_augmentation']:
        print("\nApplying augmentation to training dataset...")
        expanded_features, expanded_labels = expand_dataset_with_augmentation(
            train_data['features'], train_data['labels'],
            augmentation_factor=config['augmentation_factor'],
            augmentation_ops=config['augmentation_ops']
        )
        train_data['features'] = expanded_features
        train_data['labels'] = expanded_labels
        print(f"  Training data augmented: {original_train_size} -> {len(train_data['features'])} samples")

    if config.get('augment_validation', False) and config['use_augmentation']:
        print("\nApplying augmentation to validation dataset...")
        val_expanded_features, val_expanded_labels = expand_dataset_with_augmentation(
            val_data['features'], val_data['labels'],
            augmentation_factor=config['augmentation_factor'],
            augmentation_ops=config['augmentation_ops']
        )
        val_data['features'] = val_expanded_features
        val_data['labels'] = val_expanded_labels

    final_train_size = len(train_data['features'])
    final_val_size = len(val_data['features'])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, config,
        normalizer=None,
        normalize_features=False,
        normalize_labels=False,
        verbose=False
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss function
    if config['loss_function'].lower() == 'mse':
        criterion = nn.MSELoss()
        print(f"\nUsing MSE Loss")
    elif config['loss_function'].lower() == 'mae':
        criterion = nn.L1Loss()
        print(f"\nUsing MAE Loss")
    elif config['loss_function'].lower() == 'huber':
        delta = config.get('huber_delta', 1.0)
        criterion = HuberLoss(delta=delta, reduction='mean')
        print(f"\nUsing Huber Loss with delta={delta}")
    else:
        raise ValueError("Loss function must be 'mse', 'mae', or 'huber'")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and
                                             config.get('use_mixed_precision', True)) else None

    max_lr_reductions = config['early_stop_patience'] // config['scheduler_patience']
    if max_lr_reductions < 1:
        max_lr_reductions = 1

    lr_reduction_count = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    no_improve_count = 0

    min_delta = config.get('scheduler_min_delta', 1e-6)
    scheduler_patience = config['scheduler_patience']
    scheduler_factor = config['scheduler_factor']
    min_lr = config['min_lr']

    print(f"\nStarting training:")
    print(f"  Model: {get_model_display_name(model_type)}")
    print(f"  Input channels: {config['in_channels']}")
    print(f"  Output channels: {config['out_channels']}")
    print(f"  Normalization: {'Enabled (Min-Max to [0,1])' if normalizer else 'Disabled'}")
    print(f"  Training samples: {final_train_size} (original: {original_train_size})")
    print(f"  Validation samples: {final_val_size} (original: {original_val_size})")
    print(f"  Test samples: {test_size}")
    print(f"  Resolution: {config['resolution']}")
    print(f"  Augmentation: {'Enabled' if config['use_augmentation'] else 'Disabled'}")
    print(f"  Scheduler patience: {scheduler_patience} epochs")
    print(f"  Scheduler min delta: {min_delta:.2e}")
    print(f"  Scheduler factor: {scheduler_factor}")
    print(f"  Min LR: {min_lr:.2e}\n")

    # Start timer for training
    start_time = time.time()

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        clear_gpu_memory()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 config, scaler, epoch, config['epochs'])
        val_loss = validate_epoch(model, val_loader, criterion, device,
                                  epoch, config['epochs'])

        epoch_time = time.time() - epoch_start
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        loss_gap = train_loss - val_loss

        # Calculate loss improvement with minimum delta check
        if len(val_losses) >= 2:
            loss_improvement = val_losses[-2] - val_loss
            min_delta_met = loss_improvement > min_delta
        else:
            loss_improvement = 0.0
            min_delta_met = True

        # Check if this is the best model
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            if loss_improvement > min_delta:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config,
                'image_dimensions': {'height': config['image_height'], 'width': config['image_width']},
                'normalization_params': normalizer.norm_params if normalizer else None,
                'timestamp': datetime.now().isoformat()
            }, checkpoints_dir / 'best_model.pth')
            checkpoint_message = f"Best model saved (val loss: {val_loss:.6f})"
        else:
            if loss_improvement <= min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0
            checkpoint_message = ""

        old_lr = current_lr

        # LR scheduler logic
        if no_improve_count >= scheduler_patience:
            new_lr = max(old_lr * scheduler_factor, min_lr)
            if new_lr < old_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                lr_reduction_count += 1
                no_improve_count = 0
                if checkpoint_message:
                    checkpoint_message += f" | LR reduced from {old_lr:.2e} to {new_lr:.2e}"
                else:
                    checkpoint_message = f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}"

        current_lr = optimizer.param_groups[0]['lr']

        # Log to CSV
        save_training_log(logs_dir, epoch, train_loss, val_loss, current_lr, best_val_loss,
                         lr_reduction_count, no_improve_count, loss_improvement, min_delta_met)

        # Verbose printing
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / 1e9
            mem_res = torch.cuda.memory_reserved(device) / 1e9
            print(f"\nEpoch {epoch}/{config['epochs']} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Gap: {loss_gap:.6f}")
            print(f"  Improvement: {loss_improvement:+.6e} (delta: {min_delta:.2e}) | Best: {best_val_loss:.6f}")
            print(f"  No improvement: {no_improve_count}/{scheduler_patience} | LR reductions: {lr_reduction_count}/{max_lr_reductions}")
            print(f"  LR: {current_lr:.2e} | GPU Mem: {mem_alloc:.2f}/{mem_res:.2f} GB")
            if checkpoint_message:
                print(f"  {checkpoint_message}")
        else:
            print(f"\nEpoch {epoch}/{config['epochs']} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Gap: {loss_gap:.6f}")
            print(f"  Improvement: {loss_improvement:+.6e} (delta: {min_delta:.2e}) | Best: {best_val_loss:.6f}")
            print(f"  No improvement: {no_improve_count}/{scheduler_patience} | LR reductions: {lr_reduction_count}/{max_lr_reductions}")
            print(f"  LR: {current_lr:.2e}")
            if checkpoint_message:
                print(f"  {checkpoint_message}")

        # Early stopping
        if lr_reduction_count >= max_lr_reductions:
            print(f"\nEarly stopping after {epoch} epochs (max LR reductions reached)")
            print(f"Best validation loss: {best_val_loss:.6f}\n")
            break

    total_time = time.time() - start_time
    final_epoch = len(train_losses)

    # Save final model
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1] if val_losses else float('inf'),
        'train_loss': train_losses[-1] if train_losses else float('inf'),
        'best_val_loss': best_val_loss,
        'config': config,
        'image_dimensions': {'height': config['image_height'], 'width': config['image_width']},
        'normalization_params': normalizer.norm_params if normalizer else None,
        'timestamp': datetime.now().isoformat()
    }, checkpoints_dir / 'final_model.pth')

    print(f"\nFinal model saved to: {checkpoints_dir / 'final_model.pth'}")

    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(logs_dir, train_losses, val_losses, model_type)

    # Test evaluation - load best model with error handling
    checkpoint_path = checkpoints_dir / 'best_model.pth'
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                best_state = checkpoint['model_state_dict']
                model.load_state_dict(best_state)
                print(f"  Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                print(f"Warning: Checkpoint missing 'model_state_dict' key")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model state for testing")
    else:
        print(f"Warning: Best model not found at {checkpoint_path}, using current model")

    model.eval()
    test_loss = 0.0
    print("\nTesting best model...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pbar.set_postfix({'loss': f'{test_loss/(pbar.n+1):.6f}'})
    test_loss /= len(test_loader) if len(test_loader) > 0 else 1

    print(f"\nTest Loss ({config['loss_function'].upper()}): {test_loss:.6f}")
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {model_train_dir}\n")

    # Save detailed summary
    summary = {
        'model_name': model_type,
        'model_display_name': get_model_display_name(model_type),
        'model_description': get_model_description(model_type),
        'in_channels': config['in_channels'],
        'out_channels': config['out_channels'],
        'unique_folder_name': unique_model_name,
        'directory_structure': str(base_train_dir),
        'total_training_time_seconds': total_time,
        'total_epochs_completed': final_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'normalization_enabled': config.get('normalize_data', False),
        'normalization_type': 'minmax' if config.get('normalize_data', False) else 'none',
        'normalization_range': config['normalization_range'] if config.get('normalize_data', False) else None,
        'normalization_params_file': str(model_train_dir / 'norm_params.json') if normalizer else None,
        'scheduler_settings': {
            'patience': scheduler_patience,
            'factor': scheduler_factor,
            'min_delta': min_delta,
            'min_lr': min_lr,
            'total_reductions': lr_reduction_count
        },
        'checkpoints': {
            'best_model': str(checkpoints_dir / 'best_model.pth'),
            'final_model': str(checkpoints_dir / 'final_model.pth')
        },
        'config': config
    }
    with open(model_train_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    return model_train_dir

# ============================================================
# MAIN
# ============================================================
def main():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("\n" + "="*70)
    print("U-NET ABLATION STUDY TRAINING PIPELINE (REFINED)")
    print("="*70)

    models_to_train = get_models_to_train(CONFIG)
    print(f"\nModels to train: {len(models_to_train)}")
    for i, model in enumerate(models_to_train, 1):
        print(f"  {i}. {model} - {get_model_description(model)}")
    print("="*70)

    # Load data to determine dimensions first
    print("\nLoading data...")
    train_data_raw = load_split_data(Path(CONFIG['data_dir']) / 'train', load_binary=False)
    image_info = get_image_shape_from_data(train_data_raw)

    # Auto-detect or use configured channels
    if CONFIG.get('in_channels') is None:
        CONFIG['in_channels'] = image_info['input_channels']
        print(f"Auto-detected input channels: {CONFIG['in_channels']}")
    if CONFIG.get('out_channels') is None:
        CONFIG['out_channels'] = image_info['output_channels']
        print(f"Auto-detected output channels: {CONFIG['out_channels']}")

    CONFIG['image_height'] = image_info['height']
    CONFIG['image_width'] = image_info['width']
    CONFIG['resolution'] = image_info['resolution']
    CONFIG['max_lead_steps'] = CONFIG['out_channels']

    print(f"\nData Configuration:")
    print(f"  Resolution: {CONFIG['image_height']}x{CONFIG['image_width']}")
    print(f"  Input Channels: {CONFIG['in_channels']} (previous timesteps)")
    print(f"  Output Channels: {CONFIG['out_channels']} (future timesteps)")
    print("="*70)

    # Initialize normalizer if enabled
    normalizer = None
    norm_params_path = None

    if CONFIG.get('normalize_data', False):
        normalizer = DataNormalizer(
            norm_range=CONFIG['normalization_range'],
            epsilon=CONFIG['epsilon']
        )
        print("\n" + "="*60)
        print("MIN-MAX NORMALIZATION ENABLED")
        print("="*60)
        print("Fitting normalizer on training data...")
        normalizer.fit(train_data_raw['features'], train_data_raw['labels'])
        norm_params_path = Path(CONFIG['data_dir']) / 'norm_params.json'
        normalizer.save_params(norm_params_path)
    else:
        print("\n" + "="*60)
        print("NORMALIZATION DISABLED - Using raw data")
        print("="*60)

    # Create base training directory
    base_train_dir = get_base_train_dir(CONFIG)

    # Train each model
    results = {}
    pipeline_start_time = time.time()

    for i, model_type in enumerate(models_to_train, 1):
        print(f"\n\n{'#'*80}")
        print(f"TRAINING MODEL {i}/{len(models_to_train)}: {model_type}")
        print(f"{'#'*80}\n")

        # Load validation and test data (only once, reused for all models)
        val_data_raw = load_split_data(Path(CONFIG['data_dir']) / 'val', load_binary=False)
        test_data_raw = load_split_data(Path(CONFIG['data_dir']) / 'test', load_binary=False)

        # Train the model
        result_dir = train_single_model(
            model_type, CONFIG, train_data_raw, val_data_raw, test_data_raw,
            normalizer, image_info, base_train_dir
        )

        if result_dir is not None:
            results[model_type] = str(result_dir)

        # Clear memory between models
        clear_gpu_memory()

    total_pipeline_time = time.time() - pipeline_start_time

    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"Total time: {total_pipeline_time/60:.2f} minutes")
    print(f"Models trained: {len(results)}/{len(models_to_train)}")
    print("\nResults:")
    for model_type, dir_path in results.items():
        print(f"  - {model_type}: {dir_path}")

    if len(results) < len(models_to_train):
        skipped = set(models_to_train) - set(results.keys())
        print(f"\nSkipped models: {skipped}")

    print("="*70)

if __name__ == "__main__":
    main()
