# -*- coding: utf-8 -*-
"""
Training Pipeline for SA-IRUNet
================================
Dynamic in/out channels, min-max normalisation, best+final checkpoints only.
Losses are provided by losses.py (selected via CONFIG['loss_function']).
"""

import gc
import csv
import json
import re
import sys
import time
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thop import profile

# ------------------------------------------------------------------ imports
from IR_UNET import SAIRUNet
from losses import build_loss


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'data_dir':  'D:/PhD Work III/INSAT 3DR HEM/IR_UNet/Data/128_128',
    'train_dir': 'D:/PhD Work III/INSAT 3DR HEM/IR_UNet/ablation_study',

    'batch_size':  8,
    'epochs':      200,
    'learning_rate': 1e-4,
    'weight_decay':  1e-5,

    'loss_function': 'rain_mse',          # 'mse' | 'mae' | 'huber' | 'rain_mse'
    'huber_delta':   1.0,
    'rain_threshold': 0.05,
    'rain_weight':    5.0,

    'in_channels':  None,            # auto-detect
    'out_channels': None,            # auto-detect

    'n_filters':    64,
    'dropout':      0.05,
    'kernel_size':  3,
    'expand_ratio': 4,

    'scheduler_patience':  5,
    'scheduler_factor':    0.5,
    'scheduler_min_delta': 1e-5,
    'early_stop_patience': 15,
    'min_lr':              1e-7,

    'depthwise_lambda':   1e-4,
    'use_unified_lambda': True,

    'normalize_data':      True,
    'normalization_range': (0, 1),
    'epsilon':             1e-8,

    'use_mixed_precision': True,
    'num_workers':         0,
    'pin_memory':          True,
    'prefetch_factor':     2,

    'use_augmentation':    False,
    'augmentation_factor': 4,
    'augmentation_ops':    ['hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
    'augment_validation':  False,

    'max_lead_steps':       6,
    'save_all_checkpoints': False,

    # verbosity
    'verbose_epoch_line':  True,
    'verbose_every_n':     1,
    'verbose_on_events':   True,
}


# ============================================================
# LOGGING
# ============================================================
class Log:
    """Small aligned logger. No decorative banners."""

    WIDTH = 76

    @staticmethod
    def header(title):
        print(title)
        print("-" * Log.WIDTH)

    @staticmethod
    def section(title):
        print(f"\n{title}")
        print("-" * Log.WIDTH)

    @staticmethod
    def kv(key, value, indent=2, key_width=32):
        print(f"{' ' * indent}{key:<{key_width}} {value}")

    @staticmethod
    def line(text, indent=2):
        print(f"{' ' * indent}{text}")


# ============================================================
# NORMALISATION
# ============================================================
class DataNormalizer:
    """Min-max normaliser for precipitation data."""

    def __init__(self, norm_range=(0, 1), epsilon=1e-8):
        self.norm_range = norm_range
        self.epsilon = epsilon
        self.norm_params = None

    # ---- fit / transform ----
    def fit(self, features, labels):
        self.norm_params = {
            'feat_min':  float(features.min()),
            'feat_max':  float(features.max()),
            'label_min': float(labels.min()),
            'label_max': float(labels.max()),
            'norm_min':  self.norm_range[0],
            'norm_max':  self.norm_range[1],
        }
        return self.norm_params

    def transform_features(self, features):
        p = self._require()
        rng = p['feat_max'] - p['feat_min']
        return (features - p['feat_min']) / (rng + self.epsilon) \
               * (p['norm_max'] - p['norm_min']) + p['norm_min']

    def transform_labels(self, labels):
        p = self._require()
        rng = p['label_max'] - p['label_min']
        return (labels - p['label_min']) / (rng + self.epsilon) \
               * (p['norm_max'] - p['norm_min']) + p['norm_min']

    def transform(self, features, labels=None):
        f = self.transform_features(features)
        if labels is None:
            return f
        return f, self.transform_labels(labels)

    def inverse_transform_labels(self, labels_norm):
        p = self._require()
        labels_norm = (labels_norm - p['norm_min']) / \
                      (p['norm_max'] - p['norm_min'] + self.epsilon)
        return labels_norm * (p['label_max'] - p['label_min']) + p['label_min']

    # ---- persistence ----
    def save_params(self, save_path):
        if self.norm_params:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.norm_params, f, indent=4)

    def load_params(self, load_path):
        with open(load_path, 'r') as f:
            self.norm_params = json.load(f)

    # ---- helpers ----
    def _require(self):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.norm_params


# ============================================================
# AUGMENTATION
# ============================================================
def apply_augmentation(features, labels, aug_type):
    if aug_type == 'hflip':
        return np.flip(features, -1).copy(), np.flip(labels, -1).copy()
    if aug_type == 'vflip':
        return np.flip(features, -2).copy(), np.flip(labels, -2).copy()
    if aug_type == 'rot90':
        return (np.rot90(features, 1, (-2, -1)).copy(),
                np.rot90(labels,   1, (-2, -1)).copy())
    if aug_type == 'rot180':
        return (np.rot90(features, 2, (-2, -1)).copy(),
                np.rot90(labels,   2, (-2, -1)).copy())
    if aug_type == 'rot270':
        return (np.rot90(features, 3, (-2, -1)).copy(),
                np.rot90(labels,   3, (-2, -1)).copy())
    return features, labels


def expand_dataset_with_augmentation(features, labels,
                                     augmentation_factor=4, augmentation_ops=None):
    if augmentation_ops is None:
        augmentation_ops = ['hflip', 'vflip', 'rot90', 'rot180', 'rot270']

    n = len(features)
    total = n * (augmentation_factor + 1)
    exp_f = np.zeros((total, *features.shape[1:]), dtype=features.dtype)
    exp_l = np.zeros((total, *labels.shape[1:]),   dtype=labels.dtype)

    exp_f[:n] = features
    exp_l[:n] = labels

    idx = n
    for i in range(n):
        for a in range(augmentation_factor):
            op = augmentation_ops[a % len(augmentation_ops)]
            af, al = apply_augmentation(features[i], labels[i], op)
            exp_f[idx] = af
            exp_l[idx] = al
            idx += 1
    return exp_f, exp_l


# ============================================================
# DATASET
# ============================================================
class PrecipitationDataset(Dataset):
    def __init__(self, features, labels, normalizer=None,
                 normalize_features=True, normalize_labels=True):
        if normalizer is not None and (normalize_features or normalize_labels):
            if normalize_features and normalize_labels:
                features, labels = normalizer.transform(features, labels)
            elif normalize_features:
                features = normalizer.transform_features(features)
            elif normalize_labels:
                labels = normalizer.transform_labels(labels)

        self.features = features.astype(np.float32)
        self.labels   = labels.astype(np.float32)

        if normalizer is not None and normalize_features:
            self.features = np.clip(self.features, 0, 1)
        if normalizer is not None and normalize_labels:
            self.labels = np.clip(self.labels, 0, 1)

        self.in_channels  = self.features.shape[1]
        self.out_channels = self.labels.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.from_numpy(self.labels[idx]))


# ============================================================
# DATA IO
# ============================================================
def load_split_data(split_dir):
    split_path = Path(split_dir)
    features = np.load(split_path / 'features.npz')['features']
    labels   = np.load(split_path / 'labels.npz')['labels']

    metadata = {}
    mpath = split_path / 'metadata.json'
    if mpath.exists():
        with open(mpath, 'r') as f:
            metadata = json.load(f)

    return {'features': features, 'labels': labels, 'metadata': metadata}


def get_image_shape_from_data(data_dict):
    f = data_dict['features'].shape
    l = data_dict['labels'].shape
    if len(f) != 4 or len(l) != 4:
        raise ValueError(f"Expected (N, C, H, W); got {f} and {l}")

    n, in_ch, h, w = f
    _, out_ch, h2, w2 = l
    if (h, w) != (h2, w2):
        raise ValueError(f"Spatial mismatch: features {h}x{w}, labels {h2}x{w2}")

    return {
        'height': h, 'width': w,
        'input_channels': in_ch, 'output_channels': out_ch,
        'resolution': f"{h}x{w}",
    }


def create_dataloaders(train_data, val_data, test_data, config):
    num_workers = 0 if sys.platform == 'win32' else config.get('num_workers', 0)

    kwargs = {
        'batch_size':  config['batch_size'],
        'num_workers': num_workers,
        'pin_memory':  config.get('pin_memory', False) if torch.cuda.is_available() else False,
    }
    if num_workers > 0 and config.get('prefetch_factor'):
        kwargs['prefetch_factor'] = config['prefetch_factor']

    def make(data, shuffle):
        return DataLoader(
            PrecipitationDataset(data['features'], data['labels'],
                                 normalizer=None,
                                 normalize_features=False,
                                 normalize_labels=False),
            shuffle=shuffle, **kwargs)

    return make(train_data, True), make(val_data, False), make(test_data, False)


# ============================================================
# REGULARISATION
# ============================================================
def compute_regularization_loss(model):
    lam = CONFIG.get('depthwise_lambda', 1e-4)
    reg = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'depthwise.weight' in name or \
           ('shortcut.weight' in name and 'bn_shortcut' not in name):
            reg = reg + lam * torch.norm(param, 2) ** 2
    return reg


# ============================================================
# MODEL METRICS
# ============================================================
def get_block_wise_summary(model, dummy_input, device='cpu'):
    block_info = []
    blocks = ['conv_in', 'down1', 'down2', 'down3', 'down4',
              'up1', 'up2', 'up3', 'up4', 'conv_out']

    def hook_fn(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                block_info.append({
                    'block': name,
                    'type': module.__class__.__name__,
                    'input_shape': list(inp[0].shape) if inp else [],
                    'output_shape': list(out.shape),
                })
        return hook

    hooks = []
    for name in blocks:
        if hasattr(model, name):
            hooks.append(getattr(model, name).register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()
    return block_info


def calculate_model_metrics(model, input_shape, device='cpu'):
    model.eval().to(device)
    dummy = torch.randn(1, *input_shape, device=device)

    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        macs = flops / 2
    except Exception:
        flops, macs = 0, 0

    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    act_bytes = 0
    def act_hook(m, inp, out):
        nonlocal act_bytes
        if isinstance(out, torch.Tensor):
            act_bytes += out.nelement() * out.element_size()

    hooks = [m.register_forward_hook(act_hook) for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                               nn.BatchNorm2d, nn.ReLU, nn.Sequential))]
    with torch.no_grad():
        _ = model(dummy)
    for h in hooks:
        h.remove()

    layer_stats = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                               nn.BatchNorm2d, nn.MaxPool2d, nn.Dropout2d)):
            params_layer = sum(p.numel() for p in module.parameters())
            if params_layer > 0:
                layer_stats.append({
                    'name': name or 'root',
                    'type': module.__class__.__name__,
                    'parameters': params_layer,
                })

    return {
        'total_parameters':         sum(p.numel() for p in model.parameters()),
        'trainable_parameters':     sum(p.numel() for p in model.parameters() if p.requires_grad),
        'non_trainable_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'flops': int(flops), 'flops_giga': flops / 1e9,
        'macs':  int(macs),  'macs_giga':  macs / 1e9,
        'model_size_mb':        (param_size + buffer_size) / (1024 ** 2),
        'activation_memory_mb': act_bytes / (1024 ** 2),
        'layer_breakdown':      layer_stats,
        'block_summary':        get_block_wise_summary(model, dummy, device),
    }


def write_model_metrics(metrics, stream, model_type="SA-IRUNet"):
    """Write model metrics to any text stream (stdout or file)."""
    w = stream.write

    w(f"MODEL METRICS - {model_type}\n")
    w("=" * 76 + "\n")

    w("\nBlock-wise architecture summary\n")
    w("-" * 76 + "\n")
    w(f"{'Block':<12} {'Type':<22} {'Input':<22} {'Output':<22}\n")
    w("-" * 76 + "\n")
    for b in metrics['block_summary']:
        inp = ' x '.join(str(x) for x in b['input_shape'][1:]) if b['input_shape'] else '-'
        out = ' x '.join(str(x) for x in b['output_shape'][1:]) if b['output_shape'] else '-'
        w(f"{b['block']:<12} {b['type'][:22]:<22} {inp:<22} {out:<22}\n")

    w("\nParameters\n")
    w("-" * 76 + "\n")
    w(f"  Total:        {metrics['total_parameters']:,}\n")
    w(f"  Trainable:    {metrics['trainable_parameters']:,}\n")
    w(f"  Non-trainable:{metrics['non_trainable_parameters']:,}\n")

    w("\nCompute\n")
    w("-" * 76 + "\n")
    w(f"  FLOPs: {metrics['flops_giga']:.2f} G\n")
    w(f"  MACs:  {metrics['macs_giga']:.2f} G\n")

    w("\nMemory (batch = 1)\n")
    w("-" * 76 + "\n")
    w(f"  Model size: {metrics['model_size_mb']:.2f} MB\n")
    w(f"  Activations: {metrics['activation_memory_mb']:.2f} MB\n")

    w("\nParameter distribution by layer type\n")
    w("-" * 76 + "\n")
    type_stats = {}
    for layer in metrics['layer_breakdown']:
        s = type_stats.setdefault(layer['type'], {'params': 0, 'count': 0})
        s['params'] += layer['parameters']
        s['count']  += 1

    total = metrics['total_parameters']
    w(f"  {'Type':<30} {'Count':>8} {'Params':>16} {'%':>8}\n")
    for t, s in sorted(type_stats.items(), key=lambda kv: kv[1]['params'], reverse=True):
        pct = 100.0 * s['params'] / total if total else 0.0
        w(f"  {t:<30} {s['count']:>8} {s['params']:>16,} {pct:>7.2f}%\n")
    w("=" * 76 + "\n")


# ============================================================
# TRAINING / VALIDATION
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="  train", leave=False, ncols=100,
                bar_format='{l_bar}{bar:22}{r_bar}')

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                out = model(x)
                task = criterion(out, y)
                loss = task + compute_regularization_loss(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            task = criterion(out, y)
            loss = task + compute_regularization_loss(model)
            loss.backward()
            optimizer.step()

        running += task.item()
        pbar.set_postfix_str(f"loss={task.item():.5f}  avg={running/(pbar.n+1):.5f}")

    return running / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    pbar = tqdm(loader, desc="  val  ", leave=False, ncols=100,
                bar_format='{l_bar}{bar:22}{r_bar}')

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        running += loss.item()
        pbar.set_postfix_str(f"loss={loss.item():.5f}  avg={running/(pbar.n+1):.5f}")

    return running / max(len(loader), 1)


# ============================================================
# CHECKPOINT / FILE HELPERS
# ============================================================
def build_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                     best_val_loss, config, normalizer):
    return {
        'epoch': epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss':   val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
        'image_dimensions': {'height': config['image_height'],
                             'width':  config['image_width']},
        'normalization_params': normalizer.norm_params if normalizer else None,
        'timestamp': datetime.now().isoformat(),
    }


def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def get_base_train_dir(config):
    root = Path(config['train_dir'])
    aug = 'augmented' if config.get('use_augmentation', False) else 'non_augmented'
    return root / aug / config.get('resolution', 'unknown')


def get_model_unique_name(config):
    loss = config['loss_function'].upper()
    if loss == 'HUBER':
        loss = f"HUBER_delta{config.get('huber_delta', 1.0)}"
    elif loss == 'RAIN_MSE':
        loss = f"RAINMSE_w{config.get('rain_weight', 25)}"

    aug = f"_aug{config['augmentation_factor']}x" if config.get('use_augmentation') else ""
    nrm = "_norm" if config.get('normalize_data') else ""

    name = (f"SA_IRUNet_{loss}"
            f"_in{config.get('in_channels', 'auto')}"
            f"_out{config.get('out_channels', 'auto')}{aug}{nrm}")
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def save_training_log(log_dir, row):
    log_file = Path(log_dir) / 'training_log.csv'
    exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate',
                        'best_val_loss', 'lr_reduction_count',
                        'no_improve_count', 'loss_improvement',
                        'min_delta_met', 'timestamp'])
        w.writerow(row)


def plot_training_curves(log_dir, train_losses, val_losses):
    if not train_losses or not val_losses:
        return
    tr = np.nan_to_num(np.array(train_losses), nan=1e6, posinf=1e6, neginf=1e6)
    vl = np.nan_to_num(np.array(val_losses),   nan=1e6, posinf=1e6, neginf=1e6)
    ep = np.arange(1, len(tr) + 1)

    plt.figure(figsize=(10, 6))
    if tr.max() / (tr.min() + 1e-8) > 1000:
        plt.yscale('log')
        ylabel = 'Loss (log)'
    else:
        ylabel = 'Loss'

    plt.plot(ep, tr, 'b-', label='Train', linewidth=2)
    plt.plot(ep, vl, 'r-', label='Val',   linewidth=2)
    best_idx = int(np.argmin(vl))
    plt.plot(best_idx + 1, vl[best_idx], 'ro', markersize=8,
             label=f'Best val = {vl[best_idx]:.6f}')
    plt.xlabel('Epoch'); plt.ylabel(ylabel)
    plt.title('SA-IRUNet - training curves')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(Path(log_dir) / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def find_existing_model_with_same_config(base_train_dir, current_config):
    base = Path(base_train_dir)
    if not base.exists():
        return None

    keys = [
        'data_dir', 'batch_size', 'epochs', 'learning_rate', 'weight_decay',
        'loss_function', 'huber_delta', 'rain_threshold', 'rain_weight',
        'num_workers', 'pin_memory', 'prefetch_factor',
        'scheduler_patience', 'scheduler_factor', 'scheduler_min_delta',
        'early_stop_patience', 'min_lr',
        'in_channels', 'out_channels', 'n_filters', 'dropout',
        'kernel_size', 'expand_ratio',
        'use_augmentation', 'augmentation_factor', 'augment_validation',
        'depthwise_lambda', 'use_unified_lambda',
        'normalize_data', 'normalization_range',
    ]
    current = {k: current_config[k] for k in keys if k in current_config}

    for d in base.iterdir():
        if not d.is_dir():
            continue
        cfg = d / 'config.json'
        if not cfg.exists():
            continue
        with open(cfg, 'r') as f:
            existing = json.load(f)
        comparable = {k: existing[k] for k in keys if k in existing}
        if comparable == current:
            return d
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    Log.header("SA-IRUNet training pipeline")

    # ---------- data discovery ----------
    train_data_raw = load_split_data(Path(CONFIG['data_dir']) / 'train')
    image_info = get_image_shape_from_data(train_data_raw)

    if CONFIG['in_channels'] is None:
        CONFIG['in_channels'] = image_info['input_channels']
    if CONFIG['out_channels'] is None:
        CONFIG['out_channels'] = image_info['output_channels']

    CONFIG['image_height'] = image_info['height']
    CONFIG['image_width']  = image_info['width']
    CONFIG['resolution']   = image_info['resolution']
    CONFIG['max_lead_steps'] = CONFIG['out_channels']

    Log.section("Data")
    Log.kv("Data root",       CONFIG['data_dir'])
    Log.kv("Resolution",      CONFIG['resolution'])
    Log.kv("Input channels",  CONFIG['in_channels'])
    Log.kv("Output channels", CONFIG['out_channels'])

    # ---------- normalisation ----------
    normalizer = None
    if CONFIG.get('normalize_data'):
        normalizer = DataNormalizer(CONFIG['normalization_range'], CONFIG['epsilon'])
        normalizer.fit(train_data_raw['features'], train_data_raw['labels'])

        lab_norm = normalizer.transform_labels(train_data_raw['labels'])
        thr = CONFIG.get('rain_threshold', 0.05)
        wet_pct = float((lab_norm > thr).mean() * 100)

        Log.section("Normalisation")
        p = normalizer.norm_params
        Log.kv("Features range",  f"[{p['feat_min']:.4f}, {p['feat_max']:.4f}]")
        Log.kv("Labels range",    f"[{p['label_min']:.4f}, {p['label_max']:.4f}]")
        Log.kv("Output range",    f"[{p['norm_min']}, {p['norm_max']}]")
        Log.kv("Label mean/std",  f"{lab_norm.mean():.4f} / {lab_norm.std():.4f}")
        Log.kv(f"Wet pixels (>{thr})", f"{wet_pct:.3f}%")
        if wet_pct < 3.0:
            Log.kv("Note", "wet fraction < 3% - rain-weighted loss recommended")

    # ---------- output dir ----------
    base_train_dir = get_base_train_dir(CONFIG)
    existing = find_existing_model_with_same_config(base_train_dir, CONFIG)
    if existing is not None:
        Log.section("Existing run detected")
        Log.kv("Directory", str(existing))
        if input("  Continue training anyway? (y/n): ").strip().lower() not in ('y', 'yes'):
            print("Training halted by user.")
            sys.exit(0)

    model_name = get_model_unique_name(CONFIG)
    model_dir  = base_train_dir / model_name
    ckpt_dir   = model_dir / 'checkpoints'
    logs_dir   = model_dir / 'logs'
    for d in (model_dir, ckpt_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, model_dir / 'config.json')
    if normalizer:
        normalizer.save_params(model_dir / 'norm_params.json')
        normalizer.save_params(Path(CONFIG['data_dir']) / 'norm_params.json')

    # ---------- environment ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Log.section("Environment")
    Log.kv("Platform",  platform.platform())
    Log.kv("PyTorch",   torch.__version__)
    Log.kv("Device",    str(device))
    if torch.cuda.is_available():
        Log.kv("GPU",         torch.cuda.get_device_name(0))
        Log.kv("GPU memory",  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    Log.kv("Mixed precision", "on" if CONFIG['use_mixed_precision'] else "off")
    Log.kv("Output dir",      str(model_dir))

    # ---------- model ----------
    Log.section("Model")
    model = SAIRUNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        n_filters=CONFIG['n_filters'],
        dropout=CONFIG['dropout'],
        kernel_size=CONFIG['kernel_size'],
        expand_ratio=CONFIG['expand_ratio'],
    ).to(device)

    input_shape = (CONFIG['in_channels'], CONFIG['image_height'], CONFIG['image_width'])
    with torch.no_grad():
        dummy_out = model(torch.randn(1, *input_shape, device=device))
    assert dummy_out.shape[1] == CONFIG['out_channels']

    Log.kv("Architecture",   "SA-IRUNet")
    Log.kv("Input shape",    f"(B, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})")
    Log.kv("Output shape",   f"(B, {dummy_out.shape[1]}, "
                             f"{dummy_out.shape[2]}, {dummy_out.shape[3]})")

    metrics = calculate_model_metrics(model, input_shape, device)
    Log.section("Model statistics")
    Log.kv("Total parameters",  f"{metrics['total_parameters']:,} "
                                f"({metrics['total_parameters']/1e6:.2f} M)")
    Log.kv("Trainable",         f"{metrics['trainable_parameters']:,}")
    Log.kv("Non-trainable",     f"{metrics['non_trainable_parameters']:,}")
    Log.kv("Model size",        f"{metrics['model_size_mb']:.2f} MB")
    Log.kv("FLOPs",             f"{metrics['flops_giga']:.2f} G")
    Log.kv("MACs",              f"{metrics['macs_giga']:.2f} G")
    Log.kv("Activation memory", f"{metrics['activation_memory_mb']:.2f} MB (b=1)")

    with open(model_dir / 'model_metrics.txt', 'w') as f:
        write_model_metrics(metrics, f)
    with open(model_dir / 'model_metrics.json', 'w') as f:
        json.dump({k: v for k, v in metrics.items()
                   if k not in ('layer_breakdown', 'block_summary')}, f, indent=4)

    # ---------- datasets ----------
    val_data_raw  = load_split_data(Path(CONFIG['data_dir']) / 'val')
    test_data_raw = load_split_data(Path(CONFIG['data_dir']) / 'test')

    if normalizer:
        tr_f, tr_l = normalizer.transform(train_data_raw['features'], train_data_raw['labels'])
        va_f, va_l = normalizer.transform(val_data_raw['features'],   val_data_raw['labels'])
        te_f, te_l = normalizer.transform(test_data_raw['features'],  test_data_raw['labels'])
        train_data = {'features': tr_f, 'labels': tr_l}
        val_data   = {'features': va_f, 'labels': va_l}
        test_data  = {'features': te_f, 'labels': te_l}
    else:
        train_data, val_data, test_data = train_data_raw, val_data_raw, test_data_raw

    n_train, n_val, n_test = (len(train_data['features']),
                              len(val_data['features']),
                              len(test_data['features']))

    if CONFIG['use_augmentation']:
        ef, el = expand_dataset_with_augmentation(
            train_data['features'], train_data['labels'],
            augmentation_factor=CONFIG['augmentation_factor'],
            augmentation_ops=CONFIG['augmentation_ops'])
        train_data['features'] = ef
        train_data['labels']   = el

    n_train_final = len(train_data['features'])

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, CONFIG)

    Log.section("Dataset")
    Log.kv("Training samples",   f"{n_train_final}"
           + (f" (augmented from {n_train})" if CONFIG['use_augmentation'] else ""))
    Log.kv("Validation samples", n_val)
    Log.kv("Test samples",       n_test)
    Log.kv("Batch size",         CONFIG['batch_size'])
    Log.kv("Batches per epoch",  len(train_loader))

    # ---------- loss / optimiser ----------
    criterion, loss_str = build_loss(CONFIG)
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda') if (
        torch.cuda.is_available() and CONFIG['use_mixed_precision']) else None

    max_lr_reductions = max(1, CONFIG['early_stop_patience'] // CONFIG['scheduler_patience'])

    Log.section("Training configuration")
    Log.kv("Loss",           loss_str)
    Log.kv("Optimizer",      f"Adam (lr={CONFIG['learning_rate']:.2e}, "
                             f"wd={CONFIG['weight_decay']:.2e})")
    Log.kv("LR scheduler",   f"patience={CONFIG['scheduler_patience']}  "
                             f"factor={CONFIG['scheduler_factor']}  "
                             f"min_delta={CONFIG['scheduler_min_delta']:.1e}  "
                             f"min_lr={CONFIG['min_lr']:.1e}")
    Log.kv("Early stopping", f"after {max_lr_reductions} LR reductions")
    Log.kv("Epochs",         CONFIG['epochs'])

    # ============================================================
    # Training loop
    # ============================================================
    state = {
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'no_improve_count': 0,
        'lr_reductions': 0,
        'train_losses': [],
        'val_losses': [],
    }
    start_time = time.time()

    Log.section("Training progress")
    header = (f"  {'Ep':>4} {'Train':>10} {'Val':>10} {'Gap':>11} "
              f"{'Best':>10} {'LR':>10} {'NImp':>8} {'LRR':>6} {'Time':>7}  Events")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start = time.time()
        clear_gpu_memory()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss   = validate_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        state['train_losses'].append(train_loss)
        state['val_losses'].append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        gap = train_loss - val_loss

        if len(state['val_losses']) >= 2:
            improvement = state['val_losses'][-2] - val_loss
            delta_met = improvement > CONFIG['scheduler_min_delta']
        else:
            improvement = 0.0
            delta_met = True

        events = []

        # ---- best model ----
        if val_loss < state['best_val_loss']:
            state['best_val_loss'] = val_loss
            state['best_epoch'] = epoch
            state['no_improve_count'] = (0 if improvement > CONFIG['scheduler_min_delta']
                                         else state['no_improve_count'] + 1)
            torch.save(build_checkpoint(model, optimizer, epoch,
                                        train_loss, val_loss,
                                        state['best_val_loss'],
                                        CONFIG, normalizer),
                       ckpt_dir / 'best_model.pth')
            events.append("BEST")
        else:
            state['no_improve_count'] = (state['no_improve_count'] + 1
                                         if improvement <= CONFIG['scheduler_min_delta']
                                         else 0)

        # ---- LR scheduler ----
        if state['no_improve_count'] >= CONFIG['scheduler_patience']:
            old_lr = current_lr
            new_lr = max(old_lr * CONFIG['scheduler_factor'], CONFIG['min_lr'])
            if new_lr < old_lr:
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr
                state['lr_reductions'] += 1
                state['no_improve_count'] = 0
                events.append(f"LR {old_lr:.1e}->{new_lr:.1e}")

        current_lr = optimizer.param_groups[0]['lr']

        save_training_log(logs_dir, [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{current_lr:.6e}", f"{state['best_val_loss']:.6f}",
            state['lr_reductions'], state['no_improve_count'],
            f"{improvement:.6e}", delta_met,
            datetime.now().isoformat(),
        ])

        # ---- verbose epoch line ----
        verbose = CONFIG.get('verbose_epoch_line', True)
        every_n = max(1, CONFIG.get('verbose_every_n', 1))
        on_event = CONFIG.get('verbose_on_events', True) and bool(events)

        if verbose and ((epoch % every_n == 0) or on_event):
            ev = " ".join(f"[{e}]" for e in events)
            print(f"  {epoch:>4} "
                  f"{train_loss:>10.6f} {val_loss:>10.6f} "
                  f"{gap:>+11.6f} "
                  f"{state['best_val_loss']:>10.6f} "
                  f"{current_lr:>10.2e} "
                  f"{state['no_improve_count']:>3}/{CONFIG['scheduler_patience']:<4} "
                  f"{state['lr_reductions']:>2}/{max_lr_reductions:<3} "
                  f"{epoch_time:>6.1f}s  {ev}")

        if state['lr_reductions'] >= max_lr_reductions:
            print(f"\n  Early stopping at epoch {epoch} (max LR reductions reached).")
            break

    total_time = time.time() - start_time
    final_epoch = len(state['train_losses'])

    torch.save(build_checkpoint(model, optimizer, final_epoch,
                                state['train_losses'][-1], state['val_losses'][-1],
                                state['best_val_loss'], CONFIG, normalizer),
               ckpt_dir / 'final_model.pth')

    plot_training_curves(logs_dir, state['train_losses'], state['val_losses'])

    # ---------- training trajectory ----------
    Log.section("Training trajectory")
    tl, vl = state['train_losses'], state['val_losses']
    Log.kv("Train loss first/last", f"{tl[0]:.6f}  ->  {tl[-1]:.6f}")
    Log.kv("Val loss first/last",   f"{vl[0]:.6f}  ->  {vl[-1]:.6f}")
    Log.kv("Best val loss",         f"{state['best_val_loss']:.6f} "
                                    f"(epoch {state['best_epoch']})")
    Log.kv("LR reductions used",    f"{state['lr_reductions']}/{max_lr_reductions}")

    # ---------- test ----------
    ckpt_path = ckpt_dir / 'best_model.pth'
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing best model", leave=False, ncols=100)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            test_loss += criterion(model(x), y).item()
            pbar.set_postfix_str(f"avg={test_loss/(pbar.n+1):.5f}")
    test_loss /= max(len(test_loader), 1)

    # ---------- final summary ----------
    Log.header("Training complete")
    Log.section("Results")
    Log.kv("Best validation loss", f"{state['best_val_loss']:.6f} "
                                   f"(epoch {state['best_epoch']})")
    Log.kv("Test loss",            f"{test_loss:.6f}")
    Log.kv("Epochs completed",     final_epoch)
    Log.kv("Total time",           f"{total_time/60:.2f} min "
                                   f"({total_time/final_epoch:.1f} s/epoch)")
    Log.kv("LR reductions",        f"{state['lr_reductions']}/{max_lr_reductions}")

    Log.section("Checkpoints")
    Log.kv("Best",  str(ckpt_dir / 'best_model.pth'))
    Log.kv("Final", str(ckpt_dir / 'final_model.pth'))

    summary = {
        'model_name': 'SA-IRUNet',
        'in_channels': CONFIG['in_channels'],
        'out_channels': CONFIG['out_channels'],
        'unique_folder_name': model_name,
        'loss_function': CONFIG['loss_function'],
        'loss_description': loss_str,
        'total_training_time_seconds': total_time,
        'total_epochs_completed': final_epoch,
        'best_val_loss': state['best_val_loss'],
        'best_epoch': state['best_epoch'],
        'test_loss': test_loss,
        'total_parameters': metrics['total_parameters'],
        'trainable_parameters': metrics['trainable_parameters'],
        'normalization_enabled': CONFIG.get('normalize_data', False),
        'normalization_type': 'minmax' if CONFIG.get('normalize_data') else 'none',
        'scheduler_settings': {
            'patience': CONFIG['scheduler_patience'],
            'factor':   CONFIG['scheduler_factor'],
            'min_delta': CONFIG['scheduler_min_delta'],
            'min_lr':   CONFIG['min_lr'],
            'total_reductions': state['lr_reductions'],
            'max_reductions':   max_lr_reductions,
        },
        'checkpoints': {
            'best_model':  str(ckpt_dir / 'best_model.pth'),
            'final_model': str(ckpt_dir / 'final_model.pth'),
        },
        'config': CONFIG,
    }
    with open(model_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()