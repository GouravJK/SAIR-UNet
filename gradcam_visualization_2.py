# gradcam_visualization.py
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from typing import List, Optional, Tuple
import warnings

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ====================== IMPORTS ======================
try:
    from IR_UNET import SAIRUNet
except ImportError:
    print("ERROR: IR_UNET module not found.")
    exit(1)

# ====================== HELPERS ======================
def select_directory_gui(title: str) -> Optional[Path]:
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(selected_dir) if selected_dir else None

class DataNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.norm_params = None
    def load_params(self, load_path):
        with open(load_path, 'r') as f:
            self.norm_params = json.load(f)
        return self.norm_params
    def transform_features(self, features):
        if self.norm_params is None:
            raise ValueError("Normalizer not fitted.")
        feat_range = self.norm_params['feat_max'] - self.norm_params['feat_min']
        features_norm = (features - self.norm_params['feat_min']) / (feat_range + self.epsilon)
        features_norm = features_norm * (self.norm_params['norm_max'] - self.norm_params['norm_min']) + self.norm_params['norm_min']
        return features_norm

class CaseStudyDataset:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        features_file = self.data_path / 'features.npz'
        labels_file = self.data_path / 'labels.npz'
        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"Required files not found in {data_path}")
        self.features = np.load(features_file)['features']
        self.labels = np.load(labels_file)['labels']
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = np.nan_to_num(self.labels, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = np.maximum(self.features, 0)
        self.labels = np.maximum(self.labels, 0)
        print(f"Dataset: {self.features.shape[0]} samples, features {self.features.shape}")

def load_sairunet_model(model_dir: Path, checkpoint_path: Path, device, config: dict):
    print("\n" + "="*60)
    print("LOADING SAIRUNet MODEL")
    print("="*60)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Use parameters from config (these must match the training run)
    in_ch = config.get('in_channels', 3)
    out_ch = config.get('out_channels', 6)
    n_f = config.get('n_filters', 16)
    d = config.get('dropout', 0.05)
    ks = config.get('kernel_size', 3)          # from your config: 3
    expand_ratio = config.get('expand_ratio', 4) # from your config: 4
    
    print(f"Model Configuration:")
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
    
    # Get the state dict from checkpoint (might have 'module.' prefix)
    state = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
        print("  Removed 'module.' prefix from state dict")
    
    # Filter out keys that don't match the current model's shape
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for k, v in state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(f"{k} (checkpoint: {v.shape}, model: {model_state[k].shape})")
        else:
            skipped_keys.append(f"{k} (not in current model)")
    
    if skipped_keys:
        print(f"  Warning: Skipped {len(skipped_keys)} keys due to shape mismatch or absence:")
        for msg in skipped_keys[:5]:  # show first 5
            print(f"    - {msg}")
        if len(skipped_keys) > 5:
            print(f"    ... and {len(skipped_keys)-5} more.")
    
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("="*60)
    return model

def discover_layers(model, prefix=""):
    layers = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if len(list(module.children())) == 0:
            layers.append((full_name, module))
        else:
            layers.extend(discover_layers(module, full_name))
    return layers

def print_model_structure(model):
    print("\n" + "="*60)
    print("MODEL STRUCTURE (only Conv2d/ConvTranspose2d shown)")
    print("="*60)
    layers = discover_layers(model)
    conv_layers = [(i, name, mod) for i, (name, mod) in enumerate(layers) 
                   if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))]
    for i, name, mod in conv_layers:
        print(f"  {i:3d}: {name} -> {mod.__class__.__name__}")
    print("="*60)
    return layers, conv_layers

def get_target_layers(model, layer_indices=None):
    """Return only Conv2d/ConvTranspose2d layers, either by index or auto."""
    layers = discover_layers(model)
    conv_layers = [(i, name, mod) for i, (name, mod) in enumerate(layers) 
                   if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))]
    
    if layer_indices is not None:
        selected = []
        for idx in layer_indices:
            matched = [(i, name, mod) for i, name, mod in conv_layers if i == idx]
            if matched:
                selected.append((matched[0][1], matched[0][2]))  # (name, module)
            else:
                print(f"Warning: index {idx} is not a conv layer or out of range.")
        return selected
    
    # Auto-select: pick one early, one middle, one late conv layer
    if len(conv_layers) >= 3:
        idxs = [0, len(conv_layers)//2, -1]
        selected = []
        for idx in idxs:
            i, name, mod = conv_layers[idx]
            selected.append((name, mod))
            print(f"  Auto-selected: {name}")
        return selected
    else:
        # fallback: all conv layers
        return [(name, mod) for _, name, mod in conv_layers]

# ====================== MAIN ======================
def run_gradcam_visualization():
    print("\n" + "="*70)
    print("GRAD-CAM VISUALIZATION FOR SAIRUNet")
    print("="*70)
    
    model_dir = select_directory_gui("Select Model Root Directory")
    if model_dir is None:
        return
    data_dir = select_directory_gui("Select Case Study Data Directory")
    if data_dir is None:
        return
    
    # Load config
    config_file = model_dir / 'config.json'
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {'in_channels':3, 'out_channels':6, 'n_filters':16, 'dropout':0.05, 'kernel_size':3, 'expand_ratio':4}
    
    # Find checkpoint
    checkpoint_dir = model_dir / 'checkpoints'
    best_model = checkpoint_dir / 'best_model.pth'
    if not best_model.exists():
        best_model = checkpoint_dir / 'final_model.pth'
        if not best_model.exists():
            pth_files = list(checkpoint_dir.glob('*.pth'))
            best_model = pth_files[0] if pth_files else None
    if best_model is None:
        print("ERROR: No model checkpoint found.")
        return
    
    model = load_sairunet_model(model_dir, best_model, DEVICE, config)
    
    # Show only conv layers
    all_layers, conv_layers = print_model_structure(model)
    
    # User selection
    print("\nSelect target layers (indices from the list above, comma-separated)")
    print("or press Enter for automatic selection (recommended).")
    user_input = input("Enter indices or press Enter: ").strip()
    
    if user_input:
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            target_pairs = get_target_layers(model, indices)
        except:
            target_pairs = get_target_layers(model)
    else:
        target_pairs = get_target_layers(model)
    
    if not target_pairs:
        print("No valid conv layers selected. Exiting.")
        return
    
    target_layers = [mod for _, mod in target_pairs]
    layer_names = [name for name, _ in target_pairs]
    print(f"\nSelected {len(target_layers)} layers:")
    for name in layer_names:
        print(f"  - {name}")
    
    # Data
    dataset = CaseStudyDataset(data_dir)
    norm_file = model_dir / 'norm_params.json'
    if norm_file.exists():
        normalizer = DataNormalizer()
        normalizer.load_params(norm_file)
        features = dataset.features
        orig_shape = features.shape
        features_flat = features.reshape(-1, *orig_shape[2:])
        features_norm = normalizer.transform_features(features_flat)
        dataset.features = features_norm.reshape(orig_shape)
    
    sample_idx = 0
    if len(dataset.features) > 1:
        try:
            sample_idx = int(input(f"Sample index (0-{len(dataset.features)-1}): ") or "0")
        except:
            sample_idx = 0
    input_data = dataset.features[sample_idx]
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True
    
    out_dir = data_dir / 'gradcam_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_channels = config.get('out_channels', 6)
    lead_idx = 0
    if out_channels > 1:
        try:
            lead_idx = int(input(f"Lead time index (0-{out_channels-1}): ") or "0")
        except:
            lead_idx = 0
    
    # Target for regression
    class ChannelOutputTarget:
        def __init__(self, channel_idx):
            self.channel_idx = channel_idx
        def __call__(self, model_output):
            return model_output[0, self.channel_idx, :, :].sum()
    targets = [ChannelOutputTarget(lead_idx)]
    
    # Use EigenCAM by default (no gradients, more robust)
    print("\nUsing EigenCAM (robust for regression tasks).")
    cam_method = EigenCAM
    
    # Prepare base image
    base_image = input_tensor[0, 0, ...].cpu().detach().numpy()
    base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min() + 1e-8)
    base_image_rgb = np.stack([base_image]*3, axis=-1)
    
    # ========== MODIFIED PLOTTING: OVERLAY ONLY ==========
    # Determine grid columns based on number of layers
    num_layers = len(target_layers)
    if num_layers % 3 == 0:
        cols = 3
    elif num_layers % 2 == 0:
        cols = 4
    else:
        cols = min(2, num_layers) if num_layers > 1 else 1

    rows = (num_layers + cols - 1) // cols

    # Create figure with one subplot per layer
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.5, rows * 3.5),
                             squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Loop over layers and plot only the overlay
    for idx, (layer_name, layer_module) in enumerate(zip(layer_names, target_layers)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        print(f"\nProcessing: {layer_name}")
        # Hook to show output shape (optional)
        def hook_fn(module, inp, out):
            print(f"  Output shape: {out.shape}")
        handle = layer_module.register_forward_hook(hook_fn)

        try:
            with cam_method(model=model, target_layers=[layer_module]) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # (H, W)
                visualization = show_cam_on_image(base_image_rgb, grayscale_cam, use_rgb=True)

                # Display overlay
                ax.imshow(visualization)
                ax.set_title(layer_name, fontsize=9)
                ax.axis('off')

                # Save individual images (heatmap + overlay)
                safe_name = layer_name.replace('.', '_')
                plt.imsave(out_dir / f'heatmap_{safe_name}.png', grayscale_cam, cmap='jet')
                plt.imsave(out_dir / f'overlay_{safe_name}.png', visualization)
                print(f"  Saved heatmap_{safe_name}.png and overlay_{safe_name}.png")

        except Exception as e:
            print(f"  ERROR: {e}")
            ax.text(0.5, 0.5, "Error", ha='center', va='center')
            ax.axis('off')
        finally:
            handle.remove()

    # Hide any unused subplots in the last row
    for r in range(rows):
        for c in range(cols):
            if r * cols + c >= num_layers:
                axes[r, c].axis('off')

    # Add a single y-label for "Overlay" on the left margin (only if rows>0)
    if rows > 0:
        axes[0, 0].set_ylabel("Overlay", fontsize=10, labelpad=10)

    plt.suptitle(f'EigenCAM Overlays - Sample {sample_idx}, Lead {lead_idx}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f'gradcam_overlays_sample_{sample_idx}_lead_{lead_idx}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    # ========== END OF MODIFIED SECTION ==========
    
    # Prediction reference (unchanged)
    print("\nGenerating prediction...")
    with torch.no_grad():
        output = model(input_tensor)
        pred = output[0, lead_idx, ...].cpu().numpy()
    gt = dataset.labels[sample_idx, lead_idx, ...] if dataset.labels.shape[1] > lead_idx else None
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(base_image, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(pred, cmap='turbo')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    if gt is not None:
        axes[2].imshow(gt, cmap='turbo')
        axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f'prediction_sample_{sample_idx}_lead_{lead_idx}.png', dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("DONE! Results saved to:", out_dir)
    print("="*70)

if __name__ == "__main__":
    run_gradcam_visualization()