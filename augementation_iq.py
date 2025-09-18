import torch
import numpy as np
import torch
import torch.nn.functional as F
def rotate_signal(signal: torch.Tensor, degree: float) -> torch.Tensor:
    """Rotate I/Q signal counterclockwise by given degree (0, 90, 180, 270)."""
    rad = np.deg2rad(degree)
    cos_theta = np.cos(rad)
    sin_theta = np.sin(rad)

    rot_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ], dtype=signal.dtype, device=signal.device)

    rotated = torch.matmul(rot_matrix, signal.squeeze(0))  # [2, 512]
    return rotated.unsqueeze(0)  # [1, 2, 512]

def add_gaussian_noise(signal: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise with std to the signal."""
    noise = torch.randn_like(signal) * std
    return signal + noise

def flip_horizontal_signal(signal: torch.Tensor) -> torch.Tensor:
    """Flip horizontally: I → -I"""
    flipped = signal.clone()
    flipped[:, 0, :] = -flipped[:, 0, :]  # Flip I
    return flipped

def flip_vertical_signal(signal: torch.Tensor) -> torch.Tensor:
    """Flip vertically: Q → -Q"""
    flipped = signal.clone()
    flipped[:, 1, :] = -flipped[:, 1, :]  # Flip Q
    return flipped

def flip_both_signal(signal: torch.Tensor) -> torch.Tensor:
    """Flip both I and Q."""
    flipped = signal.clone()
    flipped[:, 0, :] = -flipped[:, 0, :]
    flipped[:, 1, :] = -flipped[:, 1, :]
    return flipped
from scipy.interpolate import CubicSpline
import numpy as np

def magnitude_warp_signal(signal: torch.Tensor, num_knots: int = 4, std: float = 0.2) -> torch.Tensor:
    """Apply magnitude warping using cubic spline through Gaussian-sampled anchor points."""
    B, C, N = signal.shape
    device = signal.device

    knot_x = np.linspace(0, N - 1, num=num_knots)
    knot_y = np.random.normal(loc=1.0, scale=std, size=(num_knots,))
    cs = CubicSpline(knot_x, knot_y)
    warp_curve = cs(np.arange(N)).astype(np.float32)
    warp_curve = torch.tensor(warp_curve, device=device).view(1, 1, N)

    return signal * warp_curve

def scale_signal(signal: torch.Tensor, mean: float = 1.0, std: float = 0.1) -> torch.Tensor:
    """Apply scaling augmentation: multiply signal by scalar from N(mean, std)."""
    device = signal.device
    r = torch.normal(mean, std, size=(1, 1, 1), device=device)
    return signal * r

from scipy.interpolate import interp1d

def time_warp_signal(signal: torch.Tensor, warp_factor: float) -> torch.Tensor:
    """
    Apply time warping to an IQ signal.

    Parameters:
    - signal: torch.Tensor of shape [1, 2, N], where 2 corresponds to [I, Q]
    - warp_factor: float, factor by which to warp the time axis (>1.0 stretches, <1.0 compresses)

    Returns:
    - warped_signal: torch.Tensor of shape [1, 2, N]
    """
    _, _, N = signal.shape
    device = signal.device

    # Original time indices
    orig_indices = torch.linspace(0, 1, steps=N, device=device)

    # Warped time indices
    warped_indices = torch.linspace(0, 1, steps=int(N * warp_factor), device=device)
    warped_indices = torch.clamp(warped_indices, 0, 1)

    # Interpolate I and Q channels separately
    warped_I = F.interpolate(signal[:, 0:1, :], size=warped_indices.shape[0], mode='linear', align_corners=False)
    warped_Q = F.interpolate(signal[:, 1:2, :], size=warped_indices.shape[0], mode='linear', align_corners=False)

    # Resize back to original length
    warped_I = F.interpolate(warped_I, size=N, mode='linear', align_corners=False)
    warped_Q = F.interpolate(warped_Q, size=N, mode='linear', align_corners=False)

    # Concatenate I and Q channels
    warped_signal = torch.cat([warped_I, warped_Q], dim=1)

    return warped_signal

#----------------------------------------------------------------
import random
class SignalAugmenter:
    def __init__(self, augmentations=None, apply_prob=1.0, mode='all'):
        """
        augmentations: list of functions (or dicts with params if needed)
        apply_prob: overall probability of applying augmentation(s)
        mode: 'all' = apply each augmentation with apply_prob
              'one' = apply one randomly selected augmentation
        """
        self.augmentations = augmentations or []
        self.apply_prob = apply_prob
        self.mode = mode.lower()

    def __call__(self, signal: torch.Tensor):
        if random.random() > self.apply_prob or not self.augmentations:
            return signal  # Skip augmentation completely

        if self.mode == 'one':
            aug = random.choice(self.augmentations)
            return aug(signal)
        elif self.mode == 'all':
            for aug in self.augmentations:
                if random.random() < self.apply_prob:
                    signal = aug(signal)
            return signal
        else:
            raise ValueError("Invalid mode. Choose 'all' or 'one'.")
        

AUGMENTATION_FUNCTIONS = {
    "scale": lambda x: scale_signal(
        x,
        mean=random.uniform(0.9, 1.1),
        std=random.uniform(0.01, 0.4)
    ),
    "gaussian_noise": lambda x: add_gaussian_noise(
        x, std=random.choice([.05, .04, 0.03, 0.02, 0.01])
       # x, std=random.choice([ 0.02, 0.01])
    ),
    "rotate": lambda x: rotate_signal(
        x, degree=random.choice([90, 180, 270])
    ),
    "flip_horizontal": lambda x: flip_horizontal_signal(x),
    "flip_vertical": lambda x: flip_vertical_signal(x),
    "time_warp": lambda x: time_warp_signal(
        x, warp_factor=random.uniform(0.1, 0.5)
    ),
    "magnitude_warp": lambda x: magnitude_warp_signal(
        x, num_knots=random.randint(4, 6), std=random.uniform(0.1, 0.4)
    ),
}
    

def build_transform(args, aug_list_name):
    """
    Create a SignalAugmenter transform based on args and selected augmentation set name.

    Args:
        args (Namespace): Parsed arguments with attributes:
            - args.Aug_prob: probability of applying augmentation
            - args.Aug_type: 'one' or 'all'
            - args.augmentation: a dict with keys like 'weak' or 'full', each a list of aug names
        aug_list_name (str): Key from args.augmentation, e.g., 'weak' or 'full'

    Returns:
        SignalAugmenter or None
    """
    if args.Aug_prob == 0 or aug_list_name not in args.augmentation:
        return None

    selected_augs = args.augmentation[aug_list_name]

    aug_list = []
    for name in selected_augs:
        if name in AUGMENTATION_FUNCTIONS:
            aug_list.append(AUGMENTATION_FUNCTIONS[name])
        else:
            print(f"⚠️ Unknown augmentation: {name}")

    transform = SignalAugmenter(
        augmentations=aug_list,
        apply_prob=args.Aug_prob,
        mode=args.Aug_type
    )
    return transform

        