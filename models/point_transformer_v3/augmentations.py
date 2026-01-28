import numpy as np
from typing import Tuple


def normalize_center_scale(points: np.ndarray) -> np.ndarray:
    xyz = points.astype(np.float32)
    center = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - center
    denom = np.max(np.linalg.norm(xyz, axis=1))
    if denom > 0:
        xyz = xyz / denom
    return xyz


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    xyz = points.astype(np.float32)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return xyz @ R.T


def random_flip_xy(points: np.ndarray, px: float = 0.5, py: float = 0.5) -> np.ndarray:
    xyz = points.astype(np.float32)
    if np.random.rand() < px:
        xyz[:, 0] = -xyz[:, 0]
    if np.random.rand() < py:
        xyz[:, 1] = -xyz[:, 1]
    return xyz


def random_scale(points: np.ndarray, s_min: float = 0.9, s_max: float = 1.1) -> np.ndarray:
    xyz = points.astype(np.float32)
    s = np.random.uniform(float(s_min), float(s_max))
    return xyz * s


def jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.03) -> np.ndarray:
    xyz = points.astype(np.float32)
    noise = np.clip(float(sigma) * np.random.randn(*xyz.shape), -float(clip), float(clip)).astype(np.float32)
    return xyz + noise


def apply_randla_augmentations(points: np.ndarray, config=None) -> np.ndarray:
    """RandLA-Net style augmentations: normalize, Z-rotate, flip X/Y, uniform scale, jitter."""
    xyz = normalize_center_scale(points)
    xyz = random_rotate_z(xyz)
    xyz = random_flip_xy(xyz)
    smin = getattr(config, 'augment_scale_min', 0.9) if config is not None else 0.9
    smax = getattr(config, 'augment_scale_max', 1.1) if config is not None else 1.1
    xyz = random_scale(xyz, s_min=smin, s_max=smax)
    sigma = getattr(config, 'augment_noise', 0.01) if config is not None else 0.01
    xyz = jitter(xyz, sigma=sigma, clip=0.03)
    return xyz


def kpconv_vertical_augment(points: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KPConv-style simple vertical rotation + anisotropic scale + symmetries + noise.
    Returns (points, scale, R) to match existing interfaces.
    """
    R = np.eye(points.shape[1], dtype=np.float32)
    if points.shape[1] == 3 and getattr(config, 'augment_rotation', 'vertical') == 'vertical':
        theta = np.random.rand() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    min_s = getattr(config, 'augment_scale_min', 0.9)
    max_s = getattr(config, 'augment_scale_max', 1.1)
    if getattr(config, 'augment_scale_anisotropic', True):
        scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
    else:
        scale = np.random.rand() * (max_s - min_s) + min_s
    sym = np.array(getattr(config, 'augment_symmetries', [True, False, False]), dtype=np.int32)
    sym *= np.random.randint(2, size=points.shape[1])
    scale = (scale * (1 - sym * 2)).astype(np.float32)
    noise_sigma = getattr(config, 'augment_noise', 0.005)
    pts = points.astype(np.float32) @ R.T
    pts = pts * scale
    if noise_sigma > 0:
        pts = pts + np.random.normal(scale=noise_sigma, size=pts.shape).astype(np.float32)
    return pts, np.array(scale, dtype=np.float32), R


def default_augmentation(points: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Default augmentation entry point used across models.
    Returns (points, scale, R). For RandLA-style, scale=ones and R=identity.
    """
    if getattr(config, 'use_randla_augs', False):
        pts = apply_randla_augmentations(points, config)
        scale = np.array([1.0], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        return pts, scale, R
    return kpconv_vertical_augment(points, config) 