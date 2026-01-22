"""
Farthest Point Sampling (FPS) utilities
GPU-accelerated and NumPy implementations for point cloud sampling
"""

import numpy as np
import torch


def farthest_point_sample_numpy(points, n_samples):
    """
    Farthest Point Sampling using NumPy (for preprocessing)
    
    Args:
        points: numpy array of shape [N, 3] - point cloud coordinates
        n_samples: int - number of points to sample
    
    Returns:
        indices: numpy array of shape [n_samples] - indices of selected points
    """
    N = points.shape[0]
    if N <= n_samples:
        # If we have fewer points than requested, return all indices
        indices = np.arange(N)
        # Pad with repeated indices if necessary
        if N < n_samples:
            indices = np.concatenate([indices, np.random.choice(N, n_samples - N)])
        return indices
    
    # Initialize
    centroids = np.zeros(n_samples, dtype=np.int32)
    distance = np.full(N, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, N)
    
    for i in range(n_samples):
        # Save current farthest point
        centroids[i] = farthest
        
        # Calculate distances from current centroid to all points
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        
        # Update minimum distances
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Find the farthest point from all selected centroids
        farthest = np.argmax(distance)
    
    return centroids


def farthest_point_sample_torch(points, n_samples, device=None):
    """
    GPU-accelerated Farthest Point Sampling using PyTorch
    
    Args:
        points: torch.Tensor or numpy array of shape [N, 3] - point cloud coordinates
        n_samples: int - number of points to sample
        device: torch.device - GPU device (auto-detect if None)
    
    Returns:
        indices: torch.Tensor of shape [n_samples] - indices of selected points
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensor if needed
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points.astype(np.float32))
    
    # Move to device
    points = points.to(device)
    N = points.shape[0]
    
    if N <= n_samples:
        # If we have fewer points than requested, return all indices
        indices = torch.arange(N, device=device)
        # Pad with repeated indices if necessary
        if N < n_samples:
            additional_indices = torch.randint(0, N, (n_samples - N,), device=device)
            indices = torch.cat([indices, additional_indices])
        return indices.cpu()
    
    # Initialize
    centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
    distance = torch.full((N,), 1e10, dtype=torch.float32, device=device)
    farthest = torch.randint(0, N, (1,), device=device).item()
    
    for i in range(n_samples):
        # Save current farthest point
        centroids[i] = farthest
        
        # Calculate distances from current centroid to all points
        centroid = points[farthest:farthest+1, :]  # [1, 3]
        dist = torch.sum((points - centroid) ** 2, dim=1)  # [N]
        
        # Update minimum distances
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Find the farthest point from all selected centroids
        farthest = torch.argmax(distance).item()
    
    return centroids.cpu()


def farthest_point_sample_batch_torch(points, n_samples, batch_size, device=None):
    """
    Batched GPU-accelerated Farthest Point Sampling using PyTorch
    
    Args:
        points: torch.Tensor or numpy array of shape [N, 3] - point cloud coordinates
        n_samples: int - number of points to sample per batch item
        batch_size: int - number of FPS samples to create
        device: torch.device - GPU device (auto-detect if None)
    
    Returns:
        indices_batch: torch.Tensor of shape [batch_size, n_samples] - indices for each sample
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensor if needed
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points.astype(np.float32))
    
    # Move to device
    points = points.to(device)
    N = points.shape[0]
    
    if N <= n_samples:
        # Simple case: upsample for all batch items
        indices = torch.arange(N, device=device)
        if N < n_samples:
            additional_indices = torch.randint(0, N, (n_samples - N,), device=device)
            indices = torch.cat([indices, additional_indices])
        # Repeat for batch
        return indices.unsqueeze(0).repeat(batch_size, 1).cpu()
    
    # Batch processing - each with different random starting point
    batch_indices = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Initialize for this batch item with different random seed
        centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
        distance = torch.full((N,), 1e10, dtype=torch.float32, device=device)
        
        # Use different starting points for diversity
        torch.manual_seed(42 + b * 1000)
        farthest = torch.randint(0, N, (1,), device=device).item()
        
        for i in range(n_samples):
            centroids[i] = farthest
            centroid = points[farthest:farthest+1, :]  # [1, 3]
            dist = torch.sum((points - centroid) ** 2, dim=1)  # [N]
            
            # Update minimum distances
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Find the farthest point from all selected centroids
            farthest = torch.argmax(distance).item()
        
        batch_indices[b] = centroids
    
    return batch_indices.cpu()




