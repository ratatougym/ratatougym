import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time  # Import the time module


class Masking:
    """
    Base class for all masking utilities.
    """
    def __init__(
            self, 
            m_max: float = 0.5, 
            m_min: float = 0.0, 
            sigma_t: float = 0, 
            sigma_d: float = 0,
            t_warmup: int = 0,
            device: torch.device = torch.device('cpu')
        ):
        self.m_max = m_max
        self.m_min = m_min if m_min is not None else m_max
        self.sigma_t = sigma_t  # Temporal smoothing
        self.sigma_d = sigma_d  # Spatial smoothing
        self.t_warmup = t_warmup  # Warmup time steps
        self.device = device
        assert self.m_min <= self.m_max, "m_min must be <= m_max"

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

    def __call__(self, x, mask_idx=None):
        return self.mask(x, mask_idx)

    def to(self, device: torch.device):
        self.device = device
        return self

    def set_m_max(self, m_max: float):
        self.m_max = m_max
        return self

    def set_m_min(self, m_min: float):
        self.m_min = m_min
        return self

    def gaussian_kernel_1d(self, kernel_size: int, sigma: float):
        """Creates a 1D Gaussian kernel."""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x.float()**2 / (2 * sigma**2))
        gauss /= gauss.sum()
        return gauss

    def apply_gaussian_blur(self, mask: torch.Tensor):
        # Temporal blur
        if self.sigma_t > 0:
            k_size_t = max(int(6 * self.sigma_t) | 1, 3)
            kernel_t = self.gaussian_kernel_1d(k_size_t, self.sigma_t).to(mask.device)
            kernel_t = kernel_t.view(1, 1, -1, 1)
            pad_t = k_size_t // 2
            m = mask.unsqueeze(1)
            m = F.pad(m, (0,0, pad_t,pad_t), mode='reflect')
            mask = F.conv2d(m, kernel_t, padding=0).squeeze(1)

        # Spatial blur
        if self.sigma_d > 0:
            k_size_d = max(int(6 * self.sigma_d) | 1, 3)
            kernel_d = self.gaussian_kernel_1d(k_size_d, self.sigma_d).to(mask.device)
            kernel_d = kernel_d.view(1, 1, 1, -1)
            pad_d = k_size_d // 2
            m = mask.unsqueeze(1)
            m = F.pad(m, (pad_d,pad_d, 0,0), mode='reflect')
            mask = F.conv2d(m, kernel_d, padding=0).squeeze(1)

        return mask


    def new_mask(self, x):
        """
        Generates a mask with different masking thresholds for each dimension.
        The thresholds are determined based on per-batch and per-dimension masking ratios.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, T, D)
        
        Returns:
            torch.Tensor: Mask tensor of the same shape as x
        """
        # Generate a random mask
        if self.m_max == 0:
            return torch.ones_like(x, device=self.device)
        mask = torch.rand(x.shape, device=self.device)
        mask = self.apply_gaussian_blur(mask)  # Shape: (batch_size, T, D)        

        # Sort the mask along the time dimension
        sorted_mask, _ = mask.sort(dim=1)  # Shape: (batch_size, T, D)

        # Generate masking ratios for each dimension
        masking_ratios = torch.empty(x.shape[0], 1, device=mask.device).uniform_(self.m_min, self.m_max)  # Per-batch ratio
        sorted_mask, _ = mask.view(x.shape[0], -1).sort(dim=1)
        max_size = sorted_mask.size(1) - 1
        q_indices = (masking_ratios * (max_size)).long().clamp(max=max_size)
        thresholds = sorted_mask.gather(dim=1, index=q_indices).unsqueeze(1)
        mask = mask > thresholds  # Apply thresholds to create final mask

        if self.t_warmup > 0:
            mask[:, :self.t_warmup, :] = 1  # Unmask the first time step

        return mask.float()

    def mask(self, x, mask_idx=None):
        x = self.to_tensor(x)  # Ensure tensor
        mask = self.new_mask(x)  # Generate mask
        if mask_idx is None:
            return x * mask  # Apply mask
        else:
            mask[~mask_idx] = 1.0
            return x * mask  # Apply mask
