"""
Utility functions for decoding neuronal responses to spatial trajectories.

This module contains methods for decoding high-dimensional neuronal responses
(e.g., from place cells, grid cells) back to spatial coordinates using various 
techniques including nearest neighbor search, spatial interpolation,
and approximate search methods.

Given neuronal response vectors :math:`\\mathbf{r} \\in \\mathbb{R}^d` and response maps
:math:`\\mathbf{M} \\in \\mathbb{R}^{H \\times W \\times d}`, we seek to find coordinates
:math:`(y,x)` such that:

.. math::

    (y,x) = \\arg\\min_{(i,j)} \\|\\mathbf{r} - \\mathbf{M}[i,j,:]\\|_2^2

where :math:`\\|\\cdot\\|_2` denotes the Euclidean norm.
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple, Any
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import faiss

from rtgym.dataclass import AgentState, Trajectory


def decode_response_euclidean(
    response: np.ndarray,
    res_maps: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Decode neuronal responses using brute-force Euclidean distance computation.

    This is the reference implementation that computes the exact nearest neighbor
    by evaluating the Euclidean distance between each query response and all
    template responses in the response maps.

    For response vector :math:`\mathbf{r} \in \mathbb{R}^d` and response map
    :math:`\mathbf{M} \in \mathbb{R}^{H \times W \times d}`, compute:

    .. math::

        d^2(\mathbf{r}, \mathbf{M}[i,j]) = \sum_{k=1}^{d} (r_k - M[i,j,k])^2

    Return:

    .. math::

        \arg\min_{(i,j)} d^2(\mathbf{r}, \mathbf{M}[i,j])

    Args:
        response (np.ndarray): Sensory response array of shape:
            - (B, T, D) for trajectory decoding
            - (B, D) for single state decoding
            where B = batch size, T = time steps, D = feature dimensions
        res_maps (np.ndarray): Response template maps of shape (D, H, W)
            where H = arena height, W = arena width

    Returns:
        tuple: (decoded_coordinates, is_trajectory) where:
            - decoded_coordinates (np.ndarray): Array of shape (B,T,2) or (B,2)
              containing spatial coordinates
            - is_trajectory (bool): True if input was trajectory (3D), False if state (2D)

    Note:
        This method provides exact results but can be computationally expensive
        for large arenas or high-dimensional feature spaces. Consider using
        approximate methods for better performance.

        Time complexity: :math:`O(B \cdot T \cdot H \cdot W \cdot D)` for trajectory, :math:`O(B \cdot H \cdot W \cdot D)` for states
        Space complexity: :math:`O(H \cdot W \cdot D)`
    """
    n_cells, H, W = res_maps.shape

    # Reshape response maps: (D, H, W) → (H*W, D)
    M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, D)

    # Handle NaN values by replacing with infinity (won't be selected as minimum)
    M = np.nan_to_num(M, nan=np.inf)

    # Generate coordinate grid: each position (i,j) maps to linear index i*W + j
    coords = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
    ).reshape(-1, 2)  # shape: (H*W, 2)

    # Determine input format and flatten appropriately
    is_trajectory = len(response.shape) == 3
    if is_trajectory:
        B, T, _ = response.shape
        r_flat = response.reshape(-1, n_cells)  # shape: (B*T, D)
    else:
        B = response.shape[0]
        r_flat = response  # shape: (B, D)

    # Compute squared Euclidean distances: ||r - M||²
    # Broadcasting: (B*T, 1, D) - (1, H*W, D) → (B*T, H*W, D)
    dists = np.sum((r_flat[:, None, :] - M[None, :, :]) ** 2, axis=-1)

    # Find coordinates with minimum distance
    min_indices = np.argmin(dists, axis=-1)
    pred_coords = coords[min_indices]

    # Reshape output based on input format
    if is_trajectory:
        pred_coords = pred_coords.reshape(B, T, 2)

    return pred_coords, is_trajectory


def decode_response_kdtree(
    response: np.ndarray,
    res_maps: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Decode sensory responses using K-d tree for accelerated nearest neighbor search.

    K-d trees provide significant speedup over brute force methods, especially
    for high-dimensional feature spaces. The tree is constructed once and then
    queried efficiently for all response vectors.

    Algorithm:
        1. Build K-d tree on flattened response map features
        2. Query tree for nearest neighbors of each response vector
        3. Map indices back to spatial coordinates

    Args:
        response (np.ndarray): Sensory response array (see decode_response_euclidean)
        res_maps (np.ndarray): Response template maps (see decode_response_euclidean)

    Returns:
        tuple: Same format as decode_response_euclidean

    Note:
        Complexity:
        - Tree construction: :math:`O(H \cdot W \cdot D \cdot \log(H \cdot W))`
        - Query: :math:`O(B \cdot T \cdot \log(H \cdot W))` for trajectory
        - Space: :math:`O(H \cdot W \cdot D)`

        Advantages:
        - Significantly faster than brute force for large arenas
        - Exact nearest neighbor results
        - Memory efficient tree structure

        Limitations:
        - Performance degrades in very high dimensions (curse of dimensionality)
        - Tree construction overhead for small datasets
    """
    n_cells, H, W = res_maps.shape

    # Prepare response maps and coordinates (same as euclidean method)
    M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, D)
    M = np.nan_to_num(M, nan=np.inf)

    coords = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
    ).reshape(-1, 2)

    # Process input format
    is_trajectory = len(response.shape) == 3
    if is_trajectory:
        B, T, _ = response.shape
        r_flat = response.reshape(-1, n_cells)
    else:
        B = response.shape[0]
        r_flat = response

    # Build K-d tree on feature space
    # This is the computational bottleneck but only done once
    kdtree = KDTree(M)

    # Query tree for nearest neighbors
    # Returns (distances, indices) but we only need indices
    _, min_indices = kdtree.query(r_flat, k=1)
    min_indices = min_indices.flatten()

    # Map indices to spatial coordinates
    pred_coords = coords[min_indices]

    if is_trajectory:
        pred_coords = pred_coords.reshape(B, T, 2)

    return pred_coords, is_trajectory


def decode_response_torch(
    response: Union[np.ndarray, torch.Tensor],
    res_maps: Union[np.ndarray, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    chunk_size: int = 1024
) -> Tuple[np.ndarray, bool]:
    """Decode sensory responses using PyTorch for GPU-accelerated computation.

    This implementation leverages PyTorch's optimized tensor operations and
    optional GPU acceleration for faster distance computation. Memory usage
    is controlled through chunked processing.

    Instead of computing :math:`\|\mathbf{r} - \mathbf{M}\|^2` directly, we use the identity:

    .. math::

        \|\mathbf{r} - \mathbf{M}\|^2 = \|\mathbf{r}\|^2 - 2\langle\mathbf{r},\mathbf{M}\rangle + \|\mathbf{M}\|^2

    Since :math:`\|\mathbf{r}\|^2` is constant for :math:`\arg\min`, we compute:

    .. math::

        \text{distance} \propto -2\langle\mathbf{r},\mathbf{M}\rangle + \|\mathbf{M}\|^2

    Args:
        response (Union[np.ndarray, torch.Tensor]): Sensory response array
            (numpy or torch tensor)
        res_maps (Union[np.ndarray, torch.Tensor]): Response template maps
            (numpy or torch tensor)
        device (Optional[Union[str, torch.device]]): Computation device
            ('cpu', 'cuda', or torch.device object). If None, automatically
            selects CUDA if available
        chunk_size (int): Number of responses to process simultaneously.
            Larger values use more memory but may be faster. Defaults to 1024.

    Returns:
        tuple: (decoded_coordinates, is_trajectory) as numpy arrays

    Note:
        Performance:
        - GPU acceleration provides 10-100x speedup for large problems
        - Chunked processing prevents out-of-memory errors
        - Automatic mixed precision could be added for further optimization

        Memory Usage:
        Peak memory ≈ ``chunk_size × H × W × 4`` bytes (for float32)
    """
    # Device selection and setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Convert inputs to torch tensors if needed
    if isinstance(res_maps, np.ndarray):
        res_maps = torch.from_numpy(res_maps).to(device)
    if isinstance(response, np.ndarray):
        response = torch.from_numpy(response).to(device)

    # Ensure tensors are on correct device
    res_maps = res_maps.to(device)
    response = response.to(device)

    n_cells, H, W = res_maps.shape

    # Create coordinate grid on device
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack([y_coords, x_coords], dim=-1).reshape(-1, 2)

    # Reshape response maps: (D, H, W) → (H*W, D)
    M = res_maps.reshape(n_cells, -1).t()  # PyTorch transpose
    M = torch.nan_to_num(M, nan=float('inf'))

    # Process input format
    is_trajectory = len(response.shape) == 3
    if is_trajectory:
        B, T, _ = response.shape
        r_flat = response.reshape(-1, n_cells)
    else:
        B = response.shape[0]
        r_flat = response

    # Chunked computation to manage memory usage
    min_indices = []

    for i in range(0, r_flat.size(0), chunk_size):
        chunk = r_flat[i:i+chunk_size]

        # Optimized distance computation using matrix operations
        # ||r - M||² = ||r||² - 2⟨r,M⟩ + ||M||²
        r_norm = (chunk**2).sum(dim=1, keepdim=True)  # ||r||²
        m_norm = (M**2).sum(dim=1)                     # ||M||²

        # Efficient matrix multiplication for dot products
        dot_products = torch.mm(chunk, M.t())  # ⟨r,M⟩

        # Compute distances (||r||² term cancels in argmin)
        distances = r_norm - 2*dot_products + m_norm

        # Find minimum distance indices for this chunk
        chunk_min_indices = torch.argmin(distances, dim=1)
        min_indices.append(chunk_min_indices)

    # Concatenate results from all chunks
    min_indices = torch.cat(min_indices)
    pred_coords = coords[min_indices]

    # Convert back to numpy and reshape
    pred_coords_np = pred_coords.cpu().numpy()

    if is_trajectory:
        pred_coords_np = pred_coords_np.reshape(B, T, 2)

    return pred_coords_np, is_trajectory


def decode_response_faiss(
    response: np.ndarray,
    res_maps: np.ndarray,
    n_clusters: int = 100
) -> Tuple[np.ndarray, bool]:
    """Decode sensory responses using Facebook AI Similarity Search (FAISS).

    FAISS provides state-of-the-art performance for large-scale nearest neighbor
    search through optimized indexing structures. This implementation uses
    Inverted File Index (IVF) with exact search within clusters.

    Algorithm:
        1. Cluster response map features into n_clusters groups using k-means
        2. Build inverted index mapping clusters to their members
        3. For each query, search only relevant clusters for nearest neighbors
        4. Return exact nearest neighbor within searched clusters

    Args:
        response (np.ndarray): Sensory response array
        res_maps (np.ndarray): Response template maps
        n_clusters (int): Number of clusters for IVF index. Defaults to 100.
            - More clusters: faster search, potentially lower recall
            - Fewer clusters: slower search, higher recall
            - Recommended: :math:`\sqrt{\text{number of points}}` to :math:`\text{n_points}/39`

    Returns:
        tuple: (decoded_coordinates, is_trajectory)

    Note:
        Performance characteristics:
        - Index build time: :math:`O(H \cdot W \cdot D \cdot \log(\text{n_clusters}))`
        - Query time: :math:`O(B \cdot T \cdot \sqrt{H \cdot W})` approximately
        - Memory usage: ~1.5x the size of original data

        Requirements:
        - Requires FAISS library installation
        - Optimized for very large datasets (>10k points)
        - Can utilize GPU acceleration with appropriate FAISS build
        - May return approximate results depending on ``nprobe`` parameter
    """
    n_cells, H, W = res_maps.shape

    # Prepare data (same preprocessing as other methods)
    M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, D)
    M = np.nan_to_num(M, nan=0.0)  # FAISS doesn't handle inf well
    M = M.astype(np.float32)  # FAISS requires float32

    coords = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
    ).reshape(-1, 2)

    # Process input format
    is_trajectory = len(response.shape) == 3
    if is_trajectory:
        B, T, _ = response.shape
        r_flat = response.reshape(-1, n_cells)
    else:
        B = response.shape[0]
        r_flat = response

    r_flat = r_flat.astype(np.float32)

    # Build FAISS index
    d = n_cells  # Feature dimensionality

    # Base quantizer for exact L2 distance computation
    quantizer = faiss.IndexFlatL2(d)

    # Adjust cluster count based on data constraints
    # FAISS requires at least 39 training vectors per cluster
    n_clusters = min(n_clusters, M.shape[0] // 39)
    n_clusters = max(1, n_clusters)

    # Create Inverted File Index with Flat (exact) quantizer
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)

    # Train index on the response map data
    index.train(M)

    # Add all response map vectors to index
    index.add(M)

    # Set search parameters
    # nprobe: number of clusters to search (higher = more accurate, slower)
    index.nprobe = min(10, n_clusters)

    # Perform nearest neighbor search
    _, indices = index.search(r_flat, 1)  # k=1 for single nearest neighbor

    # Map indices back to coordinates
    min_indices = indices.flatten()
    pred_coords = coords[min_indices]

    if is_trajectory:
        pred_coords = pred_coords.reshape(B, T, 2)

    return pred_coords, is_trajectory


def decode_response_interpolation(
    response: np.ndarray,
    res_maps: np.ndarray,
    n_anchors: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, bool]:
    """Decode sensory responses using spatial interpolation with anchor points.

    This method exploits the spatial continuity assumption: nearby locations
    should have similar sensory responses. Instead of exact nearest neighbor
    search, it interpolates coordinates from multiple nearby anchor points.

    Algorithm:
        1. Select n_anchors representative points using k-means clustering
        2. For each query response, find k nearest anchor points
        3. Compute inverse-distance weighted interpolation of anchor coordinates
        4. Return interpolated coordinates (may be non-integer)

    Given :math:`k` nearest anchors with coordinates :math:`\mathbf{c}_1,\ldots,\mathbf{c}_k`
    and distances :math:`d_1,\ldots,d_k`:

    .. math::

        w_i &= \frac{1}{d_i + \varepsilon} \quad \text{where } \varepsilon \text{ prevents division by zero}

        \tilde{w}_i &= \frac{w_i}{\sum_{j=1}^k w_j} \quad \text{(normalized weights)}

        \hat{\mathbf{c}} &= \sum_{i=1}^k \tilde{w}_i \cdot \mathbf{c}_i \quad \text{(interpolated coordinate)}

    Args:
        response (np.ndarray): Sensory response array
        res_maps (np.ndarray): Response template maps
        n_anchors (int): Number of anchor points to select. Defaults to 1000.
            - More anchors: better spatial resolution, slower computation
            - Fewer anchors: faster computation, lower spatial precision
        random_state (int): Random seed for reproducible anchor selection.
            Defaults to 42.

    Returns:
        tuple: (decoded_coordinates, is_trajectory)
            Coordinates are rounded to integers and clipped to valid arena bounds

    Note:
        Advantages:
        - Smooth spatial interpolation reduces noise
        - Faster than exact nearest neighbor for large datasets
        - Naturally handles uncertainty through weighted averaging

        Limitations:
        - May not find exact nearest neighbor
        - Requires tuning of n_anchors parameter
        - Assumes local spatial smoothness in response patterns

        Complexity:
        - Anchor selection: :math:`O(H \cdot W \cdot D \cdot \log(\text{n_anchors}))`
        - Query: :math:`O(B \cdot T \cdot \log(\text{n_anchors}))`
    """
    n_cells, H, W = res_maps.shape

    # Standard preprocessing
    M = res_maps.reshape(n_cells, -1).T
    M = np.nan_to_num(M, nan=0.0)

    coords = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
    ).reshape(-1, 2)

    # Process input format
    is_trajectory = len(response.shape) == 3
    if is_trajectory:
        B, T, _ = response.shape
        r_flat = response.reshape(-1, n_cells)
    else:
        B = response.shape[0]
        r_flat = response

    # Limit anchors to available data
    n_anchors = min(n_anchors, M.shape[0])

    # Select anchor points using k-means clustering
    # This finds representative points in feature space
    kmeans = KMeans(n_clusters=n_anchors, random_state=random_state, n_init=10)
    kmeans.fit(M)

    anchor_indices = []

    # For each cluster, find the actual data point closest to centroid
    for i in range(n_anchors):
        cluster_mask = kmeans.labels_ == i
        cluster_points = M[cluster_mask]

        if len(cluster_points) > 0:
            centroid = kmeans.cluster_centers_[i]
            distances_to_centroid = np.sum((cluster_points - centroid) ** 2, axis=1)
            closest_in_cluster = np.argmin(distances_to_centroid)

            # Map back to original index
            original_indices = np.where(cluster_mask)[0]
            anchor_indices.append(original_indices[closest_in_cluster])

    # Extract anchor features and coordinates
    anchor_features = M[anchor_indices]
    anchor_coords = coords[anchor_indices]

    # Build efficient search structure for anchors
    anchor_tree = KDTree(anchor_features)

    # Find k nearest anchors for interpolation
    k = min(5, len(anchor_indices))  # Use up to 5 neighbors
    distances, neighbor_indices = anchor_tree.query(r_flat, k=k)

    # Compute inverse distance weights
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    weights = 1.0 / (distances + epsilon)

    # Normalize weights so they sum to 1
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / weight_sums

    # Perform weighted interpolation of coordinates
    pred_coords = np.zeros((r_flat.shape[0], 2))
    for i in range(k):
        neighbor_coords = anchor_coords[neighbor_indices[:, i]]
        pred_coords += weights[:, i:i+1] * neighbor_coords

    # Round to integer coordinates and clip to valid bounds
    pred_coords = np.round(pred_coords).astype(int)
    pred_coords[:, 0] = np.clip(pred_coords[:, 0], 0, H-1)
    pred_coords[:, 1] = np.clip(pred_coords[:, 1], 0, W-1)

    if is_trajectory:
        pred_coords = pred_coords.reshape(B, T, 2)

    return pred_coords, is_trajectory


def create_dataclass_result(
    pred_coords: np.ndarray,
    is_trajectory: bool
) -> Union[Trajectory, AgentState]:
    """Create appropriate dataclass result from decoded coordinates.

    Args:
        pred_coords (np.ndarray): Decoded coordinate array
        is_trajectory (bool): Whether to create Trajectory or AgentState object

    Returns:
        Union[Trajectory, AgentState]: Dataclass object containing the coordinates
    """
    if is_trajectory:
        return Trajectory(coord=pred_coords)
    else:
        return AgentState(coord=pred_coords)