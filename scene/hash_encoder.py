import torch
import torch.nn as nn
import numpy as np

class HashEncoder4D(nn.Module):
    """
    Multi-resolution 4D hash encoding for spatial-temporal fields
    
    This encoder maps 4D coordinates (x,y,z,t) to a feature vector through a series
    of hash tables at different resolutions.
    """
    def __init__(
        self,
        bounds,                  # Scene bounds for normalization
        n_levels=16,             # Number of resolution levels
        min_resolution=16,       # Base resolution (coarsest level)
        max_resolution=512,      # Maximum resolution (finest level)
        log2_hashmap_size=15,    # Log2 of hash table size (default: 2^15 = 32768 entries)
        feature_dim=2,           # Feature dimension per level
        interpolation='linear',  # Interpolation mode: 'linear' or 'nearest'
        concat_features=True,    # Whether to concatenate features from all levels
    ):
        super(HashEncoder4D, self).__init__()
        
        # Store bounds for normalization
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        # Configuration
        self.n_levels = n_levels
        self.feature_dim = feature_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.concat_features = concat_features
        self.interpolation = interpolation
        
        # Calculate level resolutions using geometric progression
        self.level_resolutions = []
        if n_levels > 1:
            for level in range(n_levels):
                # Calculate resolution for this level
                factor = level / (n_levels - 1)
                resolution = int(np.exp(np.log(min_resolution) * (1 - factor) + np.log(max_resolution) * factor))
                self.level_resolutions.append(resolution)
        else:
            self.level_resolutions = [min_resolution]
        
        # Initialize hash tables for each level
        self.hash_tables = nn.ModuleList()
        for level in range(n_levels):
            # Each hash table entry contains a feature vector
            hash_table = nn.Embedding(self.hashmap_size, feature_dim)
            # Initialize with small random values
            nn.init.uniform_(hash_table.weight, -1e-4, 1e-4)
            self.hash_tables.append(hash_table)
        
        # Calculate output dimension
        if concat_features:
            self.out_dim = feature_dim * n_levels
        else:
            self.out_dim = feature_dim
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Hash Encoder: set aabb =", self.aabb)
    
    def normalize_coords(self, coords, timestamps=None):
        """Normalize coordinates to [0, 1] range"""
        # Normalize spatial coordinates based on scene bounds
        normalized_coords = (coords - self.aabb[1]) / (self.aabb[0] - self.aabb[1])
        
        # If timestamps are provided, concatenate them with normalized coords
        if timestamps is not None:
            # Ensure timestamps are in [0, 1] range
            normalized_timestamps = torch.clamp(timestamps, 0, 1)
            normalized_coords = torch.cat([normalized_coords, normalized_timestamps], dim=-1)
        
        return normalized_coords
    
    def hash_function(self, coords, resolution):
        """
        Spatial hash function for 4D coordinates
        Args:
            coords: tensor of shape [..., 4] containing normalized coordinates
            resolution: integer resolution for this level's grid
        Returns:
            hash_indices: tensor of indices into the hash table
        """
        # Scale to the resolution of this level
        scaled_coords = coords * resolution
        
        # Get integer coordinates (voxel indices)
        grid_indices = scaled_coords.floor().long()
        
        # Get the 16 corners of the 4D hypercube
        corners = []
        for i in range(16):
            # Convert i to 4-bit binary and use as offsets
            offset = torch.zeros_like(grid_indices)
            for d in range(4):
                if i & (1 << d):
                    offset[..., d] = 1
            corner_indices = grid_indices + offset
            corners.append(corner_indices)
        
        # Apply hash function to each corner
        hashed_corners = []
        for corner in corners:
            # Use prime numbers for hashing to reduce collisions
            # Based on common spatial hashing techniques
            h = corner[..., 0] * 1
            h = h ^ (corner[..., 1] * 2654435761)
            h = h ^ (corner[..., 2] * 805459861)
            h = h ^ (corner[..., 3] * 3674653429)
            
            # Modulo hash table size
            h = h % self.hashmap_size
            hashed_corners.append(h)
        
        return hashed_corners, scaled_coords - grid_indices
    
    def trilinear_interp(self, features, weights):
        """
        Perform trilinear interpolation on features using weights
        Args:
            features: list of 16 feature tensors for hypercube corners
            weights: tensor of shape [..., 4] with interpolation weights
        """
        result = 0
        
        # For each corner of the 4D hypercube
        for i in range(16):
            # Calculate the weight for this corner
            w = 1.0
            for d in range(4):
                if i & (1 << d):
                    w = w * weights[..., d]
                else:
                    w = w * (1 - weights[..., d])
            
            # Apply weight to features
            result = result + features[i] * w.unsqueeze(-1)
        
        return result
    
    def forward(self, coords, timestamps=None):
        """
        Encode 4D coordinates to features
        Args:
            coords: tensor of shape [..., 3] containing xyz coordinates
            timestamps: tensor of shape [..., 1] containing time coordinates
        Returns:
            features: tensor of encoded features
        """
        # Save original shape for reshaping output
        original_shape = coords.shape[:-1]
        
        # Flatten input
        coords_flat = coords.reshape(-1, 3)
        if timestamps is not None:
            timestamps_flat = timestamps.reshape(-1, 1)
        else:
            timestamps_flat = None
        
        # Normalize coordinates to [0, 1]
        normalized_coords = self.normalize_coords(coords_flat, timestamps_flat)
        
        # Process each resolution level
        level_features = []
        for level, resolution in enumerate(self.level_resolutions):
            # Get hash indices and interpolation weights for this level
            hash_indices, interp_weights = self.hash_function(normalized_coords, resolution)
            
            # Look up features from hash table
            corner_features = [self.hash_tables[level](idx) for idx in hash_indices]
            
            # Interpolate features
            if self.interpolation == 'linear':
                interpolated = self.trilinear_interp(corner_features, interp_weights)
            else:  # nearest
                interpolated = corner_features[0]  # Just use the first corner
            
            level_features.append(interpolated)
        
        # Combine features from all levels
        if self.concat_features:
            features = torch.cat(level_features, dim=-1)
        else:
            features = torch.mean(torch.stack(level_features), dim=0)
        
        # Reshape to original dimensions
        features = features.reshape(*original_shape, -1)
        
        return features 