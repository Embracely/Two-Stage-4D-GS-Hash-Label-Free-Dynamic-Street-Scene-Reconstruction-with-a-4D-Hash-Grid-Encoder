import functools
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.hash_encoder import HashEncoder4D
from scene.grid import DenseGrid

class DeformationHash(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(DeformationHash, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.args = args
        
        # Initialize the encoder based on the encoder_type parameter
        if hasattr(args, 'encoder_type') and args.encoder_type == 'hash':
            print("Using HashEncoder4D for deformation field")
            # Parse hash encoder specific parameters
            hash_params = getattr(args, 'hash_config', {})
            n_levels = hash_params.get('n_levels', 16)
            min_resolution = hash_params.get('min_resolution', 16)
            max_resolution = hash_params.get('max_resolution', 512)
            log2_hashmap_size = hash_params.get('log2_hashmap_size', 15)
            feature_dim = hash_params.get('feature_dim', 2)
            
            self.grid = HashEncoder4D(
                bounds=args.bounds,
                n_levels=n_levels,
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                log2_hashmap_size=log2_hashmap_size,
                feature_dim=feature_dim,
                concat_features=True
            )
            self.feat_dim = self.grid.out_dim
        else:
            print("Using HexPlaneField for deformation field")
            self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
            self.feat_dim = self.grid.feat_dim
        
        # Initialize empty voxel grid if specified
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        
        # Initialize static MLP if specified
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        
        self.ratio = 0
        self.create_net()
    
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb", xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe != 0:
            grid_out_dim = self.feat_dim + (self.feat_dim) * 2
        else:
            grid_out_dim = self.feat_dim
        
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        # Define deformation networks for different attributes
        self.pos_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16*3))
        
        # Initialize feature head if specified
        if self.args.feat_head:
            semantic_feature_dim = self.W
            feature_mlp_layer_width = 64
            feature_embedding_dim = 3
            self.dino_head = nn.Sequential(
                nn.Linear(semantic_feature_dim, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(feature_mlp_layer_width, feature_embedding_dim),
            )

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3], time_emb[:,:1]], -1)
        else:
            # Get features from the encoder (either HexPlane or HashEncoder4D)
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            
            # Apply positional encoding if specified
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            
            hidden = torch.cat([grid_feature], -1)
        
        # Process features through MLP
        hidden = self.feature_out(hidden)   # [N,64]
        
        return hidden
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        # Get features from the encoder with time information
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        
        # Determine static/dynamic mask
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)  # [N, 1]
        
        # Position deformation
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
            dx = None
        else:
            dx = self.pos_deform(hidden)  # [N, 3]
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        
        # Scale deformation
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
        
        # Rotation deformation
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr
        
        # Opacity deformation
        if self.args.no_do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden)
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        
        # SH coefficients deformation
        if self.args.no_dshs:
            shs = shs_emb
            dshs = None
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs
        
        # Feature extraction if specified
        feat = None
        if self.args.feat_head:
            feat = self.dino_head(hidden)
        
        return pts, scales, rotations, opacity, shs, dx, feat, dshs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list

class deform_network_hash(nn.Module):
    def __init__(self, args):
        super(deform_network_hash, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        
        # Time embedding network
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )
        
        # Deformation network with hash encoder option
        self.deformation_net = DeformationHash(
            W=net_width, 
            D=defor_depth, 
            input_ch=(3)+(3*(posbase_pe))*2, 
            grid_pe=grid_pe, 
            input_ch_time=timenet_output, 
            args=args
        )
        
        # Register buffers for positional encoding
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        
        # Initialize weights
        self.apply(initialize_weights)
    
    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # Process time embeddings
        times_emb = poc_fre(times_sel, self.time_poc)
        time_feature = self.timenet(times_emb)
        
        # Forward through deformation network
        return self.deformation_net(point, scales, rotations, opacity, shs, time_feature, times_sel)
    
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
    
    def forward_static(self, points):
        return self.deformation_net.forward_static(points)
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # Process time embeddings
        times_emb = poc_fre(times_sel, self.time_poc)
        time_feature = self.timenet(times_emb)
        
        # Forward through deformation network
        return self.deformation_net(point, scales, rotations, opacity, shs, time_feature, times_sel)
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def poc_fre(input_data, poc_buf):
    """Positional encoding function"""
    input_data = input_data.unsqueeze(-1)  # [..., 1, 1]
    code = input_data * poc_buf  # [..., 1, C]
    code = torch.cat([torch.sin(code), torch.cos(code)], dim=-1)  # [..., 1, 2C]
    code = code.flatten(-2, -1)  # [..., 2C]
    return torch.cat([input_data.squeeze(-1), code], dim=-1)  # [..., 1+2C] 