import os
import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid, get_expon_lr_func
from utils.graphics_utils import compute_plane_smoothness
from scene.deformation import deform_network
from scene.deformation_hash import deform_network_hash

class GaussianModelHash:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        
        # Initialize deformation network based on encoder_type
        if hasattr(args, 'encoder_type') and args.encoder_type == 'hash':
            print("Using Hash Encoder for deformation network")
            self._deformation = deform_network_hash(args)
        else:
            print("Using HexPlane for deformation network")
            self._deformation = deform_network(args)
            
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def compute_deformation(self, time):
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz
        
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "deformation.pth"), map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)
        
    def standard_constaint(self):
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time = torch.tensor(0).to("cuda").repeat(means3D.shape[0], 1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    @torch.no_grad()
    def update_deformation_table(self, threshold):
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100, threshold)
        
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    print(name, " :", weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name, " :", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
        
    def _plane_regulation(self):
        # Check if using hash encoder or hexplane
        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):
            # HexPlane regulation
            multi_res_grids = self._deformation.deformation_net.grid.grids
            total = 0
            for grids in multi_res_grids:
                if len(grids) == 3:
                    time_grids = []
                else:
                    time_grids = [0, 1, 3]
                for grid_id in time_grids:
                    total += compute_plane_smoothness(grids[grid_id])
            return total
        else:
            # Hash encoder regulation - apply L2 regularization on hash table parameters
            total = 0
            if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'hash_tables'):
                hash_tables = self._deformation.deformation_net.grid.hash_tables
                for table in hash_tables:
                    total += torch.mean(torch.square(table.weight))
            return total * 0.01  # Scale factor to match hexplane regulation magnitude
            
    def _time_regulation(self):
        # Check if using hash encoder or hexplane
        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):
            # HexPlane regulation
            multi_res_grids = self._deformation.deformation_net.grid.grids
            total = 0
            for grids in multi_res_grids:
                if len(grids) == 3:
                    time_grids = []
                else:
                    time_grids = [2, 4, 5]
                for grid_id in time_grids:
                    total += compute_plane_smoothness(grids[grid_id])
            return total
        else:
            # For hash encoder, apply temporal smoothness by L2 regularization on consecutive time steps
            # This is a simplified approach since hash tables don't have explicit time planes
            return self._plane_regulation() * 0.5  # Use a fraction of spatial regularization
            
    def _l1_regulation(self):
        # Check if using hash encoder or hexplane
        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):
            # HexPlane regulation
            multi_res_grids = self._deformation.deformation_net.grid.grids
            total = 0.0
            for grids in multi_res_grids:
                if len(grids) == 3:
                    continue
                else:
                    spatiotemporal_grids = [2, 4, 5]
                for grid_id in spatiotemporal_grids:
                    total += torch.abs(1 - grids[grid_id]).mean()
            return total
        else:
            # For hash encoder, apply L1 regularization on hash table parameters
            total = 0.0
            if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'hash_tables'):
                hash_tables = self._deformation.deformation_net.grid.hash_tables
                for table in hash_tables:
                    total += torch.abs(table.weight).mean()
            return total * 0.01  # Scale factor to match hexplane regulation magnitude
            
    # The following methods should be implemented to match the GaussianModel class
    # These are placeholders that should be filled with actual implementations
    
    @property
    def get_xyz(self):
        return self._xyz
        
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return features_dc, features_rest
        
    def get_opacity(self):
        return self._opacity
        
    def get_scaling(self):
        return self._scaling
        
    def get_rotation(self):
        return self._rotation
        
    def get_covariance(self, scaling_modifier=1.0):
        return self.covariance_activation(self._scaling, scaling_modifier, self._rotation)
        
    def cat_tensors_to_optimizer(self, tensors_dict):
        # Implementation needed
        optimizable_tensors = {}
        for key, value in tensors_dict.items():
            if key == "xyz":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
            elif key == "f_dc":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
            elif key == "f_rest":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
            elif key == "opacity":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
            elif key == "scaling":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
            elif key == "rotation":
                optimizable_tensors[key] = nn.Parameter(value.requires_grad_(True))
        return optimizable_tensors
        
    def densify_and_clone(self, grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage):
        # Implementation needed
        pass
        
    def densify_and_split(self, grads, max_grad, extent):
        # Implementation needed
        pass
        
    def save_ply(self, path):
        # Implementation needed
        pass
        
    def save_deformation(self, path):
        # Implementation needed
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))
        
    def create_from_pcd(self, pcd, spatial_lr_scale=1.0):
        # Implementation needed
        pass
        
    def load_ply(self, path):
        # Implementation needed
        pass 