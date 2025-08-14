#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import gc
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth, l1_loss_withmask
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
# import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

import numpy as np
import time
import json
from utils.video_utils import render_pixels, save_videos
from utils.visualization_tools import compute_optical_flow_and_save
from scene.gaussian_model import merge_models
import cv2

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
TENSORBOARD_FOUND = False
   
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    "dynamic_rgbs",
    "static_rgbs",
    # "forward_flows",
    # "backward_flows",
]

@torch.no_grad()
def do_evaluation(
    viewpoint_stack_full,
    viewpoint_stack_test,
    viewpoint_stack_train,
    gaussians,
    bg,
    pipe,
    eval_dir,
    render_full,
    step: int = 0,
    args = None,
):
    # Ensure args has debug_test attribute
    if args is None or not hasattr(args, 'debug_test'):
        debug_test = False
    else:
        debug_test = args.debug_test

    if len(viewpoint_stack_test) != 0:
        print("Evaluating Test Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_test,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/test/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/test_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_test_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/test_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_test)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    if len(viewpoint_stack_train) != 0 and len(viewpoint_stack_test) != 0:
        print("Evaluating train Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_train,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=False,
            debug=debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/train/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/train_videos", exist_ok=True)
        
        train_metrics_file = f"{eval_dir}/metrics/{step}_images_train.json"
        with open(train_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {train_metrics_file}")

        video_output_pth = f"{eval_dir}/train_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_train)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results
        torch.cuda.empty_cache()

    if render_full:
        print("Evaluating Full Set...")
        render_results = render_pixels(
            viewpoint_stack_full,
            gaussians,
            bg,
            pipe,
            compute_metrics=True,
            return_decomposition=True,
            debug=debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/full/{k}"] = v
                
        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/full_videos", exist_ok=True)
        
        test_metrics_file = f"{eval_dir}/metrics/{step}_images_full_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        # if render_video_postfix is None:
        video_output_pth = f"{eval_dir}/full_videos/{step}.mp4"
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_full)//3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )
        
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_static_scene(
    viewpoint_stack,
    gaussians,
    bg,
    pipe,
    output_dir,
    step: int = 0,
    args = None,
):
    """评估静态场景重建质量，生成渲染结果与深度图
    
    Args:
        viewpoint_stack: 用于评估的相机列表
        gaussians: 高斯模型
        bg: 背景色
        pipe: 渲染管道
        output_dir: 输出目录
        step: 当前迭代步数
        args: 参数配置
    """
    print("Evaluating Static Scene...")
    
    # 创建输出目录
    os.makedirs(f"{output_dir}/static_eval", exist_ok=True)
    os.makedirs(f"{output_dir}/static_eval/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/static_eval/depth", exist_ok=True)
    os.makedirs(f"{output_dir}/static_eval/comparison", exist_ok=True)
    
    # 保存评估指标
    eval_metrics = {
        "psnr": [],
        "ssim": [],
        "l1": []
    }
    
    # 渲染每个视角
    for idx, viewpoint in enumerate(tqdm(viewpoint_stack)):
        # 只在静态区域计算指标
        render_pkg = render(viewpoint, gaussians, pipe, bg, stage="coarse")
        rendered_image = render_pkg["render"]
        depth = render_pkg["depth"]
        
        # 转换为CPU张量用于保存
        rendered_image_np = rendered_image.permute(1, 2, 0).cpu().numpy()
        depth_np = depth.cpu().numpy()
        
        # 获取GT图像
        gt_image = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
        
        # 应用动态掩码（如果有）计算静态区域的指标
        if viewpoint.dynamic_mask is not None:
            # 获取动态掩码并转换为CPU NumPy数组
            dynamic_mask = viewpoint.dynamic_mask.cpu().numpy()
            
            # 打印掩码信息以便调试
            if idx == 0:  # 只打印第一个掩码的信息
                print(f"评估函数中掩码形状: {dynamic_mask.shape}, 最小值: {dynamic_mask.min()}, 最大值: {dynamic_mask.max()}")
            
            # 确保掩码与图像形状匹配
            # 如果掩码形状为[1, H, W]，调整为[H, W, 1]
            if dynamic_mask.shape[0] == 1:
                dynamic_mask = dynamic_mask.transpose(1, 2, 0)
                
            # 归一化掩码到0-1范围
            dynamic_mask_normalized = dynamic_mask / 255.0
            
            # 创建静态区域掩码（动态区域取反）
            static_mask = 1.0 - dynamic_mask_normalized
            
            # 在静态区域计算指标
            static_rendered = rendered_image_np * static_mask
            static_gt = gt_image * static_mask
            
            # 确保掩码和图像维度匹配
            # 获取图像尺寸
            H, W, _ = rendered_image_np.shape
            
            # 重塑掩码以确保维度正确
            if static_mask.shape != (H, W, 1):
                if len(static_mask.shape) == 2:  # 如果掩码是2D的
                    static_mask = static_mask.reshape(H, W, 1)
                elif static_mask.shape[2] != 1:  # 如果掩码通道数不是1
                    static_mask = static_mask[:, :, :1]
            
            # 扁平化掩码和图像以计算指标
            mask_flat = static_mask.reshape(-1) > 0  # 确保掩码是一维布尔数组
            
            # 只有在掩码中有非零像素时才计算
            if mask_flat.sum() > 0:
                # 重塑图像数据以匹配掩码
                static_rendered_flat = static_rendered.reshape(-1, 3)
                static_gt_flat = static_gt.reshape(-1, 3)
                
                # 确保维度匹配
                if mask_flat.shape[0] != static_rendered_flat.shape[0]:
                    print(f"警告：掩码维度 {mask_flat.shape[0]} 与图像维度 {static_rendered_flat.shape[0]} 不匹配")
                    # 调整掩码维度以匹配图像
                    mask_flat = mask_flat[:static_rendered_flat.shape[0]] if mask_flat.shape[0] > static_rendered_flat.shape[0] else np.pad(mask_flat, (0, static_rendered_flat.shape[0] - mask_flat.shape[0]))
                
                # 应用掩码到扁平化的图像
                masked_rendered = static_rendered_flat[mask_flat]
                masked_gt = static_gt_flat[mask_flat]
                
                # 计算PSNR
                mse = ((masked_rendered - masked_gt) ** 2).mean()
                if mse > 0:
                    static_psnr = 10 * np.log10(1.0 / mse)
                else:
                    static_psnr = 100.0  # 防止除零
                
                # 计算L1
                static_l1 = np.abs(masked_rendered - masked_gt).mean()
                
                # 保存指标
                eval_metrics["psnr"].append(static_psnr)
                eval_metrics["l1"].append(static_l1)
        else:
            # 没有掩码时计算整体指标
            mse = ((rendered_image_np - gt_image) ** 2).mean()
            if mse > 0:
                img_psnr = 10 * np.log10(1.0 / mse)
            else:
                img_psnr = 100.0
            
            img_l1 = np.abs(rendered_image_np - gt_image).mean()
            
            eval_metrics["psnr"].append(img_psnr)
            eval_metrics["l1"].append(img_l1)
        
        # 保存渲染结果和深度图
        rgb_output = (rendered_image_np * 255).astype(np.uint8)
        depth_output = (depth_np / depth_np.max() * 255).astype(np.uint8)
        
        # 创建比较图：渲染 | GT
        comparison = np.concatenate([rendered_image_np, gt_image], axis=1)
        comparison = (comparison * 255).astype(np.uint8)
        
        # 保存图像
        rgb_path = f"{output_dir}/static_eval/rgb/{idx:03d}.png"
        depth_path = f"{output_dir}/static_eval/depth/{idx:03d}.png"
        comparison_path = f"{output_dir}/static_eval/comparison/{idx:03d}.png"
        
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_path, depth_output)
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in eval_metrics.items() if len(v) > 0}
    
    # 保存评估指标到JSON
    metrics_file = f"{output_dir}/static_eval/metrics_{step}.json"
    with open(metrics_file, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    
    print(f"Static scene evaluation complete. Results saved to {output_dir}/static_eval/")
    print(f"Average metrics: PSNR={avg_metrics.get('psnr', 0):.2f}, L1={avg_metrics.get('l1', 0):.4f}")
    
    return avg_metrics

def visualize_dynamic_masks(cameras, output_dir):
    """可视化相机中的动态掩码，确保它们被正确加载
    
    Args:
        cameras: 相机列表
        output_dir: 输出目录
    """
    print("检查动态掩码加载情况...")
    
    mask_dir = os.path.join(output_dir, "dynamic_masks_check")
    os.makedirs(mask_dir, exist_ok=True)
    
    # 随机选择几个相机进行检查
    sample_size = min(5, len(cameras))
    sample_cameras = random.sample(cameras, sample_size)
    
    for i, cam in enumerate(sample_cameras):
        if cam.dynamic_mask is not None:
            # 保存原始掩码
            original_mask = cam.dynamic_mask.cpu().numpy()
            # 打印掩码形状信息以便调试
            print(f"相机 {i} 动态掩码形状: {original_mask.shape}")
            
            # 确保掩码是3D数组，形状为[H, W, C]
            if original_mask.shape[0] == 1:  # 形状为[1, H, W]
                original_mask = original_mask.transpose(1, 2, 0)
            
            original_mask_path = os.path.join(mask_dir, f"original_mask_{i}.png")
            cv2.imwrite(original_mask_path, original_mask)
            
            # 保存归一化后的掩码
            normalized_mask = (original_mask / 255.0 * 255).astype(np.uint8)
            normalized_mask_path = os.path.join(mask_dir, f"normalized_mask_{i}.png")
            cv2.imwrite(normalized_mask_path, normalized_mask)
            
            # 保存权重掩码（静态权重为1.0，动态权重为0.1）
            weight_mask = (1.0 - normalized_mask / 255.0 * 0.9) * 255
            weight_mask_path = os.path.join(mask_dir, f"weight_mask_{i}.png")
            cv2.imwrite(weight_mask_path, weight_mask.astype(np.uint8))
            
            print(f"相机 {i} 动态掩码 - 最小值: {original_mask.min()}, 最大值: {original_mask.max()}, 均值: {original_mask.mean()}")
        else:
            print(f"相机 {i} 没有动态掩码")
    
    print(f"动态掩码检查完成，结果保存在 {mask_dir}")

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.eval_only:
        torch.save(gaussians._deformation.state_dict(),os.path.join(args.model_path, "deformation.pth"))

        eval_dir = os.path.join(args.model_path,"eval")
        os.makedirs(eval_dir,exist_ok=True)
        viewpoint_stack_full = scene.getFullCameras().copy()
        viewpoint_stack_test = scene.getTestCameras().copy()
        viewpoint_stack_train = scene.getTrainCameras().copy()

        # TODO：可视化光流 and 静动态点云分离
        do_evaluation(
            viewpoint_stack_full,
            viewpoint_stack_test,
            viewpoint_stack_train,
            gaussians,
            background,
            pipe,
            eval_dir,
            render_full=True,
            step=first_iter,
            args=args
        )
        # save 静动态点云分离
        # pcd_dir = os.path.join(eval_dir, "split_pcd")
        # os.makedirs(eval_dir,exist_ok=True)

        # gaussians.save_ply_split(pcd_dir)
        exit()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    # video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack:
        
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    
    batch_size = opt.batch_size
    print("data loading done")    
        
    count = 0
    psnr_dict = {}
    for iteration in range(first_iter, final_iter+1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             count +=1
        #             viewpoint_index = (count ) % len(video_cams)
        #             if (count //(len(video_cams))) % 2 == 0:
        #                 viewpoint_index = viewpoint_index
        #             else:
        #                 viewpoint_index = len(video_cams) - viewpoint_index - 1
        #             # print(viewpoint_index)
        #             viewpoint = video_cams[viewpoint_index]
        #             custom_cam.time = viewpoint.time
        #             # print(custom_cam.time, viewpoint_index, count)
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage)["render"]

        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
        #             break
        #     except Exception as e:
        #         print(e)
        #         network_gui.conn = None

        iter_start.record()

        position_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # batch size
        idx = 0
        viewpoint_cams = []

        while idx < batch_size :    
            
            viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
            if not viewpoint_stack :
                viewpoint_stack =  temp_list.copy()
                # print("find the worst viewpoint")
                # 对 PSNR 字典按值进行排序，找出最低的 PSNR 值对应的 UID, 最后一个psnr 没办法得到，所以实际上比较 n*3 -1 个psnr
                # with torch.no_grad():
                #     if 'fine' in stage:
                #         psnr_dict = sorted(psnr_dict.items(), key=lambda x: x[1])

                #         # 将最低 PSNR 值对应的 UID 添加到列表中，直到列表的长度达到 args.end_time / 5
                #         lowest_psnr_uids = []
                #         for uid, _ in psnr_dict[:(args.end_time+1)]:
                #             lowest_psnr_uids.append(uid)
                            
                #         psnr_dict = {}

                #         # 将 lowest_psnr_uids 中 UID 对应的 Camera 对象加到 viewpoint_stack 的末尾
                #         for uid in lowest_psnr_uids:
                #             for cam in viewpoint_stack:
                #                 if cam.uid == int(uid):
                #                     viewpoint_stack.append(cam)
                #                     break                
                
            viewpoint_cams.append(viewpoint_cam)
            idx +=1
        if len(viewpoint_cams) == 0:
            continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        depth_preds = []
        gt_depths = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,return_dx=True,render_feat = True if ('fine' in stage and args.feat_head) else False)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth_pred = render_pkg["depth"]
            depth_preds.append(depth_pred.unsqueeze(0))
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.depth_map.cuda()

            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_pred_tensor = torch.cat(depth_preds,0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths,0).float()
        # Loss
        # breakpoint()
        # 检查是否存在动态掩码并应用于损失计算
        dynamic_masks = []
        use_dynamic_mask = 'phase1_static' in args.configs and args.load_dynamic_mask
        
        if use_dynamic_mask:
            for viewpoint_cam in viewpoint_cams:  # 使用正确的变量名viewpoint_cams
                if viewpoint_cam.dynamic_mask is not None:
                    # 动态区域权重为0.1，静态区域权重为1.0
                    # 确保动态掩码在GPU上并归一化为0-1范围
                    # 动态掩码是0-255灰度图，需要归一化到0-1
                    dynamic_mask = viewpoint_cam.dynamic_mask.cuda() / 255.0
                    
                    # 检查掩码是否正确加载
                    if iteration % 1000 == 0 and viewpoint_cam == viewpoint_cams[0]:
                        print(f"动态掩码信息: 最小值={dynamic_mask.min().item()}, 最大值={dynamic_mask.max().item()}, 均值={dynamic_mask.mean().item()}")
                        
                    # 动态区域权重为0.1，静态区域权重为1.0
                    # 掩码中1表示动态区域，0表示静态区域
                    weight_mask = 1.0 - dynamic_mask * 0.9
                    dynamic_masks.append(weight_mask.unsqueeze(0))
                else:
                    # 如果没有掩码，使用全1权重
                    dynamic_masks.append(torch.ones(1, 1, viewpoint_cam.image.shape[1], viewpoint_cam.image.shape[2], device=image_tensor.device))
            
            # 合并所有相机的掩码
            weight_mask_tensor = torch.cat(dynamic_masks, 0)
            
            # 确保掩码的形状与图像张量匹配（适用于RGB通道）
            # 当前掩码形状为 [B, 1, H, W]，需要扩展为 [B, 3, H, W]
            expanded_mask = weight_mask_tensor.expand(-1, 3, -1, -1)
            
            # 使用l1_loss_withmask函数计算带掩码的L1损失
            Ll1 = l1_loss_withmask(image_tensor, gt_image_tensor[:,:3,:,:], expanded_mask)
            
            # 每1000次迭代保存一次掩码可视化（调试用）
            if iteration % 1000 == 0:
                mask_vis = weight_mask_tensor[0].permute(1, 2, 0).cpu().numpy() * 255
                mask_path = f"{scene.model_path}/mask_{iteration}.png"
                import cv2
                cv2.imwrite(mask_path, mask_vis)
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # if 'fine' in stage:
        #     psnr_dict.update({f"{viewpoint_cam.uid}": psnr_})
        # norm        
        loss = Ll1
        # dx loss
        if 'fine' in stage and not args.no_dx and opt.lambda_dx !=0:
            dx_abs = torch.abs(render_pkg['dx'])
            dx_loss = torch.mean(dx_abs) * opt.lambda_dx
            loss += dx_loss
        if 'fine' in stage and not args.no_dshs and opt.lambda_dshs != 0:
            dshs_abs = torch.abs(render_pkg['dshs'])
            dshs_loss = torch.mean(dshs_abs) * opt.lambda_dshs
            loss += dshs_loss
        if opt.lambda_depth != 0:
            depth_loss = compute_depth("l2", depth_pred_tensor, gt_depth_tensor) * opt.lambda_depth
            loss += depth_loss
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if stage == 'fine' and args.feat_head:
            feat = render_pkg['feat'].to('cuda') # [3,640,960]
            if viewpoint_cam.feat_map is not None:
                gt_feat = viewpoint_cam.feat_map.permute(2,0,1).to('cuda')
                loss_feat = l2_loss(feat, gt_feat) * opt.lambda_feat
                loss += loss_feat
            
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
                dynamic_points = 0
                if 'fine' in stage and not args.no_dx:
                    dx_abs = torch.abs(render_pkg['dx']) # [N,3]
                    max_values = torch.max(dx_abs, dim=1)[0] # [N]
                    thre = torch.mean(max_values)                    
                    mask = (max_values > thre)
                    dynamic_points = torch.sum(mask).item()

                print_dict = {
                    "step": f"{iteration}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "dynamic point": f"{dynamic_points}",
                    "point":f"{total_point}",
                    }
                progress_bar.set_postfix(print_dict)
                metrics_file = f"{scene.model_path}/logger.json"
                with open(metrics_file, "a") as f:
                    json.dump(print_dict, f)
                    f.write('\n')

                progress_bar.update(100)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 10000 and iteration % 1000 == 999) \
                    or (iteration < 30000 and iteration % 2000 == 1999) \
                        or (iteration < 60000 and iteration %  3000 == 2999) :
                    # breakpoint()
                        if len(test_cams) != 0:
                            render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time())
                        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time())

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0]>200000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 添加日志记录，跟踪修剪前的点数
                    pre_prune_point_count = gaussians.get_xyz.shape[0]
                    pre_prune_opacity_min = gaussians.get_opacity.min().item()
                    pre_prune_opacity_mean = gaussians.get_opacity.mean().item()
                    
                    # 执行修剪操作
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                    # 添加日志记录，跟踪修剪后的点数
                    post_prune_point_count = gaussians.get_xyz.shape[0]
                    if post_prune_point_count > 0:  # 避免在所有点都被修剪掉的情况下出错
                        post_prune_opacity_min = gaussians.get_opacity.min().item()
                        post_prune_opacity_mean = gaussians.get_opacity.mean().item()
                        pruned_points = pre_prune_point_count - post_prune_point_count
                        pruned_percent = (pruned_points / pre_prune_point_count) * 100 if pre_prune_point_count > 0 else 0
                        
                        print(f"[PRUNING STATS] Iter {iteration}: Before={pre_prune_point_count} points (opacity min={pre_prune_opacity_min:.6f}, mean={pre_prune_opacity_mean:.6f})")
                        print(f"[PRUNING STATS] Iter {iteration}: After={post_prune_point_count} points (opacity min={post_prune_opacity_min:.6f}, mean={post_prune_opacity_mean:.6f})")
                        print(f"[PRUNING STATS] Iter {iteration}: Pruned {pruned_points} points ({pruned_percent:.2f}%)")
                        
                        # 如果修剪率超过10%，发出警告
                        if pruned_percent > 10 and iteration > 1000:
                            print(f"[PRUNING WARNING] Iter {iteration}: High pruning rate detected ({pruned_percent:.2f}%)! Consider adjusting opacity_threshold or pruning_interval.")
                        
                        # 灾难性修剪恢复机制
                        if pruned_percent > 50 and pre_prune_point_count > 10000:
                            print(f"[PRUNING EMERGENCY] Iter {iteration}: Catastrophic pruning detected! {pruned_percent:.2f}% of points were pruned.")
                            
                            # 查找最近的检查点
                            checkpoint_files = [f for f in os.listdir(scene.model_path) if f.endswith(".pth")]
                            if checkpoint_files:
                                latest_checkpoint = os.path.join(scene.model_path, sorted(checkpoint_files)[-1])
                                print(f"[PRUNING RECOVERY] Attempting to restore from checkpoint: {latest_checkpoint}")
                                
                                try:
                                    # 加载检查点
                                    model_params, _ = torch.load(latest_checkpoint)
                                    gaussians.restore(model_params, opt)
                                    print(f"[PRUNING RECOVERY] Successfully restored {gaussians.get_xyz.shape[0]} points from checkpoint.")
                                    
                                    # 调整修剪参数以避免再次发生
                                    opt.opacity_threshold_fine_init *= 0.5
                                    opt.opacity_threshold_fine_after *= 0.5
                                    opt.pruning_interval *= 2  # 降低修剪频率
                                    print(f"[PRUNING RECOVERY] Adjusted pruning parameters: opacity_threshold *= 0.5, pruning_interval *= 2")
                                except Exception as e:
                                    print(f"[PRUNING RECOVERY] Failed to restore from checkpoint: {e}")
                            else:
                                print(f"[PRUNING EMERGENCY] No checkpoints found for recovery!")
                    else:
                        print(f"[PRUNING ERROR] Iter {iteration}: All points were pruned! Opacity threshold may be too high.")
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                # if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                #     gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            
            # Optimizer step
            if iteration < final_iter+1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                save_path= "chkpnt" +f"_{stage}_" + str(30000) + ".pth"
                for file in os.listdir(scene.model_path):
                    if file.endswith(".pth") and file != save_path:
                        os.remove(os.path.join(scene.model_path, file))

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

            # 每10000次迭代或训练结束时进行评估
            if (iteration == 30000) or (iteration % 10000 == 0 and iteration > 0 and 'phase1_static' in args.configs):
                eval_dir = os.path.join(args.model_path,"eval")
                os.makedirs(eval_dir,exist_ok=True)
                viewpoint_stack_full = scene.getFullCameras().copy()
                viewpoint_stack_test = scene.getTestCameras().copy()
                viewpoint_stack_train = scene.getTrainCameras().copy()

                # 常规评估
                do_evaluation(
                    viewpoint_stack_full,
                    viewpoint_stack_test,
                    viewpoint_stack_train,
                    gaussians,
                    background,
                    pipe,
                    eval_dir,
                    render_full=True,
                    step=iteration,
                    args=args
                )
                
                # 对静态场景进行专门评估
                if 'phase1_static' in args.configs:
                    print("\n[ITER {}] Evaluating Static Scene...".format(iteration))
                    evaluate_static_scene(
                        viewpoint_stack_test,  # 使用测试集相机
                        gaussians,
                        background,
                        pipe,
                        eval_dir,
                        step=iteration,
                        args=args
                    )

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)        

    gaussians = GaussianModel(dataset.sh_degree, hyper)
        
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    
    # 可视化动态掩码，确保它们被正确加载
    if args.load_dynamic_mask and 'phase1_static' in args.configs:
        test_cams = scene.getTestCameras()
        visualize_dynamic_masks(test_cams, args.model_path)
    
    # eval
    eval_dir = os.path.join(args.model_path,"eval")
    os.makedirs(eval_dir,exist_ok=True)
    viewpoint_stack_full = scene.getFullCameras().copy()
    viewpoint_stack_test = scene.getTestCameras().copy()
    viewpoint_stack_train = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # if args.merge and args.prior_checkpoint and args.prior_checkpoint2:
    #     # 这个是最新的，deformation的网络要用这个
    #     gaussians_new = GaussianModel(dataset.sh_degree, hyper)
    #     (model_params, first_iter) = torch.load(args.prior_checkpoint2)
    #     gaussians_new.restore(model_params, opt)
    #     deformation_net = gaussians_new._deformation
    #     del gaussians_new
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
    #     # 这个是上一个
    #     gaussians_prev = GaussianModel(dataset.sh_degree, hyper)
    #     (model_params, first_iter) = torch.load(args.prior_checkpoint)
    #     gaussians_prev.restore(model_params, opt)
               
    #     gaussians_prev._deformation = deformation_net.to('cuda')

        
    #     do_evaluation(
    #         viewpoint_stack_full,
    #         viewpoint_stack_test,
    #         gaussians_prev,
    #         background,
    #         pipe,
    #         eval_dir,
    #         render_full=True,
    #         step=99999,
    #         args=args
    #     )
        
        # merge
        # gaussians = merge_models(gaussians_new, gaussians_prev, hyper, gaussians)

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)

    # 保存第一阶段结果，用于可能的第二阶段使用
    if args.use_first_stage_result:
        first_stage_checkpoint = os.path.join(args.model_path, "first_stage_checkpoint.pth")
        print(f"\n[CONTINUOUS TRAINING] 保存第一阶段结果到 {first_stage_checkpoint}")
        torch.save((gaussians.capture(), opt.coarse_iterations), first_stage_checkpoint)
        print("[CONTINUOUS TRAINING] 将使用第一阶段训练结果作为第二阶段起点")
        # 不需要从prior_checkpoint加载，直接使用当前的gaussians继续训练

    elif args.prior_checkpoint:
        # 加载静态模型作为初始化
        print(f"正在从{args.prior_checkpoint}加载静态模型参数...")
        loaded_data = torch.load(args.prior_checkpoint)
        model_params, first_iter = loaded_data
        
        # 手动复制所有高斯参数，但保留当前的变形网络
        gaussians._xyz = model_params[1].to("cuda")  # xyz
        gaussians._features_dc = model_params[4].to("cuda")  # features_dc
        gaussians._features_rest = model_params[5].to("cuda")  # features_rest
        gaussians._scaling = model_params[6].to("cuda")  # scaling
        gaussians._rotation = model_params[7].to("cuda")  # rotation
        gaussians._opacity = model_params[8].to("cuda")  # opacity
        gaussians.max_radii2D = model_params[9].to("cuda")  # max_radii2D
        
        # 重新计算_deformation_table
        gaussians._deformation_table = torch.gt(torch.ones((gaussians.get_xyz.shape[0]), device="cuda"), 0)
        
        # 不复制变形网络
        print("已成功加载静态模型参数，初始化新的动态变形网络.")
    
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)
    
    do_evaluation(
        viewpoint_stack_full,
        viewpoint_stack_test,
        viewpoint_stack_train,
        gaussians,
        background,
        pipe,
        eval_dir,
        render_full=True,
        step=opt.iterations,
        args=args
    )

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = None
        # tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if len(scene.getTestCameras()) != 0: 
            validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})
        else:
            validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_training_modes_help():
    """打印训练模式帮助信息"""
    help_text = """
=== S3Gaussian 训练模式说明 ===

本项目支持两种训练模式:

1. 分阶段训练 (默认模式):
   - 第一阶段: 使用预训练的静态模型作为第二阶段的起点
   - 命令示例: python train.py --configs arguments/phase2_dynamic.py --model_path ./work_dirs/phase2/dynamic_scene --prior_checkpoint ./work_dirs/phase1/static_background/chkpnt_fine_30000.pth --source_path ./data/waymo/processed/training/022

2. 连续训练 (新增模式):
   - 使用第一阶段的训练结果作为第二阶段的起点
   - 命令示例: python train.py --configs arguments/phase2_dynamic.py --model_path ./work_dirs/phase2/continuous_training --source_path ./data/waymo/processed/training/022 --use_first_stage_result

注意: 连续训练模式下不需要指定 --prior_checkpoint 参数
"""
    print(help_text)

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000,20_000,30_000,40_000,50_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "waymo")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--eval_only", action="store_true", help="perform evaluation only")
    parser.add_argument("--prior_checkpoint", type=str, default = None)
    parser.add_argument("--merge", action="store_true", help="merge gaussians")
    parser.add_argument("--prior_checkpoint2", type=str, default = None)
    parser.add_argument("--use_first_stage_result", action="store_true", help="使用第一阶段训练结果作为第二阶段起点")
    parser.add_argument("--help_training_modes", action="store_true", help="显示训练模式帮助信息")
    parser.add_argument("--encoder_type", type=str, default="hexplane", choices=["hexplane", "hash"], help="Type of encoder to use")
    
    args = parser.parse_args(sys.argv[1:])
    if args.encoder_type == "hash" and not hasattr(args, "hash_config"):
        # Set default hash encoder parameters if not specified in config
        args.hash_config = {
            "n_levels": 16,
            "min_resolution": 16,
            "max_resolution": 512,
            "log2_hashmap_size": 15,
            "feature_dim": 2,
        }
    if args.encoder_type == "hash" and not hasattr(args, "hash_config"):
        # Set default hash encoder parameters if not specified in config
        args.hash_config = {
            "n_levels": 16,
            "min_resolution": 16,
            "max_resolution": 512,
            "log2_hashmap_size": 15,
            "feature_dim": 2,
        }
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # 显示训练模式帮助信息
    if args.help_training_modes:
        print_training_modes_help()
        sys.exit(0)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
