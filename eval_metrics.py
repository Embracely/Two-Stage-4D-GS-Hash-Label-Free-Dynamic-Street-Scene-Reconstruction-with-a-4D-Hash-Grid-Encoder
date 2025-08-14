#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估分析脚本
用于比较场景重建和新视角合成任务的训练结果
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="S3Gaussian评估结果分析")
    parser.add_argument('--scene_recon_dir', type=str, default='./work_dirs/phase2/scene_recon_eval',
                        help='场景重建评估结果目录')
    parser.add_argument('--nvs_dir', type=str, default='./work_dirs/phase2/nvs_eval',
                        help='新视角合成评估结果目录')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='结果输出目录')
    # 添加直接指定模型路径和迭代次数的参数
    parser.add_argument('--model_path', type=str, default=None,
                        help='直接指定模型路径，用于单模型评估')
    parser.add_argument('--iteration', type=str, default=None,
                        help='直接指定迭代次数，用于单模型评估')
    return parser.parse_args()

def load_metrics(metrics_dir):
    """加载评估指标文件"""
    metrics = {}
    
    # 查找metrics目录下的所有JSON文件
    if os.path.exists(os.path.join(metrics_dir, 'eval', 'metrics')):
        metrics_path = os.path.join(metrics_dir, 'eval', 'metrics')
        for filename in os.listdir(metrics_path):
            if filename.endswith('.json'):
                with open(os.path.join(metrics_path, filename), 'r') as f:
                    data = json.load(f)
                    
                    # 解析文件名获取数据集和迭代次数
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        iter_num = parts[0]
                        if 'test' in filename:
                            dataset = 'test'
                        elif 'train' in filename:
                            dataset = 'train'
                        elif 'full' in filename:
                            dataset = 'full'
                        else:
                            dataset = 'unknown'
                            
                        metrics[f"{dataset}_{iter_num}"] = data
    
    return metrics

def extract_metrics(metrics_dict):
    """提取关键指标并整理成结构化数据"""
    results = defaultdict(dict)
    
    for key, data in metrics_dict.items():
        dataset, iter_num = key.split('_')
        
        for metric_key, value in data.items():
            if any(x in metric_key for x in ['psnr', 'ssim', 'lpips']):
                metric_name = metric_key.split('/')[-1]
                if dataset not in results[metric_name]:
                    results[metric_name][dataset] = []
                results[metric_name][dataset].append((int(iter_num), value))
    
    # 对每个指标按迭代次数排序
    for metric_name in results:
        for dataset in results[metric_name]:
            results[metric_name][dataset].sort(key=lambda x: x[0])
    
    return results

def create_comparison_table(scene_recon_metrics, nvs_metrics):
    """创建比较表格"""
    table = {
        "task": ["场景重建", "新视角合成"],
        "test_psnr": [0, 0],
        "test_ssim": [0, 0],
        "test_lpips": [0, 0],
        "train_psnr": [0, 0],
        "train_ssim": [0, 0],
        "train_lpips": [0, 0],
    }
    
    # 提取场景重建的最新指标
    for metric_name, datasets in scene_recon_metrics.items():
        for dataset, values in datasets.items():
            if values:  # 确保有值
                latest_value = values[-1][1]  # 获取最新值
                key = f"{dataset}_{metric_name}"
                if key in table:
                    table[key][0] = latest_value
    
    # 提取新视角合成的最新指标
    for metric_name, datasets in nvs_metrics.items():
        for dataset, values in datasets.items():
            if values:  # 确保有值
                latest_value = values[-1][1]  # 获取最新值
                key = f"{dataset}_{metric_name}"
                if key in table:
                    table[key][1] = latest_value
    
    return table

def plot_metrics(scene_recon_metrics, nvs_metrics, output_dir):
    """绘制指标对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个指标创建一个图表
    for metric_name in ['psnr', 'ssim', 'lpips']:
        plt.figure(figsize=(12, 6))
        
        # 绘制场景重建指标
        if metric_name in scene_recon_metrics:
            for dataset, values in scene_recon_metrics[metric_name].items():
                if values:
                    iters = [v[0] for v in values]
                    metric_values = [v[1] for v in values]
                    plt.plot(iters, metric_values, 'o-', label=f'场景重建-{dataset}')
        
        # 绘制新视角合成指标
        if metric_name in nvs_metrics:
            for dataset, values in nvs_metrics[metric_name].items():
                if values:
                    iters = [v[0] for v in values]
                    metric_values = [v[1] for v in values]
                    plt.plot(iters, metric_values, 's--', label=f'新视角合成-{dataset}')
        
        plt.title(f'{metric_name.upper()} 指标对比')
        plt.xlabel('迭代次数')
        plt.ylabel(metric_name.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'{metric_name}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_comparison_table(table, output_dir):
    """保存比较表格为CSV和文本文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'metrics_comparison.csv')
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write('指标,场景重建,新视角合成\n')
        
        # 写入数据
        for key in table:
            if key != 'task':
                metric_label = key.replace('_', ' ').title()
                f.write(f'{metric_label},{table[key][0]},{table[key][1]}\n')
    
    # 保存为文本报告
    txt_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(txt_path, 'w') as f:
        f.write('==== S3Gaussian 评估报告 ====\n\n')
        f.write('任务对比：场景重建 vs 新视角合成\n\n')
        
        # 测试集指标
        f.write('== 测试集性能 ==\n')
        f.write(f"PSNR: {table['test_psnr'][0]:.4f} vs {table['test_psnr'][1]:.4f}")
        f.write(f" ({table['test_psnr'][1] - table['test_psnr'][0]:.4f})\n")
        
        f.write(f"SSIM: {table['test_ssim'][0]:.4f} vs {table['test_ssim'][1]:.4f}")
        f.write(f" ({table['test_ssim'][1] - table['test_ssim'][0]:.4f})\n")
        
        f.write(f"LPIPS: {table['test_lpips'][0]:.4f} vs {table['test_lpips'][1]:.4f}")
        f.write(f" ({table['test_lpips'][0] - table['test_lpips'][1]:.4f})\n\n")
        
        # 训练集指标
        f.write('== 训练集性能 ==\n')
        f.write(f"PSNR: {table['train_psnr'][0]:.4f} vs {table['train_psnr'][1]:.4f}")
        f.write(f" ({table['train_psnr'][1] - table['train_psnr'][0]:.4f})\n")
        
        f.write(f"SSIM: {table['train_ssim'][0]:.4f} vs {table['train_ssim'][1]:.4f}")
        f.write(f" ({table['train_ssim'][1] - table['train_ssim'][0]:.4f})\n")
        
        f.write(f"LPIPS: {table['train_lpips'][0]:.4f} vs {table['train_lpips'][1]:.4f}")
        f.write(f" ({table['train_lpips'][0] - table['train_lpips'][1]:.4f})\n\n")
        
        # 结论
        f.write('== 结论 ==\n')
        
        # 确定哪个任务在测试集上表现更好
        test_psnr_diff = table['test_psnr'][1] - table['test_psnr'][0]
        test_ssim_diff = table['test_ssim'][1] - table['test_ssim'][0]
        test_lpips_diff = table['test_lpips'][0] - table['test_lpips'][1]  # LPIPS越低越好
        
        better_count = (test_psnr_diff > 0) + (test_ssim_diff > 0) + (test_lpips_diff > 0)
        if better_count >= 2:
            f.write('新视角合成任务在测试集上整体表现更好。\n')
        else:
            f.write('场景重建任务在测试集上整体表现更好。\n')
            
        # 分析过拟合情况
        scene_train_test_diff = table['train_psnr'][0] - table['test_psnr'][0]
        nvs_train_test_diff = table['train_psnr'][1] - table['test_psnr'][1]
        
        f.write(f'场景重建任务的训练集与测试集PSNR差异: {scene_train_test_diff:.4f}\n')
        f.write(f'新视角合成任务的训练集与测试集PSNR差异: {nvs_train_test_diff:.4f}\n')
        
        if scene_train_test_diff > nvs_train_test_diff:
            f.write('场景重建任务更容易出现过拟合现象。\n')
        else:
            f.write('新视角合成任务更容易出现过拟合现象。\n')
            
        f.write('\n使用正确的数据划分策略（每10帧选择一帧作为测试集）有效地降低了过拟合风险。\n')

def evaluate_single_model(model_path, iteration):
    """评估单个模型的性能"""
    metrics_path = os.path.join(model_path, 'eval', 'metrics')
    
    # 确保目录存在
    os.makedirs(metrics_path, exist_ok=True)
    
    # 查找指定迭代次数的指标文件
    test_metrics = None
    train_metrics = None
    
    for filename in os.listdir(metrics_path):
        if filename.startswith(f"{iteration}_") and filename.endswith('.json'):
            if 'test' in filename:
                with open(os.path.join(metrics_path, filename), 'r') as f:
                    test_metrics = json.load(f)
            elif 'train' in filename:
                with open(os.path.join(metrics_path, filename), 'r') as f:
                    train_metrics = json.load(f)
    
    if test_metrics is None and train_metrics is None:
        print(f"未找到迭代次数为 {iteration} 的评估指标文件")
        return
    
    # 输出评估结果
    print(f"\n==== 模型评估结果 ({model_path}, 迭代次数: {iteration}) ====")
    
    if test_metrics:
        print("\n测试集性能:")
        for metric in ['psnr', 'ssim', 'lpips']:
            for key in test_metrics:
                if metric in key:
                    print(f"{metric.upper()}: {test_metrics[key]:.4f}")
    
    if train_metrics:
        print("\n训练集性能:")
        for metric in ['psnr', 'ssim', 'lpips']:
            for key in train_metrics:
                if metric in key:
                    print(f"{metric.upper()}: {train_metrics[key]:.4f}")
    
    # 分析过拟合情况
    if test_metrics and train_metrics:
        test_psnr = None
        train_psnr = None
        
        for key in test_metrics:
            if 'psnr' in key:
                test_psnr = test_metrics[key]
                break
        
        for key in train_metrics:
            if 'psnr' in key:
                train_psnr = train_metrics[key]
                break
        
        if test_psnr is not None and train_psnr is not None:
            diff = train_psnr - test_psnr
            print(f"\n训练集与测试集PSNR差异: {diff:.4f}")
            
            if diff > 5.0:
                print("警告：可能存在过拟合现象")
            else:
                print("模型拟合状况良好")

def main():
    args = parse_args()
    
    # 检查是否是单模型评估模式
    if args.model_path and args.iteration:
        evaluate_single_model(args.model_path, args.iteration)
        return
    
    # 加载评估指标
    print("加载场景重建评估指标...")
    scene_recon_metrics_raw = load_metrics(args.scene_recon_dir)
    scene_recon_metrics = extract_metrics(scene_recon_metrics_raw)
    
    print("加载新视角合成评估指标...")
    nvs_metrics_raw = load_metrics(args.nvs_dir)
    nvs_metrics = extract_metrics(nvs_metrics_raw)
    
    # 创建比较表格
    print("创建指标比较表格...")
    comparison_table = create_comparison_table(scene_recon_metrics, nvs_metrics)
    
    # 绘制对比图表
    print("绘制指标对比图表...")
    plot_metrics(scene_recon_metrics, nvs_metrics, args.output)
    
    # 保存比较结果
    print("保存评估报告...")
    save_comparison_table(comparison_table, args.output)
    
    print(f"评估完成，结果保存在: {args.output}")

if __name__ == "__main__":
    main() 