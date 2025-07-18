# ADA-GAD 复现与可重复实验说明

## 1. 项目简介

本项目为论文《ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection》的官方代码复现，支持在多个基准数据集上进行可重复实验，并对比原论文结果。

## 2. 环境配置

详见 requirements.txt，建议使用 conda 虚拟环境。

## 3. 数据集准备

请将各数据集的 .pt 文件放置于 data/数据集名/数据集名.pt 路径下。

## 4. 运行方法

以 reddit 数据集为例：
```bash

```python main.py --use_cfg --seeds 0 --dataset reddit

## 5. 主要性能指标

- AUROC（ROC曲线下的面积）
- AUPRC（PR曲线下的面积）
- Rec@K（Top-K召回率）

## 6. 原论文主要结果（AUC%）

| 方法                | Cora   | Amazon | Weibo  | Reddit | Disney | Books  | Enron  |
|---------------------|--------|--------|--------|--------|--------|--------|--------|
| ADA-GAD (原论文)    | 84.73±0.41 | 83.25±0.43 | 98.44±0.33 | 56.89±0.41 | 70.04±3.08 | 65.24±3.17 | 72.89±0.86 |
| 复现结果（AUROC）   |          |         |         |         |         |         |         |
| 复现结果（AUPRC）   |          |         |         |         |         |         |         |
| 复现结果（Rec@K）   |          |         |         |         |         |         |         |

> 注：请在复现实验后将对应指标填写到表格中。

## 7. 性能差异分析建议

- 随机种子、依赖库版本、数据预处理、超参数设置等均可能影响复现结果。
- 建议多次运行取平均，或严格对齐论文参数。
- 如有性能差异，请结合日志、输出和论文细节进行分析。

## 8. 复现实验结果与数据脚本

- 复现实验结果请补充在本README表格中。
- 数据集转换脚本、批量运行脚本等已包含在仓库中。
