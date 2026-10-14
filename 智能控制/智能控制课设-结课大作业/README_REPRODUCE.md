# Arena Challenge 报告实验复现说明

本文件夹用于复现课程报告中的主实验、探索实验和残差 TD3 训练记录。最终控制器使用的是 **A* + PD 误差反馈**，代码文件名和报告表述已经统一为 PD。

## 1. 文件夹结构

```text
codex_report_repro_20260530/
├─ code/                       # MATLAB/Python 代码、sys.ini、残差模型权重
├─ maps/                       # 8 张固定地图
├─ results/                    # 报告中已经导出的 CSV 结果
├─ astar_residual_td3_runs/    # 残差 TD3 训练日志、配置和检查点
└─ README_REPRODUCE.md
```

运行 MATLAB 评测时，请把 Current Folder 切换到 `code/`。代码中的地图路径已经设置为 `../maps/`，结果文件建议输出到 `../results/`。

## 2. 主实验复现

在 PowerShell 中运行下面命令。若 MATLAB 安装路径不同，请把 `D:\MATLAB\R2024a\bin\matlab.exe` 替换为本机 MATLAB 路径。

```powershell
cd E:\auto_control\codex_report_repro_20260530\code

D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',0,[],20240527,'..\results\reproduce_policy_static.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',10,[],20240527,'..\results\reproduce_policy_dynamic10.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',20,[],20240527,'..\results\reproduce_policy_dynamic20.csv');"
```

其中 `20240527` 是 seedBase，8 张地图依次使用 `20240527--20240534`。三条命令分别对应 0、10、20 个 Smart-Agent。

## 3. 探索实验复现

下面命令复现报告第 6 章中 20 个 Smart-Agent 下的代表性探索实验：

```powershell
cd E:\auto_control\codex_report_repro_20260530\code

D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy_Exp1_DynamicPrediction',20,[],20240527,'..\results\reproduce_exp1_dynamic_prediction.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy_Exp2_DWAHybrid',20,[],20240527,'..\results\reproduce_exp2_dwa_hybrid.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy_AStar_PD_FootprintInflation',20,[],20240527,'..\results\reproduce_footprint_inflation.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy_AStar_PD_FootprintDisk',20,[],20240527,'..\results\reproduce_footprint_disk.csv');"
```

## 4. 残差 TD3 推理

`Policy_AStarPD_RLResidual` 默认读取 `code/astar_residual_policy_weights.mat`。若要使用训练记录中的最佳权重，可先复制检查点：

```powershell
cd E:\auto_control\codex_report_repro_20260530

Copy-Item .\astar_residual_td3_runs\formal_20260527_231101\best_astar_residual_policy_weights.mat .\code\astar_residual_policy_weights.mat -Force
cd .\code
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy_AStarPD_RLResidual',20,[],20240527,'..\results\reproduce_rl_residual20.csv');"
```

## 5. 残差 TD3 重新训练

训练需要 Python、PyTorch 和 MATLAB Engine for Python。本项目之前使用的解释器为 `E:\auto_control\.venv311\Scripts\python.exe`。重新训练时从复现包根目录运行：

```powershell
cd E:\auto_control\codex_report_repro_20260530

E:\auto_control\.venv311\Scripts\python.exe -B .\code\train_astar_residual_td3.py `
  --codex-dir .\code `
  --run-dir .\astar_residual_td3_runs\retrain_20240527 `
  --max-episodes 1200 `
  --min-episodes 120 `
  --window 30 `
  --patience 8 `
  --score-tolerance 20 `
  --target-reach-rate 0.95 `
  --min-score-improvement 5
```

训练日志保存在 `astar_residual_td3_runs/` 下，`metrics.txt` 记录每个 episode 的地图、Smart-Agent 数量、score、steps、reached、critic loss 和 actor loss。

## 6. 随机种子

本文定量实验使用固定 seedBase：

| 地图 | amap | a1map | a2map | bmap | b1map | b2map | cmap | c1map |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed | 20240527 | 20240528 | 20240529 | 20240530 | 20240531 | 20240532 | 20240533 | 20240534 |

静态场景在同一 MATLAB 版本下应完全一致；动态场景可能因浮点精度有极小差异，但到达情况应保持一致。
