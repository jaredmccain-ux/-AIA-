# Arena Challenge 报告实验复现说明

请阅读 `README_REPRODUCE.md`。本复现包的代码统一放在 `code/` 中，地图在 `maps/` 中，已有实验结果在 `results/` 中，残差 TD3 训练记录在 `astar_residual_td3_runs/` 中。

最常用的主实验复现命令如下：

```powershell
cd E:\auto_control\codex_report_repro_20260530\code

D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',0,[],20240527,'..\results\reproduce_policy_static.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',10,[],20240527,'..\results\reproduce_policy_dynamic10.csv');"
D:\MATLAB\R2024a\bin\matlab.exe -batch "eval_policy_class_maps('Policy',20,[],20240527,'..\results\reproduce_policy_dynamic20.csv');"
```
