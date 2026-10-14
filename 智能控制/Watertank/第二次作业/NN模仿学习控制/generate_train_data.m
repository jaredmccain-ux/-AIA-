%% generate_train_data.m
%  利用现有模糊控制器 (Policy.m) 采集训练数据 (增强版)
%  改进: 1. 更多 episode 覆盖多种初始水位
%        2. 添加合成专家数据（大误差→饱和输出），确保 NN 学会快速响应
%        3. 极端初始水位(接近0和接近30)专门采集
%  运行方法: 在 MATLAB 命令行中执行 generate_train_data
%  输出: train_data.mat (包含 X_data, Y_data)

clear all; clc;

%% ---- 配置 ----
num_episodes  = 50;    % 仿真回合数（增加到50）
fprintf('=== 开始生成训练数据 (%d episodes) ===\n', num_episodes);

%% ---- 创建无显示的配置文件 ----
iniStr = fileread('sys.ini');
iniStr = strrep(iniStr, 'showViewer=1', 'showViewer=0');
fid = fopen('sys_train.ini', 'w');
fprintf(fid, '%s', iniStr);
fclose(fid);

%% ---- 系统参数 (从 ini 读取保持一致) ----
ip = 2.5;
op = 2.5;
satLevel = 15;  % 控制量饱和值

%% ---- 数据收集 ----
all_features = [];   % N x 4
all_actions  = [];   % N x 1

% 设计初始水位序列：覆盖极端值 + 随机值
% 前 10 个 episode 用固定的极端初始水位
fixed_inits = [0.5, 1, 2, 3, 5, 15, 20, 28, 30, 35];
for ep = 1:num_episodes
    fprintf('Episode %d/%d ... ', ep, num_episodes);
    
    % 创建环境和策略
    env    = Env('sys_train.ini');
    policy = Policy();
    
    if ~env.succeed
        warning('环境加载失败，跳过 episode %d', ep);
        continue;
    end
    
    observation = env.reset();
    
    % 手动覆盖初始水位
    if ep <= length(fixed_inits)
        init_H = fixed_inits(ep);  % 使用固定极端值
    else
        init_H = rand() * 35;     % 随机覆盖 0~35 范围（扩大范围）
    end
    observation.agent.H = init_H;
    
    last_e = 0;
    first_step = true;
    step_count = 0;
    
    while true
        % 提取状态
        H  = observation.agent.H;
        sp = observation.targetHeight;
        e  = sp - H;
        
        if first_step
            de = 0;
            first_step = false;
        else
            de = e - last_e;
        end
        
        % 前馈参考值
        u_eq = op * sqrt(max(sp, 0)) / ip;
        
        % 调用模糊控制器获取动作（ground truth）
        action = policy.action(observation);
        
        % 构造特征向量: [e, de, sp, u_eq]
        features = [e, de, sp, u_eq];
        
        % 收集数据
        all_features = [all_features; features];
        all_actions  = [all_actions;  action(1)];
        
        last_e = e;
        step_count = step_count + 1;
        
        % 环境步进
        [observation, done, ~] = env.step(action);
        
        if done
            break;
        end
    end
    
    fprintf('采集 %d 步\n', step_count);
end

%% ---- 添加合成专家数据（大量） ----
% 核心思想: 仿真数据约75000条，大部分在稳态(e≈0)附近
% 合成数据需要足够多才不会被淹没
% 用 PD 控制公式: action = u_eq + Kp(e)*e - Kd*de
%   - Kp(e): 小误差用小增益(防振荡)，大误差用大增益(快速响应)
%   - Kd: 微分阻尼，防止超调和振荡
fprintf('\n生成合成专家数据...\n');

synth_features = [];
synth_actions  = [];

% PD 参数
Kp_max  = 3.0;     % 大误差时的比例增益
Kp_min  = 0.5;     % 小误差时的比例增益（防振荡）
e_trans = 2.0;     % 增益过渡的误差阈值
Kd_synth = 1.0;    % 微分阻尼系数

sp_values = [5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30];
% 正误差密集采样（特别是0~2的精细区间）
e_values_pos  = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10, 15, 20];
e_values_neg  = [-0.1, -0.2, -0.3, -0.5, -0.8, -1.0, -1.5, -2.0, -3.0, -5.0, -8.0, -10, -15, -20];
de_values     = [-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2];

n_repeat = 30;  % 每条合成数据重复次数，确保不被仿真数据淹没

for sp = sp_values
    u_eq = op * sqrt(max(sp, 0)) / ip;
    
    % ---- 正误差：水位低于目标 ----
    for e = e_values_pos
        % 非线性增益: 小误差→小Kp，大误差→大Kp，平滑过渡
        Kp = Kp_min + (Kp_max - Kp_min) * min(abs(e) / e_trans, 1);
        for de = de_values
            % PD 修正: u = u_eq + Kp*e - Kd*de
            u_target = u_eq + Kp * e + Kd_synth * de;
            u_target = min(u_target, satLevel);
            u_target = max(u_target, 0);
            for rep = 1:n_repeat
                synth_features = [synth_features; e, de, sp, u_eq];
                synth_actions  = [synth_actions; u_target];
            end
        end
    end
    
    % ---- 负误差：水位高于目标 ----
    for e = e_values_neg
        Kp = Kp_min + (Kp_max - Kp_min) * min(abs(e) / e_trans, 1);
        for de = de_values
            u_target = u_eq + Kp * e + Kd_synth * de;
            u_target = max(u_target, 0);
            u_target = min(u_target, satLevel);
            for rep = 1:n_repeat
                synth_features = [synth_features; e, de, sp, u_eq];
                synth_actions  = [synth_actions; u_target];
            end
        end
    end
    
    % ---- 零误差：稳态，de 不同时也有微分阻尼 ----
    for de = de_values
        u_target = u_eq + Kd_synth * de;  % 稳态时微分阻尼抑制扰动
        u_target = max(u_target, 0);
        u_target = min(u_target, satLevel);
        for rep = 1:n_repeat
            synth_features = [synth_features; 0, de, sp, u_eq];
            synth_actions  = [synth_actions; u_target];
        end
    end
end

fprintf('合成数据: %d 条 (每条重复 %d 次)\n', size(synth_features, 1), n_repeat);

% 合并仿真数据和合成数据
all_features = [all_features; synth_features];
all_actions  = [all_actions;  synth_actions];

%% ---- 数据归一化参数 ----
X_data = all_features;
Y_data = all_actions;

% 输入归一化: 对每个特征计算 mean 和 std
X_mean = mean(X_data, 1);
X_std  = std(X_data, 0, 1);
X_std(X_std < 1e-6) = 1;  % 防止除 0

% 输出归一化
Y_mean = mean(Y_data);
Y_std  = std(Y_data);
if Y_std < 1e-6, Y_std = 1; end

fprintf('\n=== 数据采集完成 ===\n');
fprintf('总样本数: %d (仿真: %d + 合成: %d)\n', ...
    size(X_data, 1), size(X_data, 1) - size(synth_features, 1), size(synth_features, 1));
fprintf('特征维度: %d\n', size(X_data, 2));
fprintf('输入特征统计:\n');
feature_names = {'e(误差)', 'de(误差变化)', 'sp(设定值)', 'u_eq(平衡点)'};
for i = 1:4
    fprintf('  %s: mean=%.3f, std=%.3f, range=[%.3f, %.3f]\n', ...
        feature_names{i}, X_mean(i), X_std(i), min(X_data(:,i)), max(X_data(:,i)));
end
fprintf('输出(action): mean=%.3f, std=%.3f, range=[%.3f, %.3f]\n', ...
    Y_mean, Y_std, min(Y_data), max(Y_data));

%% ---- 保存 ----
save('train_data.mat', 'X_data', 'Y_data', 'X_mean', 'X_std', 'Y_mean', 'Y_std');
fprintf('数据已保存到 train_data.mat\n');

%% ---- 清理临时文件 ----
if exist('sys_train.ini', 'file')
    delete('sys_train.ini');
end
