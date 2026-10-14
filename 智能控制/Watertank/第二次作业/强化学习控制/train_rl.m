%% train_rl.m
%  可微仿真 (Differentiable Simulation) 训练
%  直接通过水箱动力学反向传播精确梯度，无需探索噪声
%  前置条件: nn_weights.mat (可选, warm start)
%  输出: nn_weights_rl.mat

clear all; clc; close all;

%% ---- 超参数 ----
n_epochs    = 500;
batch_size  = 8;        % 每 epoch 多次采样 (初始水位随机范围小)
lr          = 5e-4;
grad_clip   = 5.0;

% 网络结构: 4 -> 64 -> 32 -> 16 -> 1
n_in = 4; h1 = 64; h2 = 32; h3 = 16; n_out = 1;
al = 0.01;  % Leaky ReLU alpha

% 物理参数
ip_p = 2.5; op_p = 2.5; satLev = 15; dt = 0.01;
tend = 15.0; T = round(tend / dt);  % 1500 步
sp1 = 25; sp2 = 10; t_switch = 7.5;
EPS_H = 1e-6;  % sqrt 保护

% 奖励塑形参数
reach_thresh = 0.5;
reach_bonus = 50;
timeout_penalty = 500;
stability_w = 20;
steady_err_w = 30;
score_align_w = 0.5;
score_loss_threshold = 230;
score_loss_penalty_w = 0.1;
init_env_score = 10000;

%% ---- 初始化权重 ----
if exist('nn_weights.mat', 'file')
    fprintf('Warm start: 加载预训练权重\n');
    d = load('nn_weights.mat');
    W1=d.W1; b1=d.b1; W2=d.W2; b2=d.b2;
    W3=d.W3; b3=d.b3; W4=d.W4; b4=d.b4;
    Xm=d.X_mean; Xs=d.X_std; Ym=d.Y_mean; Ys=d.Y_std;
else
    fprintf('Cold start: 随机初始化\n');
    rng(42);
    W1=randn(n_in,h1)*sqrt(2/n_in); b1=zeros(1,h1);
    W2=randn(h1,h2)*sqrt(2/h1);     b2=zeros(1,h2);
    W3=randn(h2,h3)*sqrt(2/h2);     b3=zeros(1,h3);
    W4=randn(h3,n_out)*sqrt(2/h3);  b4=zeros(1,n_out);
    Xm=[0,0,17.5,4]; Xs=[10,2,7.5,1]; Ym=5; Ys=5;
end

%% ---- 创建无显示配置 (验证用) ----
iniStr = fileread('sys.ini');
iniStr = strrep(iniStr, 'showViewer=1', 'showViewer=0');
fid = fopen('sys_rl.ini','w'); fprintf(fid,'%s',iniStr); fclose(fid);

%% ---- Adam 状态 ----
np = numel(W1)+numel(b1)+numel(W2)+numel(b2)+numel(W3)+numel(b3)+numel(W4)+numel(b4);
m_ad = zeros(1,np); v_ad = zeros(1,np);
b1_ad=0.9; b2_ad=0.999; eps_ad=1e-8; t_ad=0;
fprintf('总参数量: %d\n', np);

%% ---- 训练循环 ----
best_sc = -inf;   % 训练分数
best_val = -inf;  % 真实验证分数 (用于最终选模型)
best_W1=W1; best_b1=b1; best_W2=W2; best_b2=b2;
best_W3=W3; best_b3=b3; best_W4=W4; best_b4=b4;
hist_data = zeros(n_epochs, 4);  % [mean_score, best_score, mean_p1, val_score]
val_freq = 10;  % 每10个epoch做一次真实环境验证

fprintf('\n=== 可微仿真训练 (epochs=%d, batch=%d, params=%d) ===\n', n_epochs, batch_size, np);

for ep = 1:n_epochs
    % 累积梯度
    gW1=zeros(size(W1)); gb1=zeros(size(b1));
    gW2=zeros(size(W2)); gb2=zeros(size(b2));
    gW3=zeros(size(W3)); gb3=zeros(size(b3));
    gW4=zeros(size(W4)); gb4=zeros(size(b4));
    
    scores = zeros(batch_size, 1);
    scores_p1 = zeros(batch_size, 1);
    
    for b = 1:batch_size
        % 初始水位与 Env.m 一致: rand()+5 (范围 5~6)
        H0 = rand() + 5;
        
        % ===== 前向仿真 =====
        [sc, sc_p1, dW1,db1_g,dW2,db2_g,dW3,db3_g,dW4,db4_g] = ...
            run_episode(H0, W1,b1,W2,b2,W3,b3,W4,b4, Xm,Xs,Ym,Ys, ...
            al, ip_p,op_p,satLev,dt,T,sp1,sp2,t_switch, EPS_H, ...
            reach_thresh,reach_bonus,timeout_penalty,stability_w,steady_err_w, ...
            score_align_w,score_loss_threshold,score_loss_penalty_w,init_env_score);
        
        scores(b) = sc;
        scores_p1(b) = sc_p1;
        
        % 累积梯度 (梯度上升 → 取反变为最小化)
        gW1=gW1+dW1; gb1=gb1+db1_g;
        gW2=gW2+dW2; gb2=gb2+db2_g;
        gW3=gW3+dW3; gb3=gb3+db3_g;
        gW4=gW4+dW4; gb4=gb4+db4_g;
    end
    
    % 平均梯度
    gW1=gW1/batch_size; gb1=gb1/batch_size;
    gW2=gW2/batch_size; gb2=gb2/batch_size;
    gW3=gW3/batch_size; gb3=gb3/batch_size;
    gW4=gW4/batch_size; gb4=gb4/batch_size;
    
    % 打包梯度用于裁剪和 Adam
    g_vec = [gW1(:);gb1(:);gW2(:);gb2(:);gW3(:);gb3(:);gW4(:);gb4(:)]';
    gn = norm(g_vec);
    if gn > grad_clip, g_vec = g_vec * grad_clip / gn; end
    
    % Adam 更新 (梯度上升: w = w + lr * ...)
    t_ad = t_ad + 1;
    m_ad = b1_ad*m_ad + (1-b1_ad)*g_vec;
    v_ad = b2_ad*v_ad + (1-b2_ad)*g_vec.^2;
    mh = m_ad/(1-b1_ad^t_ad);
    vh = v_ad/(1-b2_ad^t_ad);
    delta = lr * mh ./ (sqrt(vh)+eps_ad);
    
    % 解包并更新权重
    idx = 0;
    n=numel(W1); W1=W1+reshape(delta(idx+1:idx+n),size(W1)); idx=idx+n;
    n=numel(b1); b1=b1+reshape(delta(idx+1:idx+n),size(b1)); idx=idx+n;
    n=numel(W2); W2=W2+reshape(delta(idx+1:idx+n),size(W2)); idx=idx+n;
    n=numel(b2); b2=b2+reshape(delta(idx+1:idx+n),size(b2)); idx=idx+n;
    n=numel(W3); W3=W3+reshape(delta(idx+1:idx+n),size(W3)); idx=idx+n;
    n=numel(b3); b3=b3+reshape(delta(idx+1:idx+n),size(b3)); idx=idx+n;
    n=numel(W4); W4=W4+reshape(delta(idx+1:idx+n),size(W4)); idx=idx+n;
    n=numel(b4); b4=b4+reshape(delta(idx+1:idx+n),size(b4)); idx=idx+n;
    
    % 统计
    em = mean(scores); eb = max(scores); em_p1 = mean(scores_p1);
    
    % ---- 真实环境验证 ----
    val_sc = NaN;
    if mod(ep, val_freq)==0 || ep==1 || ep==n_epochs
        val_sc = validate_env(W1,b1,W2,b2,W3,b3,W4,b4, Xm,Xs,Ym,Ys, al, ip_p,op_p,satLev);
        if val_sc > best_val
            best_val = val_sc;
            best_W1=W1;best_b1=b1;best_W2=W2;best_b2=b2;
            best_W3=W3;best_b3=b3;best_W4=W4;best_b4=b4;
        end
    end
    if eb > best_sc, best_sc = eb; end
    hist_data(ep,:) = [em, eb, em_p1, val_sc];
    
    if mod(ep,5)==0 || ep==1
        if isnan(val_sc)
            fprintf('Epoch %3d | P1=%.1f | Train: Mean=%.1f Best=%.1f | BestVal=%.1f\n', ...
                ep, em_p1, em, eb, best_val);
        else
            fprintf('Epoch %3d | P1=%.1f | Train: Mean=%.1f | Val=%.1f | BestVal=%.1f\n', ...
                ep, em_p1, em, val_sc, best_val);
        end
    end
end

%% ---- 保存 ----
W1=best_W1;b1=best_b1;W2=best_W2;b2=best_b2;
W3=best_W3;b3=best_b3;W4=best_W4;b4=best_b4;
X_mean=Xm; X_std=Xs; Y_mean=Ym; Y_std=Ys; alpha=al;
save('nn_weights_rl.mat','W1','b1','W2','b2','W3','b3','W4','b4',...
     'X_mean','X_std','Y_mean','Y_std','alpha');
fprintf('\n=== 完成, 最优验证分: %.2f (训练分: %.2f) ===\n', best_val, best_sc);
fprintf('权重: nn_weights_rl.mat\n');
fprintf('使用: copyfile(''nn_weights_rl.mat'', ''nn_weights.mat'')\n');
if exist('sys_rl.ini','file'), delete('sys_rl.ini'); end

%% ---- 绘图 ----
figure('Position',[100 100 1200 400]);
subplot(1,3,1);
plot(1:n_epochs, hist_data(:,1),'b-','LineWidth',1.5); hold on;
plot(1:n_epochs, hist_data(:,2),'r-','LineWidth',1);
xlabel('Epoch'); ylabel('Score'); title('训练 Score'); grid on;
legend('Mean','Best','Location','southeast');
subplot(1,3,2);
val_idx = find(~isnan(hist_data(:,4)));
if ~isempty(val_idx)
    plot(val_idx, hist_data(val_idx,4),'go-','LineWidth',2,'MarkerSize',6);
    yline(best_val,'g--',sprintf('BestVal=%.1f',best_val));
end
xlabel('Epoch'); ylabel('Score'); title('验证 Score (真实Env)'); grid on;
subplot(1,3,3);
plot(1:n_epochs, hist_data(:,3),'m-','LineWidth',1.5);
xlabel('Epoch'); ylabel('Score'); title('Phase1 Score'); grid on;

%% ========================================================================
%  单次 episode: 前向仿真 + 反向传播 (adjoint method)
%  返回: 环境得分, phase1 得分, 各层参数梯度
%% ========================================================================
function [env_score_final, score_p1, dW1,db1_g,dW2,db2_g,dW3,db3_g,dW4,db4_g] = ...
    run_episode(H0, W1,b1,W2,b2,W3,b3,W4,b4, Xm,Xs,Ym,Ys, ...
    al, ip_p,op_p,satLev,dt,T,sp1,sp2,t_switch, EPS_H, ...
    reach_thresh,reach_bonus,timeout_penalty,stability_w,steady_err_w, ...
    score_align_w,score_loss_threshold,score_loss_penalty_w,init_env_score)

    lrelu = @(z) max(al*z, z);
    
    % ===== 预分配 =====
    H    = zeros(T+1,1);  H(1) = H0;
    e_a  = zeros(T,1);    de_a = zeros(T,1);
    sp_a = zeros(T,1);    ueq_a= zeros(T,1);
    x_a  = zeros(T,4);
    z1_a = zeros(T,64);   a1_a = zeros(T,64);
    z2_a = zeros(T,32);   a2_a = zeros(T,32);
    z3_a = zeros(T,16);   a3_a = zeros(T,16);
    z4_a = zeros(T,1);
    u_raw_a = zeros(T,1); u_a  = zeros(T,1);
    clip_lo = false(T,1); clip_hi = false(T,1);
    
    % 每步 loss 及其对 e, de, u 的梯度
    dL_de_a  = zeros(T,1);
    dL_dde_a = zeros(T,1);
    dL_du_a  = zeros(T,1);
    
    % ===== 前向仿真 =====
    env_score = init_env_score;
    score_p1 = 0; captured_p1 = false;
    reached1 = false; reached2 = false;
    last_sp = -1;
    phase1_handled = false;
    
    total_loss = 0;  % 用于监控
    
    for t = 1:T
        t_now = t * dt;
        if t_now <= t_switch, sp = sp1; else, sp = sp2; end
        sp_a(t) = sp;
        
        % 误差
        e_a(t) = sp - H(t);
        if last_sp > 0 && abs(sp - last_sp) > 1
            de_a(t) = 0;
        elseif t > 1
            de_a(t) = e_a(t) - e_a(t-1);
        else
            de_a(t) = 0;
        end
        ueq_a(t) = op_p * sqrt(max(sp,0)) / ip_p;
        
        % NN 前向
        x = ([e_a(t), de_a(t), sp, ueq_a(t)] - Xm) ./ Xs;
        x_a(t,:) = x;
        z1=x*W1+b1;     a1=lrelu(z1);
        z2=a1*W2+b2;    a2=lrelu(z2);
        z3=a2*W3+b3;    a3=lrelu(z3);
        z4=a3*W4+b4;
        z1_a(t,:)=z1; a1_a(t,:)=a1;
        z2_a(t,:)=z2; a2_a(t,:)=a2;
        z3_a(t,:)=z3; a3_a(t,:)=a3;
        z4_a(t)=z4;
        
        u_raw = z4 * Ys + Ym;
        u_raw_a(t) = u_raw;
        u = max(0, min(satLev, u_raw));
        u_a(t) = u;
        clip_lo(t) = (u_raw < 0);
        clip_hi(t) = (u_raw > satLev);
        
        % ---- 每步 Loss 及解析梯度 ----
        e = e_a(t); de = de_a(t); ueq = ueq_a(t);
        
        % 基础: L_base = e² * dt + (ueq-u)² * dt
        L_t = (e^2 + (ueq - u)^2) * dt;
        dLt_de = 2*e*dt;
        dLt_dde = 0;
        dLt_du = -2*(ueq - u)*dt;
        
        % 到达检测 & 稳态惩罚
        if t_now <= t_switch
            if ~reached1 && abs(e) < reach_thresh, reached1 = true; end
            if reached1
                L_t = L_t + steady_err_w*e^2*dt + stability_w*de^2*dt;
                dLt_de = dLt_de + 2*steady_err_w*e*dt;
                dLt_dde = dLt_dde + 2*stability_w*de*dt;
            end
        else
            if ~phase1_handled
                phase1_handled = true;
                if ~reached1, L_t = L_t + timeout_penalty; end
            end
            if ~reached2 && abs(e) < reach_thresh, reached2 = true; end
            if reached2
                L_t = L_t + steady_err_w*e^2*dt + stability_w*de^2*dt;
                dLt_de = dLt_de + 2*steady_err_w*e*dt;
                dLt_dde = dLt_dde + 2*stability_w*de*dt;
            end
        end
        
        % 环境分数对齐: L_align = score_align_w * (e² + (ueq-u)²) * dt
        L_align = score_align_w * (e^2 + (ueq-u)^2) * dt;
        L_t = L_t + L_align;
        dLt_de = dLt_de + score_align_w * 2*e*dt;
        dLt_du = dLt_du + score_align_w * (-2*(ueq-u)*dt);
        
        % 环境分数跟踪
        env_score = env_score - (e^2 + (ueq-u)^2)*dt;
        
        total_loss = total_loss + L_t;
        dL_de_a(t) = dLt_de;
        dL_dde_a(t) = dLt_dde;
        dL_du_a(t) = dLt_du;
        
        % Phase1 分数捕获
        if ~captured_p1 && t_now >= t_switch
            score_p1 = env_score;
            captured_p1 = true;
        end
        
        % 动力学 (RK4, 匹配 ode45 精度)
        Hc = H(t);
        f_dyn = @(hh) dyn_rhs(hh, u, ip_p, op_p, EPS_H);
        k1 = dt * f_dyn(Hc);
        k2 = dt * f_dyn(Hc + k1/2);
        k3 = dt * f_dyn(Hc + k2/2);
        k4 = dt * f_dyn(Hc + k3);
        H(t+1) = Hc + (k1 + 2*k2 + 2*k3 + k4) / 6;
        
        last_sp = sp;
    end
    
    % 终端: 未到达 target2 惩罚
    if ~reached2, total_loss = total_loss + timeout_penalty; end
    
    % 终端: 总扣分惩罚
    score_loss = init_env_score - env_score;
    if score_loss > score_loss_threshold
        total_loss = total_loss + score_loss_penalty_w * (score_loss - score_loss_threshold);
    end
    
    env_score_final = env_score;
    
    % ===== 反向传播 (Adjoint Method) =====
    % lambda(t) = dL_total / dH(t)，从 T+1 反传到 1
    % 同时累积 NN 参数梯度
    
    dW1=zeros(size(W1)); db1_g=zeros(size(b1));
    dW2=zeros(size(W2)); db2_g=zeros(size(b2));
    dW3=zeros(size(W3)); db3_g=zeros(size(b3));
    dW4=zeros(size(W4)); db4_g=zeros(size(b4));
    
    lambda = 0;  % dL/dH_{T+1}
    de_carry = 0; % de coupling: 从 t+1 传回的 dL/de(t) 贡献
    
    for t = T:-1:1
        Hc = H(t);
        e = e_a(t);
        
        % 动力学 Jacobian (RK4 一阶近似, 足够准确)
        sH = sqrt(max(Hc, EPS_H));
        if Hc <= 0 || Hc >= 30
            dHnext_dH = 1;
            dHnext_du = 0;
        else
            dHnext_dH = 1 - dt*op_p/(2*sH);
            dHnext_du = dt*ip_p;
        end
        
        % 若 u 被裁剪, 梯度不通过
        is_clipped = clip_lo(t) || clip_hi(t);
        if is_clipped, dHnext_du = 0; end
        
        % Loss 对 u_t 的直接梯度
        dLoss_du = dL_du_a(t);
        if is_clipped, dLoss_du = 0; end  % 裁剪后 u 不受 NN 影响
        
        % Loss 对 e_t 的直接梯度 + de coupling from future
        dLoss_de = dL_de_a(t) + de_carry;
        
        % Loss 对 de_t 的直接梯度
        dLoss_dde = dL_dde_a(t);
        
        % 总的对 u_t 的梯度信号 (直接 + 通过动力学影响未来)
        signal_u = dLoss_du + lambda * dHnext_du;
        
        % ---- NN 参数梯度反传 (用 loss signal) ----
        dz4 = signal_u * Ys;
        
        dW4 = dW4 - a3_a(t,:)' * dz4;  % 负号: 最小化 loss
        db4_g = db4_g - dz4;
        da3 = dz4 * W4';
        
        dz3 = da3 .* ((z3_a(t,:)>0) + al*(z3_a(t,:)<=0));
        dW3 = dW3 - a2_a(t,:)' * dz3;
        db3_g = db3_g - dz3;
        da2 = dz3 * W3';
        
        dz2 = da2 .* ((z2_a(t,:)>0) + al*(z2_a(t,:)<=0));
        dW2 = dW2 - a1_a(t,:)' * dz2;
        db2_g = db2_g - dz2;
        da1 = dz2 * W2';
        
        dz1 = da1 .* ((z1_a(t,:)>0) + al*(z1_a(t,:)<=0));
        dW1 = dW1 - x_a(t,:)' * dz1;
        db1_g = db1_g - dz1;
        
        % ---- 独立 NN Jacobian: ∂u_raw/∂x (不含 loss signal) ----
        % 用于 adjoint 传播, 必须和 loss backprop 分开计算
        jac_z4 = Ys;  % ∂u_raw/∂z4 = Ys
        jac_a3 = jac_z4 * W4';
        jac_z3 = jac_a3 .* ((z3_a(t,:)>0) + al*(z3_a(t,:)<=0));
        jac_a2 = jac_z3 * W3';
        jac_z2 = jac_a2 .* ((z2_a(t,:)>0) + al*(z2_a(t,:)<=0));
        jac_a1 = jac_z2 * W2';
        jac_z1 = jac_a1 .* ((z1_a(t,:)>0) + al*(z1_a(t,:)<=0));
        jac_x  = jac_z1 * W1';  % 1x4: ∂u_raw/∂x
        
        % ∂u_raw/∂H = ∂u_raw/∂x * ∂x/∂H
        % x(1) = (e-Xm1)/Xs1, ∂x1/∂H = -1/Xs1 (e=sp-H)
        % x(2) = (de-Xm2)/Xs2, ∂x2/∂H = -1/Xs2 (de=e(t)-e(t-1))
        du_dH = -(jac_x(1)/Xs(1) + jac_x(2)/Xs(2));
        if is_clipped, du_dH = 0; end
        
        % 更新 adjoint: λ(t) = ∂L_t/∂H_t + λ(t+1)*(∂H_{t+1}/∂H_t + ∂H_{t+1}/∂u * ∂u/∂H)
        dLoss_dH_direct = -dLoss_de;  % ∂e/∂H = -1
        lambda = dLoss_dH_direct + lambda * (dHnext_dH + dHnext_du * du_dH);
        
        % de coupling: de(t)=e(t)-e(t-1), ∂de(t)/∂e(t-1)=-1
        % ∂loss_t/∂e(t-1) via de = -dLoss_dde
        de_carry = -dLoss_dde;
        % NN 通过 de 对 e(t-1) 的影响: ∂u_raw/∂e(t-1) = jac_x(2)/Xs(2)*(-1)
        if ~is_clipped
            de_carry = de_carry - lambda * dHnext_du * jac_x(2)/Xs(2);
        end
    end
    
    % 梯度归一化 (按步数)
    dW1=dW1/T; db1_g=db1_g/T;
    dW2=dW2/T; db2_g=db2_g/T;
    dW3=dW3/T; db3_g=db3_g/T;
    dW4=dW4/T; db4_g=db4_g/T;
end

%% ---- 动力学右端 (RK4 用) ----
function dH = dyn_rhs(hh, u, ip_p, op_p, EPS_H)
    if hh <= 0
        dH = 0.5;
    elseif hh >= 30
        dH = -0.01;
    else
        dH = ip_p * u - op_p * sqrt(max(hh, EPS_H));
    end
end

%% ---- 真实环境验证 (用 Env + ode45, 和 main_nn 一致) ----
function score = validate_env(W1,b1,W2,b2,W3,b3,W4,b4, Xm,Xs,Ym,Ys, al, ip_p,op_p,satLev)
    lr_fn = @(x) max(al*x, x);
    
    figs0 = findall(0,'Type','figure');
    try
        env = Env('sys_rl.ini');
        if ~env.succeed, score = -99999; return; end
        obs = env.reset();
    catch
        score = -99999; return;
    end
    
    le = 0; lsp = -1;
    while true
        H = obs.agent.H; sp = obs.targetHeight; e = sp - H;
        if lsp > 0 && abs(sp-lsp) > 1, de = 0;
        elseif le ~= 0, de = e - le;
        else, de = 0; end
        
        ueq = op_p*sqrt(max(sp,0))/ip_p;
        x = ([e, de, sp, ueq] - Xm) ./ Xs;
        a1 = lr_fn(x*W1+b1);
        a2 = lr_fn(a1*W2+b2);
        a3 = lr_fn(a2*W3+b3);
        u_raw = (a3*W4+b4)*Ys + Ym;
        u = max(0, min(satLev, u_raw));
        
        le = e; lsp = sp;
        [obs, done, ~] = env.step(u);
        if done, break; end
    end
    score = obs.score;
    
    figs1 = findall(0,'Type','figure');
    nf = setdiff(figs1, figs0);
    if ~isempty(nf), close(nf); end
end
