classdef Policy < handle
    properties
        ff_gain         % 前馈控制增益
        last_e          % 上一次的误差，用于算微分
        last_sp         % 上一次的目标，用于防突变冲击
        initialized     % 初始化状态
    end

    properties(Access = private)
        % 系统配置参数存储
        start_config
        map_config
        agent_config
        sys_config
    end

    methods
        function obj = Policy()
            % 1. 加载配置并计算前馈
            obj.ff_gain = 1.0; 
            if exist('sys.ini', 'file')
                try 
                    ini_parser = INI('File', 'sys.ini');
                    ini_parser.read();
                    agent_cfg = ini_parser.get('Agent');
                    if isfield(agent_cfg, 'op') && isfield(agent_cfg, 'ip')
                        obj.ff_gain = agent_cfg.op / agent_cfg.ip;
                    end
                catch
                end
            end
            
            % 2. 状态初始化
            obj.last_e = 0;
            obj.last_sp = -1;
            obj.initialized = false;
        end
        
        function act_val = action(obj, observation)
            % 读取当前状态
            H = observation.agent.H;
            sp = observation.targetHeight;
            
            % 屏蔽目标突变造成的伪导数激增
            if obj.last_sp > 0 && abs(sp - obj.last_sp) > 1
                obj.last_e = sp - H;
            end
            
            e = sp - H;
            
            if ~obj.initialized
                de = 0;
                obj.initialized = true;
            else
                de = e - obj.last_e;
            end
            
            % ==============================================================
            % 基于目标水位参考轨迹的增益调度 (Gain Scheduling):
            % 非线性一阶水箱在大排量和低水位时具有不同的物理表现。
            % 通过当前设定值(Setpoint)分离两种模糊策略，以实现在高低位工况的不同特征控制。
            % ==============================================================
            if sp > 15
                % --------------------------------------------------
                % 工况 A (高水位满负荷爬升): 采用二维模糊 PD 矩阵
                % 特性：加入微分预测项，提供大跨度突变时的上升段平滑响应。
                % --------------------------------------------------
                e_range = 25;
                de_range = 2.0;
                du_range = 50;

                e_n  = max(-6, min(6, e  / e_range  * 6));
                de_n = max(-6, min(6, de / de_range * 6));

                e_mf  = Policy.fuzzify7(e_n);
                de_mf = Policy.fuzzify7(de_n);

                R = [
                     1,  1,  1,  1,  1,  2,  2;
                     1,  1,  2,  2,  2,  3,  3;
                     2,  2,  3,  3,  3,  4,  4;
                     3,  3,  3,  4,  5,  5,  5;
                     4,  4,  5,  5,  5,  6,  6;
                     5,  5,  6,  6,  6,  7,  7;
                     6,  6,  7,  7,  7,  7,  7;
                ];

                du_mf = zeros(1, 7);
                for i = 1:7
                    for j = 1:7
                        w = min(e_mf(i), de_mf(j));
                        if w > 0
                            k = R(i, j);
                            if w > du_mf(k)
                                du_mf(k) = w;
                            end
                        end
                    end
                end

                centers = [-6, -4, -2, 0, 2, 4, 6];
                sw = sum(du_mf);
                if sw < 1e-10
                    du_n = 0;
                else
                    du_n = sum(du_mf .* centers) / sw;
                end

                du = du_n / 6 * du_range;

                % 根据伯努利原理，计算前馈稳态输出值以抵消底部泄露
                ip = 2.5; op = 2.5;
                u_eq = op * sqrt(max(sp, 0)) / ip;
                
                u_total = u_eq + du;
                
            else
                % --------------------------------------------------
                % 工况 B (低水位阶跃下降): 采用一维标准自编模糊推断
                % 特点：取消微分项干扰以防止提前过载刹车，依赖严格的正向 P 输出进行极速放水。
                % --------------------------------------------------
                
                % 针对低水位的极速放水需求，采用紧凑型输入范围的一维模糊查表
                e_range_2 = 1.2;
                du_range_2 = 30.0;
                
                e_n = max(-6, min(6, e / e_range_2 * 6));
                e_mf = Policy.fuzzify7(e_n);
                
                % 一维线性对应：提取基础论域中心点直接比例加权，不再交叉遍历二维矩阵
                centers = [-6, -4, -2, 0, 2, 4, 6];
                
                sw = sum(e_mf);
                if sw < 1e-10
                    du_n = 0;
                else
                    du_n = sum(e_mf .* centers) / sw;
                end
                
                du = du_n / 6 * du_range_2;
                
                % 进行理论平衡点的前馈增益融合
                u_feedforward = obj.ff_gain * sqrt(max(sp, 0));
                
                u_total = du + u_feedforward;
            end
            
            % 保留状态
            obj.last_e = e;
            obj.last_sp = sp;
            
            % 返回动作并保底截断低于 0 的无效解
            act_val = [max(0, u_total)];
        end
    end

    methods (Static)
        function mf = fuzzify7(x)
            mf = zeros(1, 7);
            c  = [-6, -4, -2, 0, 2, 4, 6];
            for i = 1:7
                if i == 1
                    if x <= c(1),        mf(1) = 1;
                    elseif x < c(1) + 2, mf(1) = (c(1) + 2 - x) / 2;
                    end
                elseif i == 7
                    if x >= c(7),        mf(7) = 1;
                    elseif x > c(7) - 2, mf(7) = (x - c(7) + 2) / 2;
                    end
                else
                    d = abs(x - c(i));
                    if d < 2, mf(i) = 1 - d / 2; end
                end
            end
        end
    end
end