classdef Policy_NN < handle
    % Policy_NN  纯神经网络水箱液位控制策略（两步法）
    %   第 1 步 (离线): 用 generate_train_data.m + train_nn.m 完成模仿学习
    %   第 2 步 (离线): 用 finetune_nn.m 进化策略优化
    %   部署时: 加载 nn_weights.mat，纯 NN 前向推理输出控制量
    %
    %   网络结构: 4 -> 64 -> 32 -> 16 -> 1 (Leaky ReLU)
    %   输入特征: [e, de, sp, u_eq]
    %     e    = targetHeight - H (误差)
    %     de   = e(t) - e(t-1)    (误差变化)
    %     sp   = targetHeight     (目标水位)
    %     u_eq = op*sqrt(sp)/ip   (平衡点参考)
    %   输出: 阀门控制量 (经限幅后)

    properties
        time            % 当前采样次数
        total_output    % 总输出（控制量）
        dt              % 采样时间间隔
        total_time      % 总仿真时间
        simulate_amount % 采样总次数
        error           % 误差数列
        ip              % 进水管系数
        op              % 出水管系数
        last_sp         % 上一步的目标水位（检测阶跃变化）
    end

    properties(Access = private)
        AgentInfo       % Agent 配置信息
        SystemInfo      % System 配置信息
        % 神经网络权重（手写 4 层全连接）
        W1; b1; W2; b2; W3; b3; W4; b4
        % 归一化参数
        X_mean; X_std; Y_mean; Y_std
        % Leaky ReLU 参数
        alpha
    end

    methods
        function self = Policy_NN()
            % 检查是否存在 sys.ini 文件
            if exist('sys.ini', 'file')
                self.load_ini_file('sys.ini');
            else
                return;
            end

            % 读取物理参数
            self.ip = self.AgentInfo.ip;
            self.op = self.AgentInfo.op;

            % 时间相关
            self.time = 1;
            self.dt = self.SystemInfo.st;
            self.total_time = self.SystemInfo.tend;
            self.simulate_amount = self.total_time / self.dt + 2;
            self.error = zeros(1, self.simulate_amount);
            self.last_sp = -1;

            % ============ 加载预训练神经网络权重 ============
            if ~exist('nn_weights.mat', 'file')
                error('找不到 nn_weights.mat，请先运行 generate_train_data.m 和 train_nn.m');
            end
            data = load('nn_weights.mat');
            self.W1 = data.W1;  self.b1 = data.b1;
            self.W2 = data.W2;  self.b2 = data.b2;
            self.W3 = data.W3;  self.b3 = data.b3;
            self.W4 = data.W4;  self.b4 = data.b4;
            self.X_mean = data.X_mean;
            self.X_std  = data.X_std;
            self.Y_mean = data.Y_mean;
            self.Y_std  = data.Y_std;
            self.alpha  = data.alpha;

            fprintf('Policy_NN: 已加载预训练权重，纯 NN 控制模式\n');
        end

        function action = action(self, observation)
            sys = observation.agent;
            sp = sys.targetHeight;
            H  = sys.H;
            disp('预设水位:' + string(sp) + ', 当前水位:' + string(H));

            % ====== 第 1 步：计算误差 ======
            self.error(self.time) = sp - H;
            e = self.error(self.time);

            % ====== 第 2 步：计算误差变化量 ======
            % 检测目标水位阶跃变化：重置 de 避免误差跳变
            if self.last_sp > 0 && abs(sp - self.last_sp) > 1
                de = 0;  % 目标突变时不计算微分
            elseif self.time > 1
                de = self.error(self.time) - self.error(self.time - 1);
            else
                de = 0;
            end

            % ====== 第 3 步：构造 NN 输入特征 ======
            u_eq = self.op * sqrt(max(sp, 0)) / self.ip;  % 平衡点参考值
            features = [e, de, sp, u_eq];

            % ====== 第 4 步：NN 前向推理 ======
            % 输入归一化
            x = (features - self.X_mean) ./ self.X_std;

            % 4 层全连接 + Leaky ReLU
            a1 = self.lrelu(x * self.W1 + self.b1);
            a2 = self.lrelu(a1 * self.W2 + self.b2);
            a3 = self.lrelu(a2 * self.W3 + self.b3);
            z4 = a3 * self.W4 + self.b4;

            % 输出反归一化
            self.total_output = z4 * self.Y_std + self.Y_mean;

            % ====== 第 5 步：输出限幅 ======
            if (self.total_output > sys.satLevel)
                self.total_output = sys.satLevel;
            elseif (self.total_output < 0)
                self.total_output = 0;
            end

            action = [self.total_output];

            % 更新状态
            self.last_sp = sp;
            self.time = self.time + 1;

            % 仿真结束时展示误差曲线
            if (self.simulate_amount == self.time)
                figure;
                plot(self.error);
                title('误差曲线 (纯 NN 控制)');
                xlabel('采样次数');
                ylabel('误差');
            end
        end
    end

    methods(Access = private)
        function load_ini_file(self, filename)
            % 加载 INI 配置文件，解析 Agent 和 System 节
            ini_parser = INI('File', filename);
            ini_parser.read();
            self.AgentInfo = ini_parser.get('Agent');
            self.SystemInfo = ini_parser.get('System');
        end

        function y = lrelu(self, x)
            % Leaky ReLU 激活函数
            y = max(self.alpha * x, x);
        end
    end
end
