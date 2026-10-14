classdef Policy_AStar_PD_Optimized < handle
    properties
        dt              %仿真分辨率
        dx
        dy
        ax_hist
        ay_hist
        speed           %当前速度
        ang_speed       %当前角速度
        new_ang_speed
        new_v_speed
        vm              %用于存储最大最小速度和最大最小的角速度
        map
        mapw            %最大搜索宽度
        maph            %最大搜索高度
        localmap        %本地存储地图
        storemap 
        obstoremap
        temp_observationmap
        temp_obmap
        now_smartmap
        photo
        special

        obmap
        time            %小车运行过程中计时
        path
        twist_set
        path_history
        twist_history
        openlist
        closelist
        findflag
        end_y_bias
        start_pos
        goal_pos
        turn_flag_hist
        replan_interval
        last_astar_failed

        pd_flag
        Kp
        Kd
        error
        last_error
        a_speed_hist
        positions

        StartPos
        EndPos
        MapInfo
        RangeFinderInfo
        AgentInfo
        SystemInfo
    end

    methods        
        function self = Policy_AStar_PD_Optimized()
            if exist('sys.ini','file')
                self.load_ini_file('sys.ini');
            else
                return;
            end
            %定义地图极限
            self.mapw = self.MapInfo.w-2;
            self.maph = self.MapInfo.h-2;
            %定义一些记录数组变量
            self.ax_hist = [];
            self.ay_hist = [];
            self.path_history = [];
            self.twist_history = [];
            self.obmap = zeros(50,50);
            self.positions = [];
            self.time = 0;
            self.new_ang_speed = 0;
            %定义速度与角速度
            self.speed = 0;
            self.ang_speed = 0;
            %定义仿真时间
            self.dt = self.SystemInfo.st;
            %定义速度极限数组
            self.vm = [-self.AgentInfo.usat,self.AgentInfo.usat,-self.AgentInfo.vsat,self.AgentInfo.vsat];
            self.path = [];
            self.twist_set = [];
            self.turn_flag_hist = [];
            self.replan_interval = 6;
            self.last_astar_failed = false;
            %定义PD控制的相关变量
            self.pd_flag = 0;
            self.error = 0;
            self.last_error = 0;
            self.Kp = 650;
            self.Kd = 1000;
            self.special = 'b2map';
        end
        
        function action=action(self,observation)
            %当全局地图可见且没有蓝色智能小车时
            if self.SystemInfo.globalview == 1 && self.SystemInfo.smartAgent == 0
                self.localmap = observation.map;
                self.time = self.time + 1; 
                %实现地图膨胀
                self.expand_map_allknown(observation);     
                %记录小车位置方便调试时查看
                self.ax_hist = [self.ax_hist,observation.agent.x];
                self.ay_hist = [self.ay_hist,observation.agent.y];
                use_cached_path = strcmp(self.MapInfo.name, 'c1map');
                if ~use_cached_path || self.time == 1 || isempty(self.path) || size(self.path, 1) < 3 || mod(self.time, self.replan_interval) == 1
                    % c1map is expensive enough to benefit from sparse replanning.
                    self.path = A_star(self,observation);
                    add_bias(self);
                else
                    self.path = self.prune_path_to_agent(self.path, observation);
                end
                self.find_twist(self.path);
                %检查是否进入PD控制模式
                if ~isempty(self.twist_set)
                    len_twist = length(self.twist_set(:,1));                
                    if sqrt((observation.agent.x - self.twist_set(len_twist,1))^2 + (observation.agent.y - self.twist_set(len_twist,2))^2) < 1.5
                        self.pd_flag = 1;
                    end
                end
                %判断该执行的控制模式
                if self.pd_flag == 1
                    u = pd_move(self,observation,self.path);
                elseif self.pd_flag == 0
                    u = easy_move(self,observation,self.path);
                end
                %进行碰撞检测并作出调整
                self.predict_collide(observation,u(2),u(1));
                %更新碰撞检测函数执行后的结果为最后输出
                new_u = [u(1),self.new_ang_speed];
                action = new_u;
            %局部地图且不开智能小车
            elseif self.SystemInfo.globalview == 0 && self.SystemInfo.smartAgent == 0
                if self.time == 0
                    observation.map = zeros(50,50);
                    self.obmap = observation.map;
                else
                    observation.map = self.localmap;
                end
                observation.map = observation.map + observation.scanMap;
                self.obmap = self.obmap + observation.scanMap;
                %统一将大于1的值赋成1
                for row = 1:1:50
                    for col = 1:1:50
                        if observation.map(row,col) ~= 0
                            observation.map(row,col) = 1;
                        end
                    end
                end
                self.localmap = observation.map;
                self.time = self.time + 1;
                %膨胀局部地图
                self.expand_map_part(observation,1);
                self.ax_hist = [self.ax_hist,observation.agent.x];
                self.ay_hist = [self.ay_hist,observation.agent.y];
                self.path = A_star(self,observation);
                add_bias(self);
                self.find_twist(self.path);
                self.path_history = [self.path_history;self.path];
                self.twist_history = [self.twist_history;self.twist_set];
                % Keep training/evaluation headless and fast.
                if ~isempty(self.twist_set)
                    len_twist = length(self.twist_set(:,1));                
                    if sqrt((observation.agent.x - self.twist_set(len_twist,1))^2 + (observation.agent.y - self.twist_set(len_twist,2))^2) < 1.5
                        self.pd_flag = 1;
                    end
                end
                if self.pd_flag == 1
                    u = pd_move(self,observation,self.path);
                elseif self.pd_flag == 0
                    u = easy_move(self,observation,self.path);
                end
                self.predict_collide(observation,u(2),u(1));
                new_u = [u(1),self.new_ang_speed];
                action = new_u;
            %全局地图且开智能小车
            elseif self.SystemInfo.globalview == 1 && self.SystemInfo.smartAgent ~= 0
                self.localmap = observation.map;
                self.time = self.time + 1;
                is_car_front_matrix = observation.scanMap > 1;
                is_car_front_flag = any(is_car_front_matrix(:));
                % 扫描范围内有小车
                if is_car_front_flag == 1
                    self.localmap = self.localmap + observation.scanMap;
                    for i = 1:50
                        for j = 1:50
                            if self.localmap(i,j) >= 1
                                self.localmap(i,j) = 1;
                            end
                        end
                    end
                % 扫描范围内没有小车
                elseif is_car_front_flag == 0
                    self.localmap = observation.map;
                end
                self.localmap = observation.map;
                self.now_smartmap = self.localmap;
                self.expand_map_allknown(observation);       
                self.ax_hist = [self.ax_hist,observation.agent.x];
                self.ay_hist = [self.ay_hist,observation.agent.y];
                self.path = A_star(self,observation);
                if self.last_astar_failed
                    self.localmap = observation.map;
                    self.now_smartmap = self.localmap;
                    self.expand_map_allknown(observation);
                    self.path = A_star(self,observation);
                end
                add_bias(self);
                self.find_twist(self.path);
                self.path_history = [self.path_history;self.path];
                self.twist_history = [self.twist_history;self.twist_set];
                % Keep training/evaluation headless and fast.
                if ~isempty(self.twist_set)
                    len_twist = length(self.twist_set(:,1));                
                    if sqrt((observation.agent.x - self.twist_set(len_twist,1))^2 + (observation.agent.y - self.twist_set(len_twist,2))^2) < 1.5
                        self.pd_flag = 1;
                    end
                end
                u = easy_move(self,observation,self.path);
                if self.pd_flag == 1
                    u = pd_move(self,observation,self.path);
                elseif self.pd_flag == 0
                    u = easy_move(self,observation,self.path);
                end
                self.predict_collide(observation,u(2),u(1));
                new_u = [u(1),self.new_ang_speed];
                new_u = self.dynamic_safety_adjust(observation, new_u);
                action = new_u;   
            %局部地图且开智能小车
            elseif self.SystemInfo.globalview == 0 && self.SystemInfo.smartAgent ~= 0
                if self.time == 0
                    observation.map = zeros(50,50);
                    self.obmap = observation.map;
                else
                    observation.map = self.localmap;
                end
                %筛选挑出不是小车的障碍物部分
                map_wall = observation.scanMap == 1;
                self.temp_observationmap = observation.map;
                observation.map = observation.map + map_wall;
                for row = 1:1:50
                    for col = 1:1:50
                        if observation.map(row,col) ~= 0
                            observation.map(row,col) = 1;
                        end
                    end
                end  
                self.storemap = observation.map;
                self.expand_map_part(observation,0);
                observation.map = self.temp_observationmap;
                observation.map = observation.map + observation.scanMap;
                self.temp_obmap = self.obmap + map_wall;
                self.obmap = self.obmap + observation.scanMap;
                for row = 1:1:50
                    for col = 1:1:50
                        if observation.map(row,col) ~= 0
                            observation.map(row,col) = 1;
                        end
                    end
                end
                self.localmap = observation.map;
                self.time = self.time + 1;
                self.expand_map_part(observation,1);
                self.ax_hist = [self.ax_hist,observation.agent.x];
                self.ay_hist = [self.ay_hist,observation.agent.y];
                self.path = A_star(self,observation);
                add_bias(self);
                self.find_twist(self.path);
                self.path_history = [self.path_history;self.path];
                self.twist_history = [self.twist_history;self.twist_set];
                % Keep training/evaluation headless and fast.
                if ~isempty(self.twist_set)
                    len_twist = length(self.twist_set(:,1));                
                    if sqrt((observation.agent.x - self.twist_set(len_twist,1))^2 + (observation.agent.y - self.twist_set(len_twist,2))^2) < 1.5
                        self.pd_flag = 1;
                    end
                end
                if self.pd_flag == 1
                    u = pd_move(self,observation,self.path);
                elseif self.pd_flag == 0
                    u = easy_move(self,observation,self.path);
                end
                self.predict_collide(observation,u(2),u(1));
                new_u = [u(1),self.new_ang_speed];
                self.obmap = self.temp_obmap;
                self.localmap = self.storemap;
                action = new_u;               
            end
        end

        function path = A_star(self, observation)
            %定义self.openlist列表
            self.openlist = [];
            %定义self.closelist列表
            self.closelist = [];
            %定义路径
            self.path = [];
            %findflag用于检测while循环是否结束
            self.findflag = false;
            self.end_y_bias = 1;
            start_pos_x = floor(observation.agent.x);
            start_pos_y = floor(observation.agent.y);
            goal_pos_x = floor(observation.endPos.x);
            goal_pos_y = floor(observation.endPos.y - self.end_y_bias);
            %定义A*算法的起始点和目标点
            self.start_pos = [start_pos_x start_pos_y];
            self.goal_pos = [goal_pos_x goal_pos_y];
            %定义A*函数使用的启发函数
            h_cost = 10 * abs(self.start_pos(1) - self.goal_pos(1)) + 10 * abs(self.start_pos(2) - self.goal_pos(2));
            %self.openlist变量每一行[节点坐标，代价值F=G+H,代价值G,父节点坐标]
            self.openlist = [self.start_pos(1), self.start_pos(2), 0 + h_cost, 0, self.goal_pos(1), self.goal_pos(2)];
            next_pos = [-1,1,14;...
                        0,1,10;...
                        1,1,14;...
                        -1,0,10;...
                        1,0,10;...
                        -1,-1,14;...
                        0,-1,10;...
                        1,-1,14];
            while ~self.findflag
                if isempty(self.openlist(:,1))
                    self.last_astar_failed = true;
                    path = [self.goal_pos; self.start_pos];
                    return;
                end
                %判断目标点是否出现在open列表中
                [isopenFlag,Id]= is_open(self, self.goal_pos, self.openlist);
                if isopenFlag
                    new_close_node = self.openlist(Id,:);
                    self.closelist = [new_close_node;self.closelist];
                    self.findflag = true;
                    break;
                end
                %------------------a.按照self.openlist中的第三列（代价函数F）进行排序，查找F值最小的节点
                [~,I] = sort(self.openlist(:,3));
                self.openlist = self.openlist(I,:);
                %------------------b.将F值最小的节点(即open中第一行节点)，放到close第一行(close是不断积压的)，作为当前节点
                increase_closenode = self.openlist(1,:);
                self.closelist = [increase_closenode;self.closelist];
                currentpos = self.openlist(1,:);
                self.openlist(1,:) = [];
                %--------------------c.对当前节点周围的8个相邻节点，算法的主体：
                %遍历检索当前节点8个方向的相邻节点并计算代价值
                for in=1:length(next_pos(:,1))
                    next_pos_dx = next_pos(in,1);
                    next_pos_dy = next_pos(in,2);
                    next_pos_cost = next_pos(in,3);
                    temp_x = currentpos(1,1)+next_pos_dx;
                    temp_y = currentpos(1,2)+next_pos_dy;
                    %获得相邻节点的坐标,代价值F先等于0,代价值G先等于0  ,后面两个值是其父节点的坐标值，暂定为零(因为暂时还无法判断其父节点坐标是多少)
                    m=[temp_x, temp_y, 0 , 0 , 0 ,0];
                    %计算代价值G
                    m(4)=currentpos(1,4)+next_pos_cost;
                    %利用曼哈顿距离计算h值
                    h_cost = 10 * abs(m(1) - self.goal_pos(1)) + 10 * abs(m(2) - self.goal_pos(2));
                    %根据F = G + H,计算出F值
                    m(3) = m(4) + h_cost;                   
                    %当超过地图范围时跳出当前循环
                    if temp_x < 1 || temp_y < 1 || temp_x > self.mapw || temp_y > self.maph
                        continue
                    end
                    %当遇到障碍物或者其余移动蓝色小车时跳过当前循环
                    if self.localmap(m(1), m(2)) == 1 || self.localmap(m(1), m(2)) == 2
                        continue
                    end
                    [flag,target_Index] = FindList(self,m,self.openlist,self.closelist);
                    in_close_flag = 1;
                    noin_open_flag = 2;
                    if flag==in_close_flag
                        continue;
                    elseif flag == noin_open_flag
                        m(5:6)=[currentpos(1,1),currentpos(1,2)];%将当前节点作为其父节点
                        self.openlist = [self.openlist;m];%把节点扩充到openlist中
                    else
                        if m(3) < self.openlist(target_Index,3)
                            m(5:6)=[currentpos(1,1),currentpos(1,2)];
                            self.openlist(target_Index,:) = m;
                        end
                    end
                end
            end
            path = GetPath(self, self.closelist, self.start_pos);  
            self.last_astar_failed = false;
        end

        function path=GetPath(~,closelist,start_pos)    
            index=1;
            path=[];
            while 1
                path=[path; closelist(index,1:2)];
                if isequal(closelist(index,1:2),start_pos)
                    break;
                end
                closelist_len = length(closelist(:,1));
                for o_index=1:closelist_len
                    if isequal(closelist(o_index,1:2),closelist(index,5:6))
                        index=o_index;
                        break;
                    end
                end
            end
        end

        function [flag,target_index]=FindList(~,m,openlist,closelist)       
            if  isempty(openlist)
                flag = 2;
                target_index = [];                
            else  %open不为空时，需要检查是否在self.openlist中
                %遍历self.openlist，检查是否在self.openlist中
                openlist_len = length(openlist(:,1));
                for o_index = 1:openlist_len
                    if isequal(m(1:2) , openlist(o_index,1:2))  %在self.openlist中
                        flag = 3;
                        target_index = o_index;
                        return;
                    else  %不在self.openlist中
                        flag = 2;
                        target_index = [];
                    end
                end
            end
            %如果能到这一步，说明:一定不在self.openlist中那么需要判断是否在self.closelist中   
            %遍历self.closelist（注意self.closelist不可能为空）
            closelist_len = length(closelist(:,1));
            for c_index = 1:closelist_len
                if isequal(m(1:2), closelist(c_index,1:2)  )  %在self.closelist中
                    flag = 1;
                    target_index = c_index;
                    return;%在self.closelist中直接return
                end
            end
        end
        function draw_path(self, path)
            %每次更新画图
            delete(self.photo);
            self.photo = plot(path(:,1),path(:,2),'r');
        end

        %判断节点是否在openlist列表中，在openlist中，isopenFlag = 1,不在open中，isopenFlag = 0 .并反回索引号
        function [isopenFlag,Id] = is_open(~,node,openlist)                            
            isopenFlag = false;
            Id = 0;            
            %如果openlist列表为空，则不在openlist列表中
            if  isempty(openlist)
                isopenFlag = false;    
            %openlist列表不为空时
            else 
                openlist_len = length(openlist(:,1));
                for i = 1:openlist_len
                    if isequal(node(1:2) , openlist(i,1:2)) 
                        isopenFlag = true;
                        Id = i;
                        return;
                    end
                end
            end           
        end

        %全局地图膨胀函数
        function expand_map_allknown(self,observation)
            expand_kernel = ones(3,3);
            Start = 2;
            End = 49;
            for i = Start:1:End
                for j = Start:1:End
                    if self.SystemInfo.smartAgent == 0
                        if observation.map(i+1,j+1) == 1
                            self.localmap(i-1:i+1,j-1:j+1) = expand_kernel;
                        end
                    elseif self.SystemInfo.smartAgent ~= 0
                        if self.now_smartmap(i+1,j+1) == 1
                            self.localmap(i-1:i+1,j-1:j+1) = expand_kernel;
                        end
                    end
                end
            end
        end

        %局部地图膨胀函数
        function expand_map_part(self,observation,flag)
            expand_kernel = ones(3,3);
            search_startx = floor(observation.agent.x - 8);
            search_endx = ceil(observation.agent.x + 8);
            search_starty = floor(observation.agent.y - 8);
            search_endy = ceil(observation.agent.y + 8);   
            if search_startx < 2
                search_startx = 2;
            elseif search_endx > self.mapw
                search_endx = self.mapw;
            end
            if search_starty < 2
                search_starty = 2;
            elseif search_endy > self.maph
                search_endy = self.maph;
            end
            for i = search_startx:1:search_endx
                for j = search_starty:1:search_endy
                    if observation.scanMap(i+1,j+1) ~= 0 && flag == 1
                        self.localmap(i-1:i+1,j-1:j+1) = expand_kernel;
                    elseif observation.scanMap(i+1,j+1) == 1 && flag == 0
                        self.storemap(i-1:i+1,j-1:j+1) = expand_kernel;
                    end
                end
            end
        end
        
        %非PD正常控制模式函数
        function u = easy_move(self,observation,path)
            len_path = length(path(:,1));
            wanted_ang = atan2((path(len_path-1,2) - path(len_path,2)) , (path(len_path-1,1) - path(len_path,1)));
            angle_error = self.wrap_to_pi(wanted_ang - observation.agent.h);
            if abs(angle_error) > 0.035
                a_speed = angle_error / 0.3;
                if abs(angle_error) > 0.75
                    v_speed = 0.18;
                elseif abs(angle_error) > 0.35
                    v_speed = 0.35;
                else
                    v_speed = 0.55;
                end
            else
                a_speed = 0;
                v_speed = 1;
            end
            a_speed = max(min(a_speed, self.AgentInfo.vsat), -self.AgentInfo.vsat);
            u = [v_speed,a_speed];
        end
        
        %PD控制模式函数
        function u = pd_move(self,observation,path)
            len_path = length(path(:,1));
            if len_path < 3
                u = easy_move(self,observation,path);
                return;
            end
            theta = atan2(path(len_path - 2,2) - observation.agent.y, path(len_path - 2,1) - observation.agent.x);
            delta_theta = self.wrap_to_pi(observation.agent.h - theta);
            self.error = sin(delta_theta) * sqrt((path(len_path - 2,2) - observation.agent.y)^2 + (path(len_path - 2,1) - observation.agent.x)^2);
            pd_out = self.Kp * self.error + self.Kd * (self.error - self.last_error);
            self.last_error = self.error;
            a_speed = - pd_out/100;
            a_speed = max(min(a_speed, self.AgentInfo.vsat), -self.AgentInfo.vsat);
            v_speed = max(0.55, 1 - 0.18 * abs(a_speed));
            u = [v_speed,a_speed];
            if abs(delta_theta) < 0.1
                self.pd_flag = 0;
            end
            self.a_speed_hist = [self.a_speed_hist,a_speed];
        end
            
        %添加偏置修正路径函数
        function [] = add_bias(self)
            len_path = length(self.path(:,1));
            bias_offset = 0.5;
            self.positions = zeros(len_path,4);
            start_index1 = 1;
            end_index1 = 1;
            start_index2 = 1;
            end_index2 = 1;
            start_index3 = 1;
            end_index3 = 1;
            start_index4 = 1;
            end_index4 = 1;           
            for i = 1:1:len_path
                self.positions(i,1:2) = self.path(i,1:2);
            end
            for i = 1:1:len_path
                x = self.positions(i,1);
                y = self.positions(i,2);  
                if x > 1 && y > 1
                    if self.localmap(x-1,y) == 1 
                        self.positions(i,3) = 1;
                    end  
                    if self.localmap(x,y-1) == 1
                        self.positions(i,3) = 2;
                    end
                    if self.localmap(x+1,y) == 1 && self.SystemInfo.globalview == 0
                        self.positions(i,3) = 3;
                    end               
                    if self.localmap(x,y+1) == 1 && self.SystemInfo.globalview == 0
                        self.positions(i,3) = 4;
                    end  
                end
            end
            for i = 2:1:len_path
                if self.positions(i,3) == 1 && self.positions(i-1,3) ~= 1
                    start_index1 = i;
                end
                if self.positions(i,3) ~= 1 && self.positions(i-1,3) == 1
                    end_index1 = i-1;
                end
                if self.positions(i,3) == 2 && self.positions(i-1,3) ~= 2
                    start_index2 = i;
                end
                if self.positions(i,3) ~= 2 && self.positions(i-1,3) == 2
                    end_index2 = i-1;
                end
                if self.positions(i,3) == 3 && self.positions(i-1,3) ~= 3
                    start_index3 = i;
                end
                if self.positions(i,3) ~= 3 && self.positions(i-1,3) == 3
                    end_index3 = i-1;
                end
                if self.positions(i,3) == 4 && self.positions(i-1,3) ~= 4 
                    start_index4 = i;
                end
                if self.positions(i,3) ~= 4 && self.positions(i-1,3) == 4
                    end_index4 = i-1;
                end                
                if end_index1 - start_index1 > 3
                    for j = start_index1:1:end_index1 
                        if self.positions(j,4) ~= 1
                            self.path(j,1) = self.path(j,1) + bias_offset;
                            self.positions(j,4) = 1;
                        end
                    end
                end
                if end_index2 - start_index2 > 3
                    for k = start_index2:1:end_index2
                        if self.positions(k,4) ~= 2
                            self.path(k,2) = self.path(k,2) + bias_offset;
                            self.positions(k,4) = 2;
                        end
                    end
                end    
                if end_index2 == start_index2 &&  self.positions(end_index2,4) ~= 2
                    if strcmp(self.MapInfo.name,self.special) == 1 && self.SystemInfo.globalview == 1
                        self.path(end_index2,2) = self.path(end_index2,2) + 1.6;
                        self.positions(end_index2,4) = 2;
                    end
                end
                if end_index3 - start_index3 > 2
                    for k = start_index3:1:end_index3
                        if self.positions(k,4) ~= 3
                            self.path(k,1) = self.path(k,1) + bias_offset;
                            self.positions(k,4) = 3;
                        end
                    end
                end                
                if end_index4 - start_index4 > 4
                    for k = start_index4:1:end_index4
                        if self.positions(k,4) ~= 4
                            self.path(k,2) = self.path(k,2) - bias_offset;
                            self.positions(k,4) = 4;
                        end
                    end
                end                 
            end
        end

        %找出路径中的转折点
        function [] = find_twist(self,path)
            slopes = [];
            self.twist_set = [];
            len_path = length(path(:,1));
            for i = 1:1:len_path - 1
                slope = atan2(path(i,2) - path(i+1,2) , path(i,1) - path(i+1,1));
                slopes = [slopes,slope];
                if length(slopes) > 1
                    if slopes(length(slopes)) ~= slopes(length(slopes)-1)
                        self.twist_set = [self.twist_set;path(i,:)];
                    end
                end
            end
        end  

        function path = prune_path_to_agent(~, path, observation)
            if size(path, 1) < 3
                return;
            end
            dx = path(:,1) - observation.agent.x;
            dy = path(:,2) - observation.agent.y;
            [~, idx] = min(dx .* dx + dy .* dy);
            idx = max(3, idx);
            path = path(1:idx, :);
        end

        function u = dynamic_safety_adjust(self, observation, u)
            if self.SystemInfo.smartAgent == 0 || ~isfield(observation, 'scanMap')
                return;
            end
            [xs, ys] = find(observation.scanMap > 1);
            if isempty(xs)
                return;
            end
            dx = xs - observation.agent.x;
            dy = ys - observation.agent.y;
            h = observation.agent.h;
            dist = sqrt(dx .* dx + dy .* dy);
            forward = dx .* cos(h) + dy .* sin(h);
            lateral = -dx .* sin(h) + dy .* cos(h);
            use_early_yield = strcmp(self.MapInfo.name, 'cmap');
            if use_early_yield
                near_limit = 1.9;
                front_limit = 7.0;
                lateral_limit = 2.6;
                avoid_gain = 0.30;
            else
                near_limit = 0;
                front_limit = 5.0;
                lateral_limit = 2.2;
                avoid_gain = 0.25;
            end
            near_mask = dist < near_limit;
            if use_early_yield && any(near_mask)
                side_bias = mean(lateral(near_mask));
                avoid_turn = -0.35;
                if side_bias < 0
                    avoid_turn = 0.35;
                end
                u(1) = min(u(1), 0.05);
                u(2) = max(min(u(2) + avoid_turn, self.AgentInfo.vsat), -self.AgentInfo.vsat);
                return;
            end
            front_mask = forward > -0.4 & forward < front_limit & abs(lateral) < lateral_limit;
            if ~any(front_mask)
                return;
            end
            front_dist = min(forward(front_mask));
            side_bias = mean(lateral(front_mask));
            avoid_turn = -avoid_gain;
            if side_bias < 0
                avoid_turn = avoid_gain;
            end
            if front_dist < 1.7
                u(1) = min(u(1), 0.05);
                u(2) = max(min(u(2) + avoid_turn, self.AgentInfo.vsat), -self.AgentInfo.vsat);
            elseif front_dist < 3.0
                u(1) = min(u(1), 0.25);
                u(2) = max(min(u(2) + 0.5 * avoid_turn, self.AgentInfo.vsat), -self.AgentInfo.vsat);
            elseif use_early_yield && front_dist < 5.0
                u(1) = min(u(1), 0.42);
                u(2) = max(min(u(2) + 0.25 * avoid_turn, self.AgentInfo.vsat), -self.AgentInfo.vsat);
            else
                if use_early_yield
                    u(1) = min(u(1), 0.65);
                else
                    u(1) = min(u(1), 0.55);
                end
            end
        end

        %碰撞检测函数
        function predict_collide(self,observation,ang_speed,v_speed)
            center_point = zeros(1,2);
            center_point(1) = observation.agent.x;
            center_point(2) = observation.agent.y;
            init_h = observation.agent.h;
            up_right_now = zeros(1,2);
            up_left_now = zeros(1,2);
            down_right_now = zeros(1,2);
            down_left_now = zeros(1,2);      
            now_w = observation.agent.w;
            now_ht = observation.agent.ht;
            up_right_now = [center_point(1) + now_w/2 * cos(init_h) + now_ht/2 * sin(init_h), center_point(2) + now_w/2 * sin(init_h) - now_ht/2 * cos(init_h)];
            up_left_now = [center_point(1) + now_w/2 * cos(init_h) - now_ht/2 * sin(init_h), center_point(2) + now_w/2 * sin(init_h) + now_ht/2 * cos(init_h)];
            down_right_now = [center_point(1) - now_w/2 * cos(init_h) + now_ht/2 * sin(init_h), center_point(2) - now_w/2 * sin(init_h) - now_ht/2 * cos(init_h)];
            down_left_now = [center_point(1) - now_w/2 * cos(init_h) - now_ht/2 * sin(init_h), center_point(2) - now_w/2 * sin(init_h) + now_ht/2 * cos(init_h)];           
            car_xs_now = [up_right_now(1),up_left_now(1),down_right_now(1),down_left_now(1)];
            car_ys_now = [up_right_now(2),up_left_now(2),down_right_now(2),down_left_now(2)];
            if ang_speed > 0.5 
                ang_speed = 0.5;
            elseif ang_speed < -0.5
                ang_speed = -0.5;
            end
            up_thres = 0.3;
            overlap_flag = 0;
            turn_flag = 0;
            if abs(ang_speed) < 1e-6
                deltax = v_speed * cos(init_h) * up_thres;
                deltay = v_speed * sin(init_h) * up_thres;
            else
                deltax = v_speed / ang_speed * (sin(init_h + ang_speed * up_thres) - sin(init_h));
                deltay = -v_speed / ang_speed * (cos(init_h + ang_speed * up_thres) - cos(init_h));
            end
            center_point(1) = center_point(1) + deltax;
            center_point(2) = center_point(2) + deltay;
            up_right_next = zeros(1,2);
            up_left_next = zeros(1,2);
            down_right_next = zeros(1,2);
            down_left_next = zeros(1,2);
            next_ht = observation.agent.ht;
            next_w = observation.agent.w;
            predict_h = init_h + ang_speed * 0.3;
            %求小车四个顶点的坐标
            up_right_next = [center_point(1) + next_w/2 * cos(predict_h) + next_ht/2 * sin(predict_h), center_point(2) + next_w/2 * sin(predict_h) - next_ht/2 * cos(predict_h)];
            up_left_next = [center_point(1) + next_w/2 * cos(predict_h) - next_ht/2 * sin(predict_h), center_point(2) + next_w/2 * sin(predict_h) + next_ht/2 * cos(predict_h)];
            down_right_next = [center_point(1) - next_w/2 * cos(predict_h) + next_ht/2 * sin(predict_h), center_point(2) - next_w/2 * sin(predict_h) - next_ht/2 * cos(predict_h)];
            down_left_next = [center_point(1) - next_w/2 * cos(predict_h) - next_ht/2 * sin(predict_h), center_point(2) - next_w/2 * sin(predict_h) + next_ht/2 * cos(predict_h)];
            car_xs_next = [up_right_next(1),up_left_next(1),down_right_next(1),down_left_next(1)];
            car_ys_next = [up_right_next(2),up_left_next(2),down_right_next(2),down_left_next(2)];
            if self.SystemInfo.globalview == 0
                tmpobmap = self.obmap;
            elseif self.SystemInfo.globalview == 1
                tmpobmap = observation.map;
            end
            sample_points = [ ...
                up_right_next; up_left_next; down_right_next; down_left_next; ...
                (up_right_next + up_left_next) / 2; ...
                (up_left_next + down_left_next) / 2; ...
                (down_left_next + down_right_next) / 2; ...
                (down_right_next + up_right_next) / 2; ...
                center_point];
            for k = 1:size(sample_points, 1)
                i = floor(sample_points(k, 1)) + 1;
                j = floor(sample_points(k, 2)) + 1;
                if i < 1 || j < 1 || i > 50 || j > 50
                    turn_flag = 1;
                    break;
                end
                overlap_flag = tmpobmap(i, j) ~= 0;
                if overlap_flag
                    turn_flag = 1;
                    break;
                end
            end
            self.turn_flag_hist = [self.turn_flag_hist,turn_flag];
            if turn_flag == 1
                self.new_ang_speed = 0;
            elseif turn_flag == 0 
                self.new_ang_speed = ang_speed;
            end
        end

        function load_ini_file(self, file)
            %使用try-catch避免错误
            try 
                iniFile = INI('File', file);
                iniFile.read();
                sections = iniFile.get('Sections');
                % 根据名字对应给出ini文件中的各项信息
                for i = 1:length(sections)
                    sectionName = sections{i};
                    switch sectionName
                        case 'StartPos'
                            self.StartPos = iniFile.get('StartPos');
                        case 'EndPos'
                            self.EndPos = iniFile.get('EndPos');
                        case 'Map'
                            self.MapInfo = iniFile.get('Map');
                        case 'RangeFinder'
                            self.RangeFinderInfo = iniFile.get('RangeFinder');
                        case 'Agent'
                            self.AgentInfo = iniFile.get('Agent');
                        case 'System'
                            self.SystemInfo = iniFile.get('System');
                    end
                end
            catch ME
                %输出报错的信息
                error('Error loading INI file:%s', ME.message);
            end
        end

        function a = wrap_to_pi(~, a)
            a = atan2(sin(a), cos(a));
        end

        function setTraining(~, ~)
        end

        function finishEpisode(~, ~)
        end     
    end
end
