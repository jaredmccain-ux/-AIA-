%% main_nn.m
%  使用神经网络控制器运行水箱仿真
%  前置条件: 先运行 generate_train_data.m 和 train_nn.m

clear all;
clc;

abspath = utils('abspath');
env = Env(abspath('sys.ini'));
policy = Policy_NN();  % 使用神经网络控制器

if (env.succeed)
    observation = env.reset();
    while 1
        env.render();
        action = policy.action(observation);
        
        [observation, done, info] = env.step(action);
        
        disp(info);
        if (done)
            break;
        end
        wait(100);
    end
end


function wait(ms)
    time = ms / 1000;
    pause(time);
end
