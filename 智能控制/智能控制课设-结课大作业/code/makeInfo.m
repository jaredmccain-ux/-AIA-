function info = makeInfo(observation, step, done, reached, baseAction, action, mode)
%MAKEINFO Compact MATLAB Engine friendly episode information struct.

info = struct();
info.step = double(step);
info.t = double(observation.t);
info.score = double(observation.score);
info.x = double(observation.agent.x);
info.y = double(observation.agent.y);
info.h = double(observation.agent.h);
info.goal_x = double(observation.endPos.x);
info.goal_y = double(observation.endPos.y);
info.collide = double(observation.collide ~= 0);
info.done = double(done);
info.reached = double(reached);
info.base_u = double(baseAction(1));
info.base_v = double(baseAction(2));
info.action_u = double(action(1));
info.action_v = double(action(2));
info.residual_u = double(action(1) - baseAction(1));
info.residual_v = double(action(2) - baseAction(2));
info.mode = char(mode);
end

