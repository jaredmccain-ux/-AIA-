function logLine(tag, info, reward)
%LOGLINE Append online TD3 environment diagnostics to a txt log.

global TD3_LOG_FID

if nargin < 3
    reward = NaN;
end
if isempty(TD3_LOG_FID) || TD3_LOG_FID < 0
    return;
end

try
    fprintf(TD3_LOG_FID, ['%s step=%d t=%.3f score=%.3f reward=%.3f pos=(%.3f,%.3f) ', ...
        'done=%d reached=%d collide=%d base=(%.3f,%.3f) action=(%.3f,%.3f) ', ...
        'residual=(%.3f,%.3f) mode=%s\n'], ...
        tag, info.step, info.t, info.score, reward, info.x, info.y, info.done, ...
        info.reached, info.collide, info.base_u, info.base_v, info.action_u, ...
        info.action_v, info.residual_u, info.residual_v, info.mode);
    fflush(TD3_LOG_FID);
catch
end
end
