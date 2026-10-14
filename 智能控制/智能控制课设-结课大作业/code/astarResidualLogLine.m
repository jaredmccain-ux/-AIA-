function astarResidualLogLine(tag, info, reward)
%ASTARRESIDUALLOGLINE Append A* residual RL environment diagnostics.

global ASTAR_RL_LOG_FID

if nargin < 3
    reward = NaN;
end
if isempty(ASTAR_RL_LOG_FID) || ASTAR_RL_LOG_FID < 0
    return;
end

try
    dynamicRisk = getOptionalNumber(info, 'dynamic_risk');
    residualGate = getOptionalNumber(info, 'residual_gate');
    fprintf(ASTAR_RL_LOG_FID, ['%s step=%d t=%.3f score=%.3f reward=%.3f pos=(%.3f,%.3f) ', ...
        'done=%d reached=%d collide=%d base=(%.3f,%.3f) action=(%.3f,%.3f) ', ...
        'residual=(%.3f,%.3f) risk=%.3f gate=%d mode=%s\n'], ...
        tag, info.step, info.t, info.score, reward, info.x, info.y, info.done, ...
        info.reached, info.collide, info.base_u, info.base_v, info.action_u, ...
        info.action_v, info.residual_u, info.residual_v, dynamicRisk, round(residualGate), info.mode);
    fflush(ASTAR_RL_LOG_FID);
catch
end
end

function value = getOptionalNumber(s, fieldName)
if isstruct(s) && isfield(s, fieldName)
    value = double(s.(fieldName));
else
    value = 0;
end
end
