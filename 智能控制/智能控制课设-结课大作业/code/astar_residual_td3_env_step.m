function [nextFeatures, reward, done, info] = astar_residual_td3_env_step(residual)
%ASTAR_RESIDUAL_TD3_ENV_STEP Execute one residual action for TD3 training.

global ASTAR_RL_ENV ASTAR_RL_POLICY ASTAR_RL_OBS ASTAR_RL_STEP ASTAR_RL_MAX_STEPS

if isempty(ASTAR_RL_ENV) || isempty(ASTAR_RL_POLICY) || isempty(ASTAR_RL_OBS)
    error('Call astar_residual_td3_env_reset before astar_residual_td3_env_step.');
end

if nargin < 1 || isempty(residual)
    residual = [0, 0];
end

baseAction = ASTAR_RL_POLICY.pendingBaseAction;
action = ASTAR_RL_POLICY.actionFromResidual(ASTAR_RL_OBS, residual);
[nextObs, envDone, ~] = ASTAR_RL_ENV.step(action);
ASTAR_RL_STEP = ASTAR_RL_STEP + 1;

reached = isReached(nextObs);
timeoutDone = ASTAR_RL_STEP >= ASTAR_RL_MAX_STEPS;
done = logical(envDone) || timeoutDone;
reward = astar_residual_rl_reward(ASTAR_RL_OBS, nextObs, baseAction, action, done);

if done
    nextFeatures = zeros(1, 34);
    info = makeInfo(nextObs, ASTAR_RL_STEP, done, reached, baseAction, action, ASTAR_RL_POLICY.lastMode);
    info.dynamic_risk = double(ASTAR_RL_POLICY.pendingDynamicRisk);
    info.residual_gate = double(ASTAR_RL_POLICY.pendingResidualGate);
else
    ASTAR_RL_OBS = nextObs;
    [nextFeatures, nextBaseAction, record] = ASTAR_RL_POLICY.computeBaseDecision(ASTAR_RL_OBS);
    info = makeInfo(nextObs, ASTAR_RL_STEP, done, reached, nextBaseAction, action, record.mode);
    info.dynamic_risk = double(record.dynamicRisk);
    info.residual_gate = double(record.residualGate);
end

astarResidualLogLine('step', info, reward);
end

function reached = isReached(observation)
reached = abs(observation.agent.x - observation.endPos.x) < 1 && ...
    abs(observation.agent.y - observation.endPos.y) < 1;
end
