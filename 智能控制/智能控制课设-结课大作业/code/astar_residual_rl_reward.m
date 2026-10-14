function reward = astar_residual_rl_reward(observation, nextObservation, baseAction, executedAction, done)
%ASTAR_RESIDUAL_RL_REWARD Reward for residual RL on the A*+PD baseline.

agent = observation.agent;
nextAgent = nextObservation.agent;
goal = [observation.endPos.x; observation.endPos.y];
pos = [agent.x; agent.y];
nextPos = [nextAgent.x; nextAgent.y];

prevDist = norm(goal - pos);
nextDist = norm(goal - nextPos);
progress = prevDist - nextDist;
scoreDelta = nextObservation.score - observation.score;

residual = executedAction - baseAction;
residualCost = residual(1)^2 / (0.20^2) + residual(2)^2 / (0.16^2);
turnCost = executedAction(2)^2;

reward = 0;
reward = reward + 0.25 * scoreDelta;
reward = reward + 2.5 * progress;
reward = reward - 0.020 * residualCost;
reward = reward - 0.004 * turnCost;

if nextObservation.collide ~= 0 && observation.collide == 0
    reward = reward - 8.0;
end

if done && abs(nextAgent.x - nextObservation.endPos.x) < 1 && abs(nextAgent.y - nextObservation.endPos.y) < 1
    reward = reward + 220.0 + 0.02 * nextObservation.score;
elseif done
    reward = reward - 5000.0 - 100.0 * nextDist;
end
end
