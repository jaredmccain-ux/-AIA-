function u = dwa_hybrid_refine(policy, observation, u0)
u = u0;
if ~hasNumericScan(observation)
    return;
end

scanMap = observation.scanMap;
[dxs, dys, values] = find(scanMap);
dynMask = values > 1;
dynPoints = [dxs(dynMask), dys(dynMask)];
if isempty(dynPoints)
    return;
end

agent = observation.agent;
h = agent.h;
rel = dynPoints - [agent.x, agent.y];
forward = rel(:, 1) * cos(h) + rel(:, 2) * sin(h);
lateral = -rel(:, 1) * sin(h) + rel(:, 2) * cos(h);
frontMask = forward > -0.5 & forward < min(policy.RangeFinderInfo.scanRange, 5.0) & abs(lateral) < 2.2;
if ~any(frontMask)
    return;
end

target = localPathTarget(policy, observation);
finalGoal = [observation.endPos.x, observation.endPos.y];
staticMap = localStaticMap(observation);
staticMap(scanMap == 1) = true;

uSamples = unique(round(clampArray([u0(1) - 0.22, u0(1) - 0.10, u0(1), u0(1) + 0.10, 0.05, 0.18, 0.35, 0.55, 0.75], -policy.AgentInfo.usat, policy.AgentInfo.usat), 3));
wSamples = unique(round(clampArray([u0(2) - 0.28, u0(2) - 0.14, u0(2), u0(2) + 0.14, u0(2) + 0.28, -policy.AgentInfo.vsat, 0, policy.AgentInfo.vsat], -policy.AgentInfo.vsat, policy.AgentInfo.vsat), 3));

baseScore = scoreCandidate(policy, agent, target, finalGoal, staticMap, dynPoints, u0(1), u0(2));
bestScore = baseScore;
bestAction = u0;

for i = 1:length(uSamples)
    for j = 1:length(wSamples)
        candU = uSamples(i);
        candW = wSamples(j);
        score = scoreCandidate(policy, agent, target, finalGoal, staticMap, dynPoints, candU, candW);
        if score > bestScore
            bestScore = score;
            bestAction = [candU, candW];
        end
    end
end

nearestFront = min(forward(frontMask));
if bestScore > baseScore + 1.5 && (baseScore < -50 || nearestFront < 2.8)
    if bestAction(1) < u0(1) - 0.22 && baseScore > -500
        return;
    end
    u = bestAction;
elseif baseScore < -500
    sideBias = mean(lateral(frontMask));
    turn = -policy.AgentInfo.vsat;
    if sideBias < 0
        turn = policy.AgentInfo.vsat;
    end
    u = [min(u0(1), 0.04), turn];
end

u(1) = min(max(u(1), -policy.AgentInfo.usat), policy.AgentInfo.usat);
u(2) = min(max(u(2), -policy.AgentInfo.vsat), policy.AgentInfo.vsat);
end

function score = scoreCandidate(policy, agent, target, finalGoal, staticMap, dynPoints, u, w)
dt = max(policy.dt, 0.1);
horizon = 1.2;
n = max(1, ceil(horizon / dt));
x = agent.x;
y = agent.y;
theta = agent.h;
startTargetDist = norm(target - [x, y]);
startGoalDist = norm(finalGoal - [x, y]);
minClearance = 20;
collision = false;
outRisk = 0;
dynamicRisk = 0;

for k = 1:n
    x = x + u * cos(theta) * dt;
    y = y + u * sin(theta) * dt;
    theta = theta + w * dt;

    samples = vehicleSamples(x, y, theta, agent.w, agent.ht);
    for p = 1:size(samples, 1)
        ix = round(samples(p, 1));
        iy = round(samples(p, 2));
        if ix < 1 || iy < 1 || ix > size(staticMap, 1) || iy > size(staticMap, 2)
            collision = true;
            outRisk = outRisk + 2;
            continue;
        end
        if staticMap(ix, iy)
            collision = true;
        end
    end

    clearance = localClearance(staticMap, x, y, dynPoints);
    minClearance = min(minClearance, clearance);
    if clearance < max(1.05, 0.85 * max(agent.w, agent.ht))
        collision = true;
    end
    dynamicRisk = dynamicRisk + dynamicRiskAt(x, y, dynPoints, max(agent.w, agent.ht));
end

endPos = [x, y];
targetDist = norm(target - endPos);
goalDist = norm(finalGoal - endPos);
desired = atan2(target(2) - y, target(1) - x);
headingError = atan2(sin(desired - theta), cos(desired - theta));
progress = startTargetDist - targetDist;
goalProgress = startGoalDist - goalDist;
clearanceScore = min(minClearance, 5) / 5;

score = 8.5 * progress + 4.0 * goalProgress + 1.4 * cos(headingError) + 1.3 * clearanceScore + 0.7 * u - 0.22 * abs(w);
score = score - 3.8 * dynamicRisk - 7.0 * outRisk;
if collision
    score = score - 1000;
end
end

function samples = vehicleSamples(x, y, h, w, ht)
forward = [cos(h), sin(h)];
side = [sin(h), -cos(h)];
center = [x, y];
samples = [ ...
    center; ...
    center + 0.5 * w * forward + 0.5 * ht * side; ...
    center + 0.5 * w * forward - 0.5 * ht * side; ...
    center - 0.5 * w * forward + 0.5 * ht * side; ...
    center - 0.5 * w * forward - 0.5 * ht * side; ...
    center + 0.5 * w * forward; ...
    center - 0.5 * w * forward];
end

function clearance = localClearance(staticMap, x, y, dynPoints)
clearance = 20;
ix = round(x);
iy = round(y);
radius = 3;
xmin = max(1, ix - radius);
xmax = min(size(staticMap, 1), ix + radius);
ymin = max(1, iy - radius);
ymax = min(size(staticMap, 2), iy + radius);
if ix < 1 || iy < 1 || ix > size(staticMap, 1) || iy > size(staticMap, 2)
    clearance = 0;
    return;
end

[ox, oy] = find(staticMap(xmin:xmax, ymin:ymax));
for i = 1:length(ox)
    px = ox(i) + xmin - 1;
    py = oy(i) + ymin - 1;
    clearance = min(clearance, hypot(x - px, y - py));
end

if ~isempty(dynPoints)
    d = hypot(dynPoints(:, 1) - x, dynPoints(:, 2) - y);
    clearance = min(clearance, min(d) - 0.7);
end
end

function risk = dynamicRiskAt(x, y, dynPoints, agentSize)
risk = 0;
if isempty(dynPoints)
    return;
end
d = hypot(dynPoints(:, 1) - x, dynPoints(:, 2) - y);
limit = max(4.0, 2.6 * agentSize);
near = d < limit;
if any(near)
    risk = sum(((limit - d(near)) / limit) .^ 2);
end
end

function target = localPathTarget(policy, observation)
target = [observation.endPos.x, observation.endPos.y];
if isempty(policy.path) || size(policy.path, 1) < 3
    return;
end
path = policy.path;
d = hypot(path(:, 1) - observation.agent.x, path(:, 2) - observation.agent.y);
[~, idx] = min(d);
idx = max(1, idx - 4);
target = path(idx, :);
end

function staticMap = localStaticMap(observation)
if isprop(observation, 'map') && ~isempty(observation.map) && ~ischar(observation.map)
    staticMap = observation.map > 0;
else
    staticMap = false(50, 50);
end
end

function ok = hasNumericScan(observation)
ok = isprop(observation, 'scanMap') && ~isempty(observation.scanMap) && ~ischar(observation.scanMap);
end

function values = clampArray(values, lo, hi)
values = min(max(values, lo), hi);
end
