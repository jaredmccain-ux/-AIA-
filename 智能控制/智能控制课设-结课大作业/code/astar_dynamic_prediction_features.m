function [extra, centers, info] = astar_dynamic_prediction_features(observation, baseAction, prevCenters, dt)
%ASTAR_DYNAMIC_PREDICTION_FEATURES Estimate Smart-Agent velocity and future risk.

if nargin < 2 || isempty(baseAction)
    baseAction = [0, 0];
end
if nargin < 3
    prevCenters = [];
end
if nargin < 4 || isempty(dt)
    dt = 0.3;
end
dt = max(double(dt), 1e-3);

extra = zeros(1, 8);
centers = [];
info = struct('risk', 0, 'gate', 0, 'minForward', 1, 'lateral', 0, ...
    'closing', 0, 'crossing', 0, 'dynamicCount', 0, 'movingCount', 0);

if ~isprop(observation, 'scanMap') || isempty(observation.scanMap) || ischar(observation.scanMap)
    return;
end

scanMap = observation.scanMap;
[xs, ys, values] = find(scanMap);
mask = values > 1;
xs = xs(mask);
ys = ys(mask);
ids = values(mask);
if isempty(xs)
    return;
end

uniqueIds = unique(ids(:))';
centers = zeros(length(uniqueIds), 5);
for i = 1:length(uniqueIds)
    id = uniqueIds(i);
    idMask = ids == id;
    cx = mean(xs(idMask));
    cy = mean(ys(idMask));
    vx = 0;
    vy = 0;
    if ~isempty(prevCenters)
        prevIdx = find(prevCenters(:, 1) == id, 1);
        if isempty(prevIdx)
            d = hypot(prevCenters(:, 2) - cx, prevCenters(:, 3) - cy);
            [bestDist, prevIdx] = min(d);
            if bestDist > 3.0
                prevIdx = [];
            end
        end
        if ~isempty(prevIdx)
            vx = (cx - prevCenters(prevIdx, 2)) / dt;
            vy = (cy - prevCenters(prevIdx, 3)) / dt;
            speed = hypot(vx, vy);
            if speed > 1.5
                vx = vx / speed * 1.5;
                vy = vy / speed * 1.5;
            end
        end
    end
    centers(i, :) = [id, cx, cy, vx, vy];
end

agent = observation.agent;
h = agent.h;
ch = cos(h);
sh = sin(h);
agentSize = max(agent.w, agent.ht);
frontLimit = 7.0;
lateralLimit = max(2.0, 1.05 * agentSize + 0.55);
predTimes = [0, 0.6, 1.2, 1.8];

bestRisk = 0;
bestForward = frontLimit;
bestLateral = 0;
bestClosing = 0;
bestCrossing = 0;
movingCount = 0;

for i = 1:size(centers, 1)
    vx = centers(i, 4);
    vy = centers(i, 5);
    speed = hypot(vx, vy);
    if speed > 0.15
        movingCount = movingCount + 1;
    end
    vForward = vx * ch + vy * sh;
    vLateral = -vx * sh + vy * ch;
    closing = max(0, baseAction(1) - vForward);
    crossing = abs(vLateral);

    for j = 1:length(predTimes)
        t = predTimes(j);
        px = centers(i, 2) + vx * t;
        py = centers(i, 3) + vy * t;
        dx = px - agent.x;
        dy = py - agent.y;
        forward = dx * ch + dy * sh;
        lateral = -dx * sh + dy * ch;

        if forward < -0.4 || forward > frontLimit || abs(lateral) > lateralLimit
            continue;
        end
        frontRisk = (frontLimit - forward) / frontLimit;
        sideRisk = (lateralLimit - abs(lateral)) / lateralLimit;
        timeRisk = 1 / (1 + 0.45 * t);
        speedRisk = 0.65 + 0.35 * min(1, (closing + crossing) / 1.5);
        risk = frontRisk * sideRisk * timeRisk * speedRisk;
        if risk > bestRisk
            bestRisk = risk;
            bestForward = max(0, forward);
            bestLateral = lateral;
            bestClosing = closing;
            bestCrossing = crossing;
        end
    end
end

gate = double(bestRisk > 0.16);
extra = [ ...
    min(size(centers, 1) / 20, 1), ...
    min(movingCount / 20, 1), ...
    min(bestRisk, 1), ...
    gate, ...
    min(bestForward / frontLimit, 1), ...
    max(min(bestLateral / lateralLimit, 1), -1), ...
    min(bestClosing / 2, 1), ...
    min(bestCrossing / 2, 1) ...
];

info.risk = min(bestRisk, 1);
info.gate = gate;
info.minForward = bestForward;
info.lateral = bestLateral;
info.closing = bestClosing;
info.crossing = bestCrossing;
info.dynamicCount = size(centers, 1);
info.movingCount = movingCount;
end
