function waypoint = astar_residual_waypoint(path, observation)
%ASTAR_RESIDUAL_WAYPOINT Pick the local target used by A*+PD residual features.

waypoint = [observation.endPos.x; observation.endPos.y];
if isempty(path) || size(path, 1) < 2
    return;
end

idx = size(path, 1) - 1;
if size(path, 1) >= 3
    idx = size(path, 1) - 2;
end
idx = max(1, min(size(path, 1), idx));
waypoint = path(idx, :)';
end
