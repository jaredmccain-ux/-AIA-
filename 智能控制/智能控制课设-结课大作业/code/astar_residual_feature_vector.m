function [features, centers, info] = astar_residual_feature_vector(observation, baseAction, waypoint, prevCenters, dt)
%ASTAR_RESIDUAL_FEATURE_VECTOR State vector for residual RL on A*+PD baseline.
%
% The vector deliberately keeps the baseline action in the state. The actor is
% trained to output a small residual, not a replacement action.

if nargin < 4
    prevCenters = [];
end
if nargin < 5
    dt = 0.3;
end

baseFeatures = residual_feature_vector(observation, baseAction, waypoint);
[dynamicFeatures, centers, info] = astar_dynamic_prediction_features(observation, baseAction, prevCenters, dt);
features = [baseFeatures, dynamicFeatures];
end
