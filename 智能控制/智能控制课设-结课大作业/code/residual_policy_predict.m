function residual = residual_policy_predict(features, weightsPath)
%RESIDUAL_POLICY_PREDICT Lightweight MLP inference for exported residual policy.
if nargin < 2 || isempty(weightsPath)
    weightsPath = fullfile(pwd, 'residual_policy_weights.mat');
end

persistent cache cachedPath
if isempty(cache) || ~strcmp(cachedPath, weightsPath)
    if ~exist(weightsPath, 'file')
        residual = [0, 0];
        return;
    end
    cache = load(weightsPath);
    cachedPath = weightsPath;
end

x = features(:);
if length(x) ~= cache.feature_dim
    residual = [0, 0];
    return;
end

mu = cache.x_mean(:);
sigma = cache.x_std(:);
x = (x - mu) ./ sigma;
z1 = tanh(cache.W1 * x + cache.b1);
z2 = tanh(cache.W2 * z1 + cache.b2);
out = tanh(cache.W3 * z2 + cache.b3);

du = cache.max_du * out(1);
dv = cache.max_dv * out(2);
residual = [du, dv];
end
