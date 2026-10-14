classdef Policy_AStarPD_RLResidual < Policy_AStar_PD_Optimized
    properties
        residualPolicyPath
        residualEnabled
        pendingWaypoint
        maxResidualU
        maxResidualV
        lastBaseAction
        lastExecutedAction
        lastResidual
        prevDynamicCenters
        lastDynamicRisk
        lastResidualGate
    end

    methods
        function self = Policy_AStarPD_RLResidual(weightsPath)
            self@Policy_AStar_PD_Optimized();
            if nargin < 1 || isempty(weightsPath)
                weightsPath = fullfile(pwd, 'astar_residual_policy_weights.mat');
            end
            self.residualPolicyPath = weightsPath;
            self.residualEnabled = true;
            self.pendingWaypoint = [NaN; NaN];
            self.maxResidualU = 0.10;
            self.maxResidualV = 0.08;
            self.lastBaseAction = [0, 0];
            self.lastExecutedAction = [0, 0];
            self.lastResidual = [0, 0];
            self.prevDynamicCenters = [];
            self.lastDynamicRisk = 0;
            self.lastResidualGate = 0;
        end

        function action = action(self, observation)
            baseAction = action@Policy_AStar_PD_Optimized(self, observation);
            self.pendingWaypoint = astar_residual_waypoint(self.path, observation);
            residual = [0, 0];
            [features, centers, riskInfo] = astar_residual_feature_vector(observation, baseAction, self.pendingWaypoint, self.prevDynamicCenters, self.dt);
            self.prevDynamicCenters = centers;
            self.lastDynamicRisk = riskInfo.risk;
            self.lastResidualGate = riskInfo.gate;
            if self.residualEnabled && riskInfo.gate > 0.5
                residual = residual_policy_predict(features, self.residualPolicyPath);
            end
            action = self.safetyScaledAction(observation, baseAction, residual);
            self.lastBaseAction = baseAction;
            self.lastExecutedAction = action;
            self.lastResidual = action - baseAction;
        end
    end

    methods (Access = private)
        function action = safetyScaledAction(self, observation, baseAction, residual)
            residual = double(residual(:)');
            if numel(residual) < 2
                residual = [0, 0];
            else
                residual = residual(1:2);
            end
            residual(1) = self.clamp_value(residual(1), -self.maxResidualU, self.maxResidualU);
            residual(2) = self.clamp_value(residual(2), -self.maxResidualV, self.maxResidualV);

            action = baseAction;
            for scale = [1.0, 0.5, 0.25]
                candidate = [ ...
                    self.clamp_value(baseAction(1) + scale * residual(1), -self.AgentInfo.usat, self.AgentInfo.usat), ...
                    self.clamp_value(baseAction(2) + scale * residual(2), -self.AgentInfo.vsat, self.AgentInfo.vsat)];
                if self.isShortHorizonSafe(observation, candidate)
                    action = candidate;
                    return;
                end
            end
        end

        function ok = isShortHorizonSafe(self, observation, action)
            ok = true;
            if isempty(observation.map) || ischar(observation.map)
                return;
            end
            x = observation.agent.x;
            y = observation.agent.y;
            h = observation.agent.h;
            dtLocal = max(self.dt, 0.1);
            steps = max(1, ceil(0.6 / dtLocal));
            occ = observation.map ~= 0;
            if ~isempty(observation.scanMap) && ~ischar(observation.scanMap)
                occ = occ | (observation.scanMap == 1);
            end
            for k = 1:steps
                x = x + action(1) * cos(h) * dtLocal;
                y = y + action(1) * sin(h) * dtLocal;
                h = h + action(2) * dtLocal;
                samples = self.vehicleSamples(x, y, h, observation.agent.w, observation.agent.ht);
                for i = 1:size(samples, 1)
                    ix = round(samples(i, 1));
                    iy = round(samples(i, 2));
                    if ix < 1 || iy < 1 || ix > size(occ, 1) || iy > size(occ, 2) || occ(ix, iy)
                        ok = false;
                        return;
                    end
                end
            end
        end

        function samples = vehicleSamples(~, x, y, h, w, ht)
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

        function value = clamp_value(~, value, lo, hi)
            value = min(max(value, lo), hi);
        end
    end
end
