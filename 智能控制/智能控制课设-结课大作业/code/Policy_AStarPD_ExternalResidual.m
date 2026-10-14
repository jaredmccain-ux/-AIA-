classdef Policy_AStarPD_ExternalResidual < Policy_AStar_PD_Optimized
    properties
        pendingBaseAction
        pendingWaypoint
        pendingMode
        maxResidualU
        maxResidualV
        lastBaseAction
        lastExecutedAction
        lastResidual
        lastWaypoint
        lastMode
        prevDynamicCenters
        pendingResidualGate
        pendingDynamicRisk
    end

    methods
        function self = Policy_AStarPD_ExternalResidual()
            self@Policy_AStar_PD_Optimized();
            self.pendingBaseAction = [0, 0];
            self.pendingWaypoint = [NaN; NaN];
            self.pendingMode = 'astar_pd';
            self.maxResidualU = 0.10;
            self.maxResidualV = 0.08;
            self.lastBaseAction = [0, 0];
            self.lastExecutedAction = [0, 0];
            self.lastResidual = [0, 0];
            self.lastWaypoint = [NaN; NaN];
            self.lastMode = 'init';
            self.prevDynamicCenters = [];
            self.pendingResidualGate = 0;
            self.pendingDynamicRisk = 0;
        end

        function [features, baseAction, record] = computeBaseDecision(self, observation)
            baseAction = self.action(observation);
            waypoint = astar_residual_waypoint(self.path, observation);

            self.pendingBaseAction = baseAction;
            self.pendingWaypoint = waypoint;
            self.pendingMode = 'astar_pd';

            [features, centers, riskInfo] = astar_residual_feature_vector(observation, baseAction, waypoint, self.prevDynamicCenters, self.dt);
            self.prevDynamicCenters = centers;
            self.pendingResidualGate = riskInfo.gate;
            self.pendingDynamicRisk = riskInfo.risk;
            features = features(:)';

            record = struct();
            record.baseAction = baseAction;
            record.waypoint = waypoint;
            record.mode = self.pendingMode;
            record.features = features;
            record.dynamicRisk = riskInfo.risk;
            record.residualGate = riskInfo.gate;
        end

        function action = actionFromResidual(self, observation, residual)
            if nargin < 3 || isempty(residual)
                residual = [0, 0];
            end
            residual = double(residual(:)');
            if numel(residual) < 2
                residual = [0, 0];
            else
                residual = residual(1:2);
            end

            residual(1) = self.clamp_value(residual(1), -self.maxResidualU, self.maxResidualU);
            residual(2) = self.clamp_value(residual(2), -self.maxResidualV, self.maxResidualV);
            if self.pendingResidualGate < 0.5
                residual = [0, 0];
            end

            baseAction = self.pendingBaseAction;
            action = self.safetyScaledAction(observation, baseAction, residual);

            self.lastBaseAction = baseAction;
            self.lastExecutedAction = action;
            self.lastResidual = action - baseAction;
            self.lastWaypoint = self.pendingWaypoint;
            self.lastMode = self.pendingMode;
        end

        function record = getLastResidualRecord(self, observation)
            record = struct();
            record.features = astar_residual_feature_vector(observation, self.lastBaseAction, self.lastWaypoint, self.prevDynamicCenters, self.dt);
            record.baseAction = self.lastBaseAction;
            record.executedAction = self.lastExecutedAction;
            record.residual = self.lastResidual;
            record.waypoint = self.lastWaypoint;
            record.mode = self.lastMode;
        end
    end

    methods (Access = protected)
        function action = safetyScaledAction(self, observation, baseAction, residual)
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
            horizon = 0.6;
            steps = max(1, ceil(horizon / dtLocal));
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
