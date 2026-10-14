classdef Policy_Exp1_DynamicPrediction < Policy_AStar_PD_Optimized
    properties
        prevDynamicCentersExp1
    end

    methods
        function self = Policy_Exp1_DynamicPrediction()
            self@Policy_AStar_PD_Optimized();
            self.prevDynamicCentersExp1 = [];
        end

        function [] = add_bias(self)
            safe_base_add_bias(self);
        end

        function u = dynamic_safety_adjust(self, observation, u)
            u = dynamic_safety_adjust@Policy_AStar_PD_Optimized(self, observation, u);

            tracks = self.estimate_dynamic_tracks(observation);
            if isempty(tracks)
                return;
            end

            h = observation.agent.h;
            ch = cos(h);
            sh = sin(h);
            agentSize = max(observation.agent.w, observation.agent.ht);
            scanRange = self.RangeFinderInfo.scanRange;
            frontLimit = min(scanRange, 5.5);
            lateralLimit = max(1.65, 0.95 * agentSize + 0.75);
            stopDist = max(1.15, 0.80 * agentSize);
            predTimes = [0.9, 1.5];

            risk = 0;
            bestForward = inf;
            sideSum = 0;
            sideCount = 0;

            for i = 1:size(tracks, 1)
                trackSpeed = hypot(tracks(i, 4), tracks(i, 5));
                if trackSpeed < 0.18
                    continue;
                end
                for j = 1:length(predTimes)
                    px = tracks(i, 2) + tracks(i, 4) * predTimes(j);
                    py = tracks(i, 3) + tracks(i, 5) * predTimes(j);
                    dx = px - observation.agent.x;
                    dy = py - observation.agent.y;
                    forward = dx * ch + dy * sh;
                    lateral = -dx * sh + dy * ch;
                    if forward > -0.2 && forward < frontLimit && abs(lateral) < lateralLimit
                        bestForward = min(bestForward, forward);
                        sideSum = sideSum + lateral;
                        sideCount = sideCount + 1;
                        frontRisk = (frontLimit - forward) / frontLimit;
                        sideRisk = (lateralLimit - abs(lateral)) / lateralLimit;
                        timeRisk = (length(predTimes) - j + 1) / length(predTimes);
                        risk = max(risk, frontRisk * sideRisk * timeRisk);
                    end
                end
            end

            if sideCount == 0
                return;
            end

            sideBias = sideSum / sideCount;
            avoidTurn = -0.24;
            if sideBias < 0
                avoidTurn = 0.24;
            end

            if bestForward < stopDist
                u(1) = min(u(1), 0.10);
                u(2) = self.clamp_value(u(2) + avoidTurn, -self.AgentInfo.vsat, self.AgentInfo.vsat);
            elseif risk > 0.55
                u(1) = min(u(1), 0.35 + 0.12 * max(0, bestForward - stopDist));
                u(2) = self.clamp_value(u(2) + 0.65 * avoidTurn, -self.AgentInfo.vsat, self.AgentInfo.vsat);
            elseif risk > 0.32
                u(1) = min(u(1), 0.72);
                u(2) = self.clamp_value(u(2) + 0.30 * avoidTurn, -self.AgentInfo.vsat, self.AgentInfo.vsat);
            end
        end
    end

    methods (Access = protected)
        function tracks = estimate_dynamic_tracks(self, observation)
            tracks = [];
            if ~self.has_numeric_scan(observation)
                self.prevDynamicCentersExp1 = [];
                return;
            end

            scanMap = observation.scanMap;
            [xs, ys, values] = find(scanMap);
            mask = values > 1;
            xs = xs(mask);
            ys = ys(mask);
            ids = values(mask);
            if isempty(xs)
                self.prevDynamicCentersExp1 = [];
                return;
            end

            uniqueIds = unique(ids(:))';
            centers = zeros(length(uniqueIds), 3);
            dtLocal = max(self.dt, 1e-3);
            for i = 1:length(uniqueIds)
                id = uniqueIds(i);
                idMask = ids == id;
                cx = mean(xs(idMask));
                cy = mean(ys(idMask));
                vx = 0;
                vy = 0;

                if ~isempty(self.prevDynamicCentersExp1)
                    prevIdx = find(self.prevDynamicCentersExp1(:, 1) == id, 1);
                    if isempty(prevIdx)
                        d = hypot(self.prevDynamicCentersExp1(:, 2) - cx, self.prevDynamicCentersExp1(:, 3) - cy);
                        [bestDist, prevIdx] = min(d);
                        if bestDist > 3.0
                            prevIdx = [];
                        end
                    end
                    if ~isempty(prevIdx)
                        vx = (cx - self.prevDynamicCentersExp1(prevIdx, 2)) / dtLocal;
                        vy = (cy - self.prevDynamicCentersExp1(prevIdx, 3)) / dtLocal;
                        speed = hypot(vx, vy);
                        if speed > 1.5
                            vx = vx / speed * 1.5;
                            vy = vy / speed * 1.5;
                        end
                    end
                end

                tracks = [tracks; id, cx, cy, vx, vy]; %#ok<AGROW>
                centers(i, :) = [id, cx, cy];
            end

            self.prevDynamicCentersExp1 = centers;
        end

        function ok = has_numeric_scan(~, observation)
            ok = isprop(observation, 'scanMap') && ~isempty(observation.scanMap) && ~ischar(observation.scanMap);
        end

        function value = clamp_value(~, value, lo, hi)
            value = min(max(value, lo), hi);
        end
    end
end
