classdef Policy_AStar_PD_FootprintInflation < Policy_AStar_PD_Optimized
    properties
        footprintHeadingSamples
        footprintExtraClearance
        enableInflationFallback
        requireFootprintConnectivity
        footprintFallbackLocalmap
        biasOffset
        b2BiasOffset
    end

    methods
        function self = Policy_AStar_PD_FootprintInflation()
            self@Policy_AStar_PD_Optimized();
            self.footprintHeadingSamples = 16;
            self.footprintExtraClearance = 0;
            self.enableInflationFallback = true;
            self.requireFootprintConnectivity = true;
            self.footprintFallbackLocalmap = [];
            self.biasOffset = 0.4;
            self.b2BiasOffset = 1.6;
        end

        function path = A_star(self, observation)
            path = A_star@Policy_AStar_PD_Optimized(self, observation);
            if self.last_astar_failed && self.enableInflationFallback && ~isempty(self.footprintFallbackLocalmap)
                self.localmap = self.footprintFallbackLocalmap;
                path = A_star@Policy_AStar_PD_Optimized(self, observation);
            end
        end

        function expand_map_allknown(self, observation)
            if self.SystemInfo.smartAgent == 0
                sourceMap = observation.map;
            else
                sourceMap = self.now_smartmap;
            end

            sourceMask = sourceMap == 1;
            self.footprintFallbackLocalmap = self.inflate_square(self.localmap, sourceMask);
            footprintMap = self.inflate_footprint(self.localmap, sourceMask, observation);
            if self.should_use_fallback_map(footprintMap, observation)
                self.localmap = self.footprintFallbackLocalmap;
            else
                self.localmap = footprintMap;
            end
        end

        function expand_map_part(self, observation, flag)
            search_startx = floor(observation.agent.x - self.RangeFinderInfo.scanRange);
            search_endx = ceil(observation.agent.x + self.RangeFinderInfo.scanRange);
            search_starty = floor(observation.agent.y - self.RangeFinderInfo.scanRange);
            search_endy = ceil(observation.agent.y + self.RangeFinderInfo.scanRange);

            if search_startx < 2
                search_startx = 2;
            elseif search_endx > self.mapw
                search_endx = self.mapw;
            end
            if search_starty < 2
                search_starty = 2;
            elseif search_endy > self.maph
                search_endy = self.maph;
            end

            sourceMask = false(size(observation.scanMap));
            for i = search_startx:search_endx
                for j = search_starty:search_endy
                    if flag == 1 && observation.scanMap(i + 1, j + 1) ~= 0
                        sourceMask(i + 1, j + 1) = true;
                    elseif flag == 0 && observation.scanMap(i + 1, j + 1) == 1
                        sourceMask(i + 1, j + 1) = true;
                    end
                end
            end

            if flag == 1
                self.footprintFallbackLocalmap = self.inflate_square(self.localmap, sourceMask);
                footprintMap = self.inflate_footprint(self.localmap, sourceMask, observation);
                if self.should_use_fallback_map(footprintMap, observation)
                    self.localmap = self.footprintFallbackLocalmap;
                else
                    self.localmap = footprintMap;
                end
            else
                self.footprintFallbackLocalmap = self.inflate_square(self.storemap, sourceMask);
                footprintMap = self.inflate_footprint(self.storemap, sourceMask, observation);
                if self.should_use_fallback_map(footprintMap, observation)
                    self.storemap = self.footprintFallbackLocalmap;
                else
                    self.storemap = footprintMap;
                end
            end
        end

        function add_bias(self)
            self.add_bias_with_offset(self.biasOffset, self.b2BiasOffset);
        end

        function add_bias_with_offset(self, bias, b2Bias)
            len_path = length(self.path(:, 1));
            self.positions = zeros(len_path, 4);
            start_index1 = 1;
            end_index1 = 1;
            start_index2 = 1;
            end_index2 = 1;
            start_index3 = 1;
            end_index3 = 1;
            start_index4 = 1;
            end_index4 = 1;

            for i = 1:len_path
                self.positions(i, 1:2) = self.path(i, 1:2);
            end

            for i = 1:len_path
                x = self.positions(i, 1);
                y = self.positions(i, 2);
                if x > 1 && y > 1 && x < size(self.localmap, 1) && y < size(self.localmap, 2)
                    if self.localmap(x - 1, y) == 1
                        self.positions(i, 3) = 1;
                    end
                    if self.localmap(x, y - 1) == 1
                        self.positions(i, 3) = 2;
                    end
                    if self.localmap(x + 1, y) == 1 && self.SystemInfo.globalview == 0
                        self.positions(i, 3) = 3;
                    end
                    if self.localmap(x, y + 1) == 1 && self.SystemInfo.globalview == 0
                        self.positions(i, 3) = 4;
                    end
                end
            end

            for i = 2:len_path
                if self.positions(i, 3) == 1 && self.positions(i - 1, 3) ~= 1
                    start_index1 = i;
                end
                if self.positions(i, 3) ~= 1 && self.positions(i - 1, 3) == 1
                    end_index1 = i - 1;
                end
                if self.positions(i, 3) == 2 && self.positions(i - 1, 3) ~= 2
                    start_index2 = i;
                end
                if self.positions(i, 3) ~= 2 && self.positions(i - 1, 3) == 2
                    end_index2 = i - 1;
                end
                if self.positions(i, 3) == 3 && self.positions(i - 1, 3) ~= 3
                    start_index3 = i;
                end
                if self.positions(i, 3) ~= 3 && self.positions(i - 1, 3) == 3
                    end_index3 = i - 1;
                end
                if self.positions(i, 3) == 4 && self.positions(i - 1, 3) ~= 4
                    start_index4 = i;
                end
                if self.positions(i, 3) ~= 4 && self.positions(i - 1, 3) == 4
                    end_index4 = i - 1;
                end

                if end_index1 - start_index1 > 3
                    for j = start_index1:end_index1
                        if self.positions(j, 4) ~= 1
                            self.path(j, 1) = self.path(j, 1) + bias;
                            self.positions(j, 4) = 1;
                        end
                    end
                end
                if end_index2 - start_index2 > 3
                    for j = start_index2:end_index2
                        if self.positions(j, 4) ~= 2
                            self.path(j, 2) = self.path(j, 2) + bias;
                            self.positions(j, 4) = 2;
                        end
                    end
                end
                if end_index2 == start_index2 && self.positions(end_index2, 4) ~= 2
                    if strcmp(self.MapInfo.name, self.special) == 1 && self.SystemInfo.globalview == 1
                        self.path(end_index2, 2) = self.path(end_index2, 2) + b2Bias;
                        self.positions(end_index2, 4) = 2;
                    end
                end
                if end_index3 - start_index3 > 2
                    for j = start_index3:end_index3
                        if self.positions(j, 4) ~= 3
                            self.path(j, 1) = self.path(j, 1) + bias;
                            self.positions(j, 4) = 3;
                        end
                    end
                end
                if end_index4 - start_index4 > 4
                    for j = start_index4:end_index4
                        if self.positions(j, 4) ~= 4
                            self.path(j, 2) = self.path(j, 2) - bias;
                            self.positions(j, 4) = 4;
                        end
                    end
                end
            end
        end
    end

    methods (Access = protected)
        function inflatedMap = inflate_footprint(self, baseMap, sourceMask, observation)
            offsets = self.footprint_offsets(observation);
            inflatedMap = self.apply_offsets(baseMap, sourceMask, offsets);
        end

        function inflatedMap = inflate_square(self, baseMap, sourceMask)
            offsets = [-1, -1; -1, 0; -1, 1; 0, -1; 0, 0; 0, 1; 1, -1; 1, 0; 1, 1];
            inflatedMap = self.apply_offsets(baseMap, sourceMask, offsets);
        end

        function inflatedMap = apply_offsets(~, baseMap, sourceMask, offsets)
            inflatedMap = baseMap;
            [xs, ys] = find(sourceMask);
            if isempty(xs)
                return;
            end
            h = size(baseMap, 1);
            w = size(baseMap, 2);
            for k = 1:size(offsets, 1)
                xi = xs + offsets(k, 1);
                yi = ys + offsets(k, 2);
                valid = xi >= 1 & xi <= h & yi >= 1 & yi <= w;
                idx = sub2ind([h, w], xi(valid), yi(valid));
                inflatedMap(idx) = 1;
            end
        end

        function offsets = footprint_offsets(self, observation)
            xs = -observation.agent.w / 2:1:observation.agent.w / 2;
            ys = -observation.agent.ht / 2:1:observation.agent.ht / 2;
            [xGrid, yGrid] = meshgrid(xs, ys);
            body = [reshape(xGrid, 1, numel(xGrid)); reshape(yGrid, 1, numel(yGrid))];

            offsets = [];
            headings = linspace(0, pi, self.footprintHeadingSamples + 1);
            headings(end) = [];
            for i = 1:length(headings)
                c = cos(headings(i));
                s = sin(headings(i));
                rot = [c, -s; s, c];
                cells = ceil(rot * body);
                offsets = [offsets; -cells']; %#ok<AGROW>
            end
            offsets = unique(offsets, 'rows');

            extra = round(self.footprintExtraClearance);
            if extra > 0
                expanded = offsets;
                for dx = -extra:extra
                    for dy = -extra:extra
                        if hypot(dx, dy) <= extra + 0.01
                            expanded = [expanded; offsets + [dx, dy]]; %#ok<AGROW>
                        end
                    end
                end
                offsets = unique(expanded, 'rows');
            end
        end

        function useFallback = should_use_fallback_map(self, testMap, observation)
            useFallback = false;
            if ~self.requireFootprintConnectivity
                return;
            end

            startPos = [floor(observation.agent.x), floor(observation.agent.y)];
            goalPos = [floor(observation.endPos.x), floor(observation.endPos.y - 1)];
            if ~self.valid_planning_cell(startPos, testMap) || ~self.valid_planning_cell(goalPos, testMap)
                useFallback = true;
                return;
            end

            useFallback = ~self.has_grid_path(testMap, startPos, goalPos);
        end

        function ok = valid_planning_cell(self, pos, testMap)
            ok = pos(1) >= 1 && pos(2) >= 1 && pos(1) <= self.mapw && pos(2) <= self.maph;
            if ok
                ok = testMap(pos(1), pos(2)) ~= 1 && testMap(pos(1), pos(2)) ~= 2;
            end
        end

        function ok = has_grid_path(self, testMap, startPos, goalPos)
            visited = false(size(testMap));
            maxCells = numel(testMap);
            queue = zeros(maxCells, 2);
            head = 1;
            tail = 1;
            queue(tail, :) = startPos;
            visited(startPos(1), startPos(2)) = true;
            nextPos = [-1, 1; 0, 1; 1, 1; -1, 0; 1, 0; -1, -1; 0, -1; 1, -1];

            ok = false;
            while head <= tail
                current = queue(head, :);
                head = head + 1;
                if isequal(current, goalPos)
                    ok = true;
                    return;
                end

                for i = 1:size(nextPos, 1)
                    nx = current(1) + nextPos(i, 1);
                    ny = current(2) + nextPos(i, 2);
                    if nx < 1 || ny < 1 || nx > self.mapw || ny > self.maph
                        continue;
                    end
                    if visited(nx, ny) || testMap(nx, ny) == 1 || testMap(nx, ny) == 2
                        continue;
                    end
                    tail = tail + 1;
                    queue(tail, :) = [nx, ny];
                    visited(nx, ny) = true;
                end
            end
        end
    end
end
