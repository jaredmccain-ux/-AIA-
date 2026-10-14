classdef Policy_AStar_PD_FootprintDisk < Policy_AStar_PD_FootprintInflation
    properties
        footprintDiskPadding
    end

    methods
        function self = Policy_AStar_PD_FootprintDisk()
            self@Policy_AStar_PD_FootprintInflation();
            self.footprintDiskPadding = 0.25;
        end
    end

    methods (Access = protected)
        function offsets = footprint_offsets(self, observation)
            radius = 0.5 * hypot(observation.agent.w, observation.agent.ht) + self.footprintDiskPadding;
            maxCell = ceil(radius);
            offsets = [];
            for dx = -maxCell:maxCell
                for dy = -maxCell:maxCell
                    if hypot(dx, dy) <= radius + 1e-6
                        offsets = [offsets; dx, dy]; %#ok<AGROW>
                    end
                end
            end
            offsets = unique(offsets, 'rows');
        end
    end
end
