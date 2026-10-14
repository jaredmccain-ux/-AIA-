classdef Policy_Exp2_DWAHybrid < Policy_AStar_PD_Optimized
    methods
        function self = Policy_Exp2_DWAHybrid()
            self@Policy_AStar_PD_Optimized();
        end

        function [] = add_bias(self)
            safe_base_add_bias(self);
        end

        function u = dynamic_safety_adjust(self, observation, u)
            u = dynamic_safety_adjust@Policy_AStar_PD_Optimized(self, observation, u);
            u = dwa_hybrid_refine(self, observation, u);
        end
    end
end
