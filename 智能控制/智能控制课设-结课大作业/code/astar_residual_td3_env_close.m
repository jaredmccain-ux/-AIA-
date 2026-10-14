function astar_residual_td3_env_close()
%ASTAR_RESIDUAL_TD3_ENV_CLOSE Restore sys.ini after A* residual TD3 training.

global ASTAR_RL_ENV ASTAR_RL_POLICY ASTAR_RL_OBS ASTAR_RL_STEP ASTAR_RL_MAX_STEPS ASTAR_RL_OLD_CFG ASTAR_RL_LOG_FID

iniFile = fullfile(pwd, 'sys.ini');
if ~isempty(ASTAR_RL_OLD_CFG)
    I = INI('File', iniFile);
    I.read();
    I.set('Map', ASTAR_RL_OLD_CFG.map);
    I.set('System', ASTAR_RL_OLD_CFG.system);
    I.write();
end

if ~isempty(ASTAR_RL_LOG_FID) && ASTAR_RL_LOG_FID >= 0
    fprintf(ASTAR_RL_LOG_FID, 'closed=%s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF'));
    fclose(ASTAR_RL_LOG_FID);
end

ASTAR_RL_ENV = [];
ASTAR_RL_POLICY = [];
ASTAR_RL_OBS = [];
ASTAR_RL_STEP = [];
ASTAR_RL_MAX_STEPS = [];
ASTAR_RL_OLD_CFG = [];
ASTAR_RL_LOG_FID = [];
close all;
end
