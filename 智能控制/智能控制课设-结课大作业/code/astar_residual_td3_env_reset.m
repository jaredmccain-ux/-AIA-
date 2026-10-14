function [features, info] = astar_residual_td3_env_reset(mapName, smartAgentNumber, maxSteps, randomMap, seed)
%ASTAR_RESIDUAL_TD3_ENV_RESET Reset Arena for A*+PD residual TD3 training.

global ASTAR_RL_ENV ASTAR_RL_POLICY ASTAR_RL_OBS ASTAR_RL_STEP ASTAR_RL_MAX_STEPS ASTAR_RL_OLD_CFG ASTAR_RL_LOG_FID

if nargin < 1 || isempty(mapName)
    mapName = 'amap';
end
if nargin < 2 || isempty(smartAgentNumber)
    smartAgentNumber = 20;
end
if nargin < 3 || isempty(maxSteps)
    maxSteps = 600;
end
if nargin < 4 || isempty(randomMap)
    randomMap = false;
end
if nargin >= 5 && ~isempty(seed)
    rng(double(seed), 'twister');
end

close all;
set(0, 'DefaultFigureVisible', 'off');

iniFile = fullfile(pwd, 'sys.ini');
if isempty(ASTAR_RL_OLD_CFG)
    ASTAR_RL_OLD_CFG = readConfig(iniFile);
end
writeConfig(iniFile, mapName, smartAgentNumber, 0, 0, randomMap);

if isempty(ASTAR_RL_LOG_FID) || ASTAR_RL_LOG_FID < 0
    logDir = fullfile(pwd, 'log');
    if ~exist(logDir, 'dir')
        mkdir(logDir);
    end
    logPath = fullfile(logDir, ['astar_residual_td3_env_' datestr(now, 'yyyymmdd_HHMMSS_FFF') '.txt']);
    ASTAR_RL_LOG_FID = fopen(logPath, 'a');
    fprintf(ASTAR_RL_LOG_FID, 'A*+PD residual TD3 environment log\ncreated=%s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF'));
end

ASTAR_RL_ENV = Env(iniFile);
if ~ASTAR_RL_ENV.succeed
    error('Failed to create Env from %s', iniFile);
end
ASTAR_RL_POLICY = Policy_AStarPD_ExternalResidual();
ASTAR_RL_OBS = ASTAR_RL_ENV.reset();
ASTAR_RL_STEP = 0;
ASTAR_RL_MAX_STEPS = maxSteps;

[features, baseAction, record] = ASTAR_RL_POLICY.computeBaseDecision(ASTAR_RL_OBS);
info = makeInfo(ASTAR_RL_OBS, ASTAR_RL_STEP, false, false, baseAction, baseAction, record.mode);
info.dynamic_risk = double(record.dynamicRisk);
info.residual_gate = double(record.residualGate);
astarResidualLogLine('reset', info);
end

function cfg = readConfig(iniFile)
I = INI('File', iniFile);
I.read();
cfg.map = I.get('Map');
cfg.system = I.get('System');
end

function writeConfig(iniFile, mapName, smartAgentNumber, showViewer, record, randomMap)
I = INI('File', iniFile);
I.read();
mapInfo = I.get('Map');
systemInfo = I.get('System');
mapInfo.random = double(logical(randomMap));
mapInfo.name = char(mapName);
systemInfo.smartAgent = double(smartAgentNumber);
systemInfo.showViewer = double(showViewer);
systemInfo.record = double(record);
I.set('Map', mapInfo);
I.set('System', systemInfo);
I.write();
end
