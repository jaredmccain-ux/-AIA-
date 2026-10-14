function results = eval_policy_class_maps(policyClassName, smartAgentNumber, mapNames, seedBase, outCsv)
if nargin < 1 || isempty(policyClassName)
    policyClassName = 'Policy';
end
if isstring(policyClassName)
    policyClassName = char(policyClassName);
end
if nargin < 2 || isempty(smartAgentNumber)
    smartAgentNumber = 20;
end
if nargin < 3 || isempty(mapNames)
    mapNames = {'amap', 'a1map', 'a2map', 'bmap', 'b1map', 'b2map', 'cmap', 'c1map'};
end
if nargin < 4 || isempty(seedBase)
    seedBase = 20240527;
end
if nargin < 5
    outCsv = '';
end
if isstring(outCsv)
    outCsv = char(outCsv);
end

close all;

iniFile = 'sys.ini';
oldDefaultFigureVisible = get(0, 'DefaultFigureVisible');
set(0, 'DefaultFigureVisible', 'off');
[oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord] = configureSystem(iniFile, smartAgentNumber, 0, 0, mapNames{1});
cleanup = onCleanup(@() cleanupEval(iniFile, oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord, oldDefaultFigureVisible));

results = zeros(length(mapNames), 8);

for i = 1:length(mapNames)
    configureMap(iniFile, mapNames{i});
    rng(seedBase + i - 1, 'twister');
    env = Env(iniFile);
    if ~env.succeed
        error('Failed to create Env from %s on map %s', iniFile, mapNames{i});
    end

    policy = feval(policyClassName);
    policy.setTraining(false);
    observation = env.reset();
    done = 0;
    steps = 0;

    while ~done && steps < 600
        action = policy.action(observation);
        [observation, done, ~] = env.step(action);
        steps = steps + 1;
    end
    policy.finishEpisode(observation);

    score = observation.score;
    t = observation.t;
    x = observation.agent.x;
    y = observation.agent.y;
    reached = abs(x - observation.endPos.x) < 1 && abs(y - observation.endPos.y) < 1;
    results(i, :) = [i, score, t, steps, reached, x, y, seedBase + i - 1];

    disp(['policy=' policyClassName ...
        ' map=' mapNames{i} ...
        ' score=' num2str(score) ...
        ' time=' num2str(t) ...
        ' steps=' num2str(steps) ...
        ' reached=' num2str(reached) ...
        ' pos=' num2str(x) ',' num2str(y)]);
end

disp(['policy=' policyClassName ...
    ' maps=' num2str(length(mapNames)) ...
    ' smartAgent=' num2str(smartAgentNumber) ...
    ' seedBase=' num2str(seedBase) ...
    ' mean score=' num2str(mean(results(:, 2))) ...
    ' min=' num2str(min(results(:, 2))) ...
    ' best=' num2str(max(results(:, 2))) ...
    ' mean time=' num2str(mean(results(:, 3))) ...
    ' reached=' num2str(sum(results(:, 5))) '/' num2str(length(mapNames))]);

if ~isempty(outCsv)
    writeResults(outCsv, policyClassName, smartAgentNumber, mapNames, results);
end
end

function [oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord] = configureSystem(iniFile, smartAgentNumber, showViewer, record, mapName)
I = INI('File', iniFile);
I.read();
mapInfo = I.get('Map');
systemInfo = I.get('System');
oldMapName = mapInfo.name;
oldRandom = mapInfo.random;
oldSmartAgent = systemInfo.smartAgent;
oldShowViewer = systemInfo.showViewer;
oldRecord = systemInfo.record;
mapInfo.random = 0;
mapInfo.name = mapName;
systemInfo.smartAgent = smartAgentNumber;
systemInfo.showViewer = showViewer;
systemInfo.record = record;
I.set('Map', mapInfo);
I.set('System', systemInfo);
I.write();
end

function configureMap(iniFile, mapName)
I = INI('File', iniFile);
I.read();
mapInfo = I.get('Map');
mapInfo.random = 0;
mapInfo.name = mapName;
I.set('Map', mapInfo);
I.write();
end

function cleanupEval(iniFile, oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord, oldDefaultFigureVisible)
restoreSystem(iniFile, oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord);
set(0, 'DefaultFigureVisible', oldDefaultFigureVisible);
end

function restoreSystem(iniFile, oldMapName, oldRandom, oldSmartAgent, oldShowViewer, oldRecord)
I = INI('File', iniFile);
I.read();
mapInfo = I.get('Map');
systemInfo = I.get('System');
mapInfo.name = oldMapName;
mapInfo.random = oldRandom;
systemInfo.smartAgent = oldSmartAgent;
systemInfo.showViewer = oldShowViewer;
systemInfo.record = oldRecord;
I.set('Map', mapInfo);
I.set('System', systemInfo);
I.write();
end

function writeResults(outCsv, policyClassName, smartAgentNumber, mapNames, results)
fid = fopen(outCsv, 'w');
if fid < 0
    error('Unable to write result file: %s', outCsv);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'policy,smartAgent,map,score,time,steps,reached,x,y,seed\n');
for i = 1:length(mapNames)
    fprintf(fid, '%s,%d,%s,%.6f,%.6f,%d,%d,%.6f,%.6f,%d\n', ...
        policyClassName, smartAgentNumber, mapNames{i}, results(i, 2), results(i, 3), ...
        round(results(i, 4)), round(results(i, 5)), results(i, 6), results(i, 7), round(results(i, 8)));
end
end
