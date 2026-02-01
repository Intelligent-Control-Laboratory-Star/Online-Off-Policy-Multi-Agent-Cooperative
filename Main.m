%% Run Simulink model 10 times and export signals to Excel files

clear; clc;
warning('off','all')

% model = 'UniTire_0427_2021b_PID';
model = 'UniTire_0427_2021b_MPC';
% model = 'UniTire_0427_2021b_SAC';
% model = 'UniTire_0427_2021b_MAC';
% model = 'UniTire_0427_2021b_ET';
% model = 'UniTire_0427_2021b_DET';
% model = 'UniTire_0427_2021b_ADET';

nRuns = 10;

sigNames = { ...
    'lateral_error', ...
    'beta', ...
    'gamma', ...
    'vx_real', ...
    'lateral_acc',...
    'Steering',...
    'Brake_torque1'};

outFiles = { ...
    'Results_lateral_error.xlsx', ...
    'Results_beta.xlsx', ...
    'Results_gamma.xlsx', ...
    'Results_vx_real.xlsx', ...
    'Results_lateral_acc.xlsx', ...
    'Results_Steering.xlsx',...
    'Results_Brake_torque1.xlsx'};

useFastRestart = false;  % If the model does not support it, you can change it to false

% ====== NEW: create output folder named as model ======
outDir = fullfile(pwd, model);   % Save to the folder with the same name as the model
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
% ======================================================

% Container: TT{s,r} is the timetable of the RTH simulation of this signal (the column name includes run and channel).
TT = cell(numel(sigNames), nRuns);

%% load model
load_system(model);

if useFastRestart
    try
        set_param(model,'FastRestart','on');
    catch
        warning('FastRestart cannot be enabled for this model. Continue without FastRestart.');
        useFastRestart = false;
    end
end

%% Cyclic simulation
for r = 1:nRuns
    fprintf('Running simulation %d/%d...\n', r, nRuns);

    in = Simulink.SimulationInput(model);

    % If you need to change the parameters each time you conduct an experiment, add them here：
    % in = in.setVariable('seed', r);

    out = sim(in);

    % Collect all signals
    for s = 1:numel(sigNames)
        [t, Y] = localGetSignal_ToWorkspace(out, sigNames{s}); % t: Nx1, Y: NxC

        % Generate column names: run{r} or run{r} ch{c}
        C = size(Y,2);
        if C == 1
            vnames = {sprintf('run%d', r)};
        else
            vnames = arrayfun(@(c) sprintf('run%d_ch%d', r, c), 1:C, 'UniformOutput', false);
        end

        TT{s,r} = array2timetable(Y, ...
            'RowTimes', seconds(t(:)), ...
            'VariableNames', vnames);
    end
end

if useFastRestart
    set_param(model,'FastRestart','off');
end

%% Export as Excel files (to outDir)
for s = 1:numel(sigNames)
    ttMerged = TT{s,1};
    for r = 2:nRuns
        ttMerged = synchronize(ttMerged, TT{s,r}, 'union', 'linear');
    end

    time_s = seconds(ttMerged.Time);

    outTab = timetable2table(ttMerged, 'ConvertRowTimes', true);
    outTab.Properties.VariableNames{1} = 'Time';

    outTab.time_s = time_s;
    outTab.Time = [];
    outTab = movevars(outTab, 'time_s', 'Before', 1);

    fn = fullfile(outDir, outFiles{s});   % ====== CHANGED: save into model folder ======
    writetable(outTab, fn);
    fprintf('Saved: %s\n', fn);
end

fprintf('All done.\n');

%% ====== Local function: Extract the signal from the To Workspace variable ======
function [t, Y] = localGetSignal_ToWorkspace(out, name)

    sig = [];

    % out.(name)
    if isprop(out, name) || isfield(out, name)
        try
            sig = out.(name);
        catch
        end
    end

    % out.get(name)
    if isempty(sig)
        try
            sig = out.get(name);
        catch
        end
    end

    if isempty(sig)
        error('Cannot find To Workspace variable "%s" in SimulationOutput out. Check To Workspace variable name.', name);
    end

    % timeseries
    if isa(sig, 'timeseries')
        t = double(sig.Time(:));
        Y = sig.Data;

    % old style struct: Structure with time
    elseif isstruct(sig) && isfield(sig,'time') && isfield(sig,'signals') && isfield(sig.signals,'values')
        t = double(sig.time(:));
        Y = sig.signals.values;

    % numeric vector/matrix (no time)
    elseif isnumeric(sig)
        Y = sig;
        try
            t = double(out.tout(:));
        catch
            error('"%s" is numeric but out.tout not found. Use To Workspace with time or enable tout.', name);
        end

    else
        error('Unsupported To Workspace type for "%s": %s', name, class(sig));
    end

    % Ensure Y is numeric 2D [N x C]
    Y = double(squeeze(Y));
    if isvector(Y)
        Y = Y(:); % Nx1
    end

    % Basic length check
    if size(Y,1) ~= numel(t)
        % Sometimes data comes as CxN
        if size(Y,2) == numel(t) && size(Y,1) ~= numel(t)
            Y = Y.'; % transpose to NxC
        else
            error('Length mismatch for "%s": length(t)=%d, size(Y)=[%d %d].', ...
                name, numel(t), size(Y,1), size(Y,2));
        end
    end
end
