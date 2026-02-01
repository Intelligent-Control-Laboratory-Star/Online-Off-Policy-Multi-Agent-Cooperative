%% ====== Parameter area ======
file = 'lateral_error.xlsx';        % Excel files 
% file = 'beta.xlsx';               % Excel files 
% file = 'gamma.xlsx';              % Excel files 
% file = 'vx_real.xlsx';            % Excel files 
% file = 'lateral_acc.xlsx';        % Excel files  
% file = 'brake_torque.xlsx';       % Excel files 
% file = 'Triggering.xlsx';         % Excel files 

alpha = 0.05;           % 95% confidence interval
timeCol = 1;            % Time column position
dataCols = 2:11;        % Ten groups of duplicate data column positions (to be changed as needed)

for i=1
%% ====== Read-in data ======
T = readtable(file);

t = T{:, timeCol};          % Time(s)
Y = T{:, dataCols};         % N×10

% Basic inspection
if size(Y,2) < 2
    error('Insufficient number of repeated experiment columns: At least 2 columns of repeated data are required.');
end

% If the time may be out of order, sort it
[t, idx] = sort(t);
Y = Y(idx,:);

%% ====== Abnormal trajectory detection (column exclusion) ======
% Adjustable parameters
k = 4;                   % Robust z threshold: 3 to 4 is commonly used; the smaller it is, the stricter it is
fracBadThresh = 0.15;    % If the proportion of "out-of-threshold points" in a trajectory exceeds this value, it will be judged as abnormal (for example, 15%).
minValidFrac = 0.6;      % This trajectory is also regarded as unreliable if the proportion of valid points (non-nan) is too low

N = size(Y,1);
M = size(Y,2);

% Robust center/scale: Cross-repetition calculation based on "time points"
med = median(Y, 2, 'omitnan');                          % N×1
mad0 = median(abs(Y - med), 2, 'omitnan');              % N×1
sigma = 1.4826 * mad0;                                  % MAD is converted to a scale similar to standard deviation
sigma(sigma < eps) = NaN;                               % Prevent division by zero (or too small)

% Calculate the robust Z-score of each point on each curve
Z = abs((Y - med) ./ sigma);                            % N×M

% Column by column statistics: proportion of bad points, proportion of effective points
isBadPoint = Z > k;
validPoint = ~isnan(Y) & isfinite(Y);

badFrac = sum(isBadPoint & validPoint, 1) ./ max(sum(validPoint,1),1);
validFrac = sum(validPoint,1) / N;

% Determine abnormal columns
badCol = (badFrac > fracBadThresh) | (validFrac < minValidFrac);

fprintf('Outlier trajectories: %s\n', mat2str(find(badCol)));

% Selection: Exclude abnormal columns (not participating in statistics & not drawing)
Y_clean = Y(:, ~badCol);
Y_plot  = Y(:, ~badCol);  

%  (optional) Check if it is at equal intervals (100Hz => dt=0.01)
dt = median(diff(t));
fprintf('Estimated dt = %.6g s, fs = %.3f Hz\n', dt, 1/dt);

%% ====== Repeat the statistics 10 times across "each time point" ======
% n(t) : Number of valid repetitions at this moment (excluding NaN)
n = sum(~isnan(Y_clean ), 2);                 % N×1

% Mean value: Across duplicates (take the mean value along the column direction)
mu = mean(Y_clean , 2, 'omitnan');            % N×1

% Variance: Across duplicates (sample variance, divided by n-1
v  = var(Y_clean , 0, 2, 'omitnan');          % N×1
sd = sqrt(v);

% Mean standard error SE = sd/sqrt(n)
se = sd ./ sqrt(n);

% The critical value of t (degrees of freedom df = n-1 at each time point)
df = n - 1;
tcrit = tinv(1 - alpha/2, df);

% Confidence interval of the mean (moment by moment)
ciLo = mu - tcrit .* se;
ciHi = mu + tcrit .* se;

% When n<2, the variance /CI cannot be estimated. Let them be NaN (to avoid misleading)
bad = (n < 2) | ~isfinite(tcrit);
v(bad)    = NaN;
sd(bad)   = NaN;
se(bad)   = NaN;
ciLo(bad) = NaN;
ciHi(bad) = NaN;

%% ====== Plotting: 10 repetitions + mean curve + 95% confidence band ======
figure; hold on

plot(t, Y_plot, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.8);

% Confidence band (Skip the NaN section)
ok = isfinite(t) & isfinite(ciLo) & isfinite(ciHi);
x = t(ok); y1 = ciLo(ok); y2 = ciHi(ok);
patch([x; flipud(x)], [y1; flipud(y2)], 'b', ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none');

% Mean line
plot(t, mu, 'b', 'LineWidth', 2);

grid on
xlabel('Time (s)');
ylabel('Value');
title(sprintf('Mean across %d repeats with 95%% CI', size(Y,2)));
legend([repmat("Repeat",1,size(Y,2)) "95% CI" "Mean"], 'Location','bestoutside');
end
%% ====== Export the results to a table /Excel (optional) ======
% Out = table(t, mu, v, sd, n, ciLo, ciHi, ...
%     'VariableNames', {'time_s','mean','variance','std','n','ci_low','ci_high'});
% 假设你已有 t, Y(=N×5), mu, ciLo, ciHi
% ciHalf = (ciHi - ciLo)/2;
% 
% Out = table(t, Y(:,1), Y(:,2), Y(:,3), Y(:,4), Y(:,5), Y(:,6), Y(:,7), Y(:,8), Y(:,9), Y(:,10), mu, ciLo, ciHi, ciHalf, ...
%     'VariableNames', {'time_s','rep1','rep2','rep3','rep4','rep5', 'rep6','rep7','rep8','rep9','rep10','mean','ci_low','ci_high','ci_half'});
% 
% writetable(Out, 'A.ADET_origin_plot_data_triggerFix_RF_Triggering.xlsx');

% 写回Excel
% writetable(Out, 'time_stats.xlsx');
