%% ====== Parameter area ======
files = { ...
    'Results_beta.xlsx', ...
    'Results_gamma.xlsx', ...
    'Results_vx_real.xlsx', ...
    'Results_lateral_acc.xlsx', ...
    'Results_Brake_torque1.xlsx', ...
    'Results_Steering.xlsx', ...
    'Results_lateral_error.xlsx' ...
    };

alpha = 0.05;           % 95% confidence interval
timeCol = 1;            % Time column position
dataCols = 2:11;        % Ten groups

% Abnormal trajectory detection params
k = 4;
fracBadThresh = 0.15;
minValidFrac = 0.6;

for f = 1:numel(files)

    file = files{f};
    fprintf('\n===== Processing: %s =====\n', file);

    %% ====== Read-in data ======
    T = readtable(file);

    t = T{:, timeCol};
    Y = T{:, dataCols};         % N×10

    if size(Y,2) < 2
        error('Insufficient repeats in %s (need >=2).', file);
    end

    % sort by time
    [t, idx] = sort(t);
    Y = Y(idx,:);

    %% ====== Abnormal trajectory detection ======
    N = size(Y,1);

    med = median(Y, 2, 'omitnan');
    mad0 = median(abs(Y - med), 2, 'omitnan');
    sigma = 1.4826 * mad0;
    sigma(sigma < eps) = NaN;

    Z = abs((Y - med) ./ sigma);

    isBadPoint = Z > k;
    validPoint = ~isnan(Y) & isfinite(Y);

    badFrac   = sum(isBadPoint & validPoint, 1) ./ max(sum(validPoint,1),1);
    validFrac = sum(validPoint,1) / N;

    badCol = (badFrac > fracBadThresh) | (validFrac < minValidFrac);

    fprintf('Outlier trajectories (cols): %s\n', mat2str(find(badCol)));

    Y_clean = Y(:, ~badCol);
    Y_plot  = Y(:, ~badCol);

    dt = median(diff(t));
    fprintf('Estimated dt = %.6g s, fs = %.3f Hz\n', dt, 1/dt);

    %% ====== Statistics ======
    n  = sum(~isnan(Y_clean), 2);
    mu = mean(Y_clean, 2, 'omitnan');
    v  = var(Y_clean, 0, 2, 'omitnan');
    sd = sqrt(v);
    se = sd ./ sqrt(n);

    df = n - 1;
    tcrit = tinv(1 - alpha/2, df);

    ciLo = mu - tcrit .* se;
    ciHi = mu + tcrit .* se;

    bad = (n < 2) | ~isfinite(tcrit);
    v(bad) = NaN; sd(bad) = NaN; se(bad) = NaN; ciLo(bad) = NaN; ciHi(bad) = NaN;

    %% ====== Plot ======
    figure('Name', file); hold on
    plot(t, Y_plot, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.8);

    ok = isfinite(t) & isfinite(ciLo) & isfinite(ciHi);
    x = t(ok); y1 = ciLo(ok); y2 = ciHi(ok);
    patch([x; flipud(x)], [y1; flipud(y2)], 'b', 'FaceAlpha', 0.18, 'EdgeColor', 'none');

    plot(t, mu, 'b', 'LineWidth', 2);

    grid on
    xlabel('Time (s)');
    ylabel('Value');
    title(sprintf('%s | mean across %d repeats with 95%% CI', file, size(Y,2)));

    legStr = [repmat("Repeat", 1, size(Y_plot,2)) "95% CI" "Mean"];
    legend(legStr, 'Location','bestoutside');

    %% ====== Export to Excel ======
    ciHalf = (ciHi - ciLo)/2;

    % 如果你的文件不一定都有10列重复数据，这里建议按 size(Y,2) 自动生成 rep 列；
    % 但你当前固定rep1~rep10，我先保持原样（要求Y必须是10列）。
    Out = table(t, Y(:,1), Y(:,2), Y(:,3), Y(:,4), Y(:,5), Y(:,6), Y(:,7), Y(:,8), Y(:,9), Y(:,10), ...
        mu, ciLo, ciHi, ciHalf, ...
        'VariableNames', {'time_s','rep1','rep2','rep3','rep4','rep5','rep6','rep7','rep8','rep9','rep10', ...
                          'mean','ci_low','ci_high','ci_half'});

    newFile = ['MPC_origin_plot_data_LR_' file];
    writetable(Out, newFile);

    fprintf('Saved: %s\n', newFile);
end
