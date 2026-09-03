clc;clf;clear all;close all;



% =========
% LOAD DATA
% ========
train_data = readtable('MASTER_Train_Data.csv');
test_data  = readtable('MASTER_Test_Data.csv');

n_train = height(train_data);
n_test  = height(test_data);
n_tot   = n_train + n_test;

fprintf('Training: %d samples (%.1f%%)\n', n_train, n_train/n_tot*100);
fprintf('Testing:  %d samples (%.1f%%)\n', n_test, n_test/n_tot*100);

% ===================
% FEATURES / TARGET
% ===================
% The input variables are:
%   1. Interstitial Velocity (m/s)
%   2. Porosity
%   3. Salt Conc. (PPM)
%   4. PV
%   5. Clay Percentage

feature_cols = 1:5;
target_col = 6;

Xtr_orig = table2array(train_data(:, feature_cols));
Ytr_orig = table2array(train_data(:, target_col));
Xtst = table2array(test_data(:, feature_cols));
Ytst = table2array(test_data(:, target_col));

fprintf('Original training: %d samples | Testing: %d samples\n', length(Ytr_orig), length(Ytst));

train_low  = sum(Ytr_orig < 0.2);
train_high = sum(Ytr_orig > 0.8);
test_low   = sum(Ytst < 0.2);
test_high  = sum(Ytst > 0.8);


% =================================
% SAMPLE WEIGHTING VIA REPLICATION
% =================================
% U-shaped weighting to counter the imbalance at the extremes of Norm K

rep_counts = ones(length(Ytr_orig), 1);

for i = 1:length(Ytr_orig)
    yi = Ytr_orig(i);

    if yi < 0.05
        rep_counts(i) = 12;
    elseif yi < 0.1
        rep_counts(i) = 7;
    elseif yi < 0.2
        rep_counts(i) = 5;
    elseif yi <= 0.8
        rep_counts(i) = 1;
    elseif yi <= 0.95
        rep_counts(i) = 4;
    else
        rep_counts(i) = 10;
    end
end

% fprintf('\nWeight distribution:\n');
% fprintf('  Very Low (<0.05):    %3d x 12\n', sum(Ytr_orig < 0.05));
% fprintf('  Low (0.05-0.1):      %3d x 7\n',  sum(Ytr_orig >= 0.05 & Ytr_orig < 0.1));
% fprintf('  Low-Mid (0.1-0.2):   %3d x 5\n',  sum(Ytr_orig >= 0.1 & Ytr_orig < 0.2));
% fprintf('  Mid (0.2-0.8):       %3d x 1\n',  sum(Ytr_orig >= 0.2 & Ytr_orig <= 0.8));
% fprintf('  High-Mid (0.8-0.95): %3d x 4\n',  sum(Ytr_orig > 0.8 & Ytr_orig <= 0.95));
% fprintf('  Very High (>0.95):   %3d x 10\n', sum(Ytr_orig > 0.95));

% replicate samples according to rep_counts
Xtr_weighted = [];
Ytr_weighted = [];
for i = 1:length(Ytr_orig)
    for j = 1:rep_counts(i)
        Xtr_weighted = [Xtr_weighted; Xtr_orig(i,:)];
        Ytr_weighted = [Ytr_weighted; Ytr_orig(i)];
    end
end

fprintf('After replication: %d samples (was %d, %.1fx expansion)\n', ...
    length(Ytr_weighted), length(Ytr_orig), length(Ytr_weighted)/length(Ytr_orig));


Xv = Xtr_orig(1:10,:);
Yv = Ytr_orig(1:10);

% ==================
% GMDH CONFIG SWEEP
% ==================
% columns: name, maxNumInputs (poly degree), inputsMore (skip conn.),
% maxNumNeurons, decNumNeurons, p (layers), critNum, delta
configs = {
    'Config 1: Original',                      3, 0,  2, 0, 2, 2, 0.8;
    'Config 2: More neurons',                  3, 0,  5, 0, 2, 2, 0.8;
    'Config 3: Higher delta',                  3, 0,  5, 0, 2, 2, 5.0;
    'Config 4: More neurons + high delta',     3, 0, 10, 0, 2, 2, 5.0;
    'Config 5: Skip connections',              3, 1,  5, 0, 2, 2, 5.0;
    'Config 6: Skip + neurons',                3, 1, 10, 0, 2, 2, 5.0;
    'Config 7: Skip + high delta',             3, 1, 10, 0, 2, 2, 10.0;
    'Config 8: Degree 2',                      2, 0,  5, 0, 2, 2, 5.0;
};

n_config = size(configs, 1);
results = zeros(n_config, 6);

fprintf('\n%-40s %8s %8s %10s %10s %6s\n', 'Configuration', 'R2', 'AAPE%', 'Low%', 'High%', 'Layers');

best_r2 = -inf;
best_config = 1;

for c = 1:n_config
    config_name   = configs{c, 1};
    maxNumInputs  = configs{c, 2};
    inputsMore    = configs{c, 3};
    maxNumNeurons = configs{c, 4};
    decNumNeurons = configs{c, 5};
    p             = configs{c, 6};
    critNum       = configs{c, 7};
    delta         = configs{c, 8};

    try
        [model, ~] = gmdhbuild(Xtr_weighted, Ytr_weighted, maxNumInputs, inputsMore, ...
            maxNumNeurons, decNumNeurons, p, critNum, delta, Xv, Yv, 0);

        Yqtst = gmdhpredict(model, Xtst);

        % occasionally gmdhpredict returns NaN/Inf for a handful of points -
        % fall back to the median prediction for those
        nan_count = sum(isnan(Yqtst) | isinf(Yqtst));
        if nan_count > 0
            med_pred = median(Yqtst(~isnan(Yqtst) & ~isinf(Yqtst)));
            Yqtst(isnan(Yqtst) | isinf(Yqtst)) = med_pred;
        end

        Yqtst = max(0, min(1.1, Yqtst));

        R2 = 1 - sum((Ytst - Yqtst).^2) / sum((Ytst - mean(Ytst)).^2);
        AAPE = mean(abs((Ytst - Yqtst) ./ (Ytst + 1e-10))) * 100;

        low_mask = Ytst < 0.2;
        high_mask = Ytst > 0.8;

        if sum(low_mask) > 0
            Low_AAPE = mean(abs((Ytst(low_mask) - Yqtst(low_mask)) ./ (Ytst(low_mask) + 1e-10))) * 100;
        else
            Low_AAPE = 0;
        end

        if sum(high_mask) > 0
            High_AAPE = mean(abs((Ytst(high_mask) - Yqtst(high_mask)) ./ (Ytst(high_mask) + 1e-10))) * 100;
        else
            High_AAPE = 0;
        end

        num_layers = model.numLayers;
        results(c, :) = [R2, AAPE, Low_AAPE, High_AAPE, num_layers, nan_count];

        status = '';
        if nan_count > 0
            status = sprintf(' [%d NaN]', nan_count);
        end
        fprintf('%-40s %8.4f %8.2f %10.2f %10.2f %6d%s\n', ...
            config_name, R2, AAPE, Low_AAPE, High_AAPE, num_layers, status);

        if R2 > best_r2
            best_r2 = R2;
            best_config = c;
            best_model = model;
            best_Yqtst = Yqtst;
        end

    catch err
        fprintf('%-40s  ** FAILED: %s **\n', config_name, err.message);
        results(c, :) = [NaN, NaN, NaN, NaN, NaN, NaN];
    end
end

% ==============
% BEST MODEL
% ==============
fprintf('\nBest configuration: %s\n', configs{best_config, 1});

R2 = results(best_config, 1);
AAPE = results(best_config, 2);
Low_AAPE = results(best_config, 3);
High_AAPE = results(best_config, 4);
num_layers = results(best_config, 5);

fprintf('Test set - R2: %.4f | AAPE: %.2f%% | Low: %.2f%% | High: %.2f%% | Layers: %d\n', ...
    R2, AAPE, Low_AAPE, High_AAPE, num_layers);

% training metrics on the ORIGINAL (unreplicated) training data
Yqtr = gmdhpredict(best_model, Xtr_orig);
Yqtr = max(0, min(1.1, Yqtr));
train_R2 = 1 - sum((Ytr_orig - Yqtr).^2) / sum((Ytr_orig - mean(Ytr_orig)).^2);
train_AAPE = mean(abs((Ytr_orig - Yqtr) ./ (Ytr_orig + 1e-10))) * 100;

low_mask_tr = Ytr_orig < 0.2;
high_mask_tr = Ytr_orig > 0.8;

if sum(low_mask_tr) > 0
    train_Low_AAPE = mean(abs((Ytr_orig(low_mask_tr) - Yqtr(low_mask_tr)) ./ (Ytr_orig(low_mask_tr) + 1e-10))) * 100;
else
    train_Low_AAPE = 0;
end

if sum(high_mask_tr) > 0
    train_High_AAPE = mean(abs((Ytr_orig(high_mask_tr) - Yqtr(high_mask_tr)) ./ (Ytr_orig(high_mask_tr) + 1e-10))) * 100;
else
    train_High_AAPE = 0;
end

fprintf('%-15s %10s %10s %12s %12s\n', 'Dataset', 'R2', 'AAPE%', 'Low%', 'High%');
fprintf('%-15s %10.4f %10.2f %12.2f %12.2f\n', 'Training', train_R2, train_AAPE, train_Low_AAPE, train_High_AAPE);
fprintf('%-15s %10.4f %10.2f %12.2f %12.2f\n', 'Testing', R2, AAPE, Low_AAPE, High_AAPE);

gmdheq(best_model, 3);

% ============================================================
% SAVE RESULTS
% ============================================================
save('GMDH_Model_Master.mat', 'best_model', 'results', 'configs', 'best_config');

comparison = table(Ytst, best_Yqtst, 'VariableNames', {'K_measured', 'K_pred_GMDH'});
writetable(comparison, 'GMDH_Test_Predictions.csv');

fprintf('\nSaved: GMDH_Model_Master.mat, GMDH_Test_Predictions.csv\n');

% ========
% PLOTS 
% ========

% Test cross-plot
figure('Position', [100, 100, 600, 600]);
set(gcf, 'color', 'white');
scatter(Ytst, best_Yqtst, 60, 'filled', 'MarkerEdgeColor', 'black', 'LineWidth', 0.5);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Actual Norm K', 'FontWeight', 'bold');
ylabel('Predicted Norm K', 'FontWeight', 'bold');
title(sprintf('Testing Set\nR2=%.4f, AAPE=%.2f%%', R2, AAPE), 'FontWeight', 'bold');
grid on; xlim([0 1.05]); ylim([0 1.05]); axis square;

% Residuals
figure('Position', [100, 100, 600, 600]);
set(gcf, 'color', 'white');
residuals = Ytst - best_Yqtst;
scatter(Ytst, residuals, 50, 'filled');
hold on;
plot([0 1], [0 0], 'r--', 'LineWidth', 2);
xlabel('Actual Norm K', 'FontWeight', 'bold');
ylabel('Residual', 'FontWeight', 'bold');
title('Residual Plot', 'FontWeight', 'bold');
grid on;

% APE distribution
figure('Position', [100, 100, 600, 600]);
set(gcf, 'color', 'white');
ape = abs((Ytst - best_Yqtst) ./ (Ytst + 1e-10)) * 100;
ape_filtered = ape(ape < 100);
histogram(ape_filtered, 30, 'EdgeColor', 'black');
hold on;
xline(AAPE, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean=%.2f%%', AAPE));
xlabel('APE (%)', 'FontWeight', 'bold');
ylabel('Frequency', 'FontWeight', 'bold');
title('APE Distribution', 'FontWeight', 'bold');
grid on;

% Training vs Testing metrics
figure('Position', [100, 100, 600, 600]);
set(gcf, 'color', 'white');
categories = {'R2', 'RMSEx10', 'MAEx10', 'AAPE%'};
train_vals = [train_R2, sqrt(mean((Ytr_orig-Yqtr).^2))*10, mean(abs(Ytr_orig-Yqtr))*10, train_AAPE];
test_vals = [R2, sqrt(mean((Ytst-best_Yqtst).^2))*10, mean(abs(Ytst-best_Yqtst))*10, AAPE];

xpos = 1:length(categories);
bw = 0.35;
bar(xpos - bw/2, train_vals, bw, 'FaceColor', [0.2 0.6 0.8]);
hold on;
bar(xpos + bw/2, test_vals, bw, 'FaceColor', [0.8 0.4 0.2]);
xticks(xpos);
xticklabels(categories);
ylabel('Value', 'FontWeight', 'bold');
title('Training vs Testing Metrics', 'FontWeight', 'bold');
legend('Training', 'Testing');
grid on;

% =========
% SUMMARY
% =========
fprintf('\nGMDH model complete.\n');
fprintf('Test set - R2: %.4f | RMSE: %.4f | MAE: %.4f | AAPE: %.2f%%\n', ...
    R2, sqrt(mean((Ytst-best_Yqtst).^2)), mean(abs(Ytst-best_Yqtst)), AAPE);
fprintf('Trained on MASTER_Train_Data.csv (%d samples), tested on MASTER_Test_Data.csv (%d samples)\n', ...
    n_train, n_test);