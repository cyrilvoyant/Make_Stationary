%% =========================================================
% OPTIMIZATION — LagH vs Hidden neurons (PROJECTION ONLY)
% DATASET: PV POWER (Corse)
%% =========================================================
clear; clc; close all;

outdir = 'results_optimization_pv';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% =========================
% LOAD DATA (PV Power – Corse, Excel)
% =========================
T = readtable('Data.xlsx','VariableNamingRule','preserve');

series = T.("Solaire photovoltaïque (MW)");
series(series < 0) = 0;          % enforce physical constraint

annee = 6;    

%% =========================
% GRID DEFINITION 
% =========================
LagH_list = 6:6:48;           % hours [6 12 ... 48]
m_list    = 100:100:2000;     % neurons [100 200 ... 2000]

nL = numel(LagH_list);
nM = numel(m_list);

Opti_Statio = nan(nL, nM);

fprintf('PV Optimization grid: %d LagH × %d neurons = %d runs\n', ...
        nL, nM, nL*nM);

%% =========================
% GRID SEARCH (PROJECTION ONLY)
% =========================
tStart = tic;

for i = 1:nL
    LagH = LagH_list(i);
    fprintf('LagH = %d h\n', LagH);

    parfor j = 1:nM
        m = m_list(j);

        % === PROJECTION ONLY ===
        % S_r1 = aggregated PACF of residuals
        [~,~,~,~,~,~,~,S_r1,~,~] = ...
            Make_Stationary(series, annee, LagH, m);

        Opti_Statio(i,j) = S_r1;
    end
end

execTime = toc(tStart);
fprintf('\nPV optimization finished in %.1f seconds\n', execTime);

%% =========================
% FIND OPTIMUM
% =========================
M = Opti_Statio;
M(~isfinite(M)) = Inf;

[bestVal, idx] = min(M(:));
[iL, iM] = ind2sub(size(M), idx);

bestLagH = LagH_list(iL);
bestm    = m_list(iM);

fprintf('\n=== BEST PV CONFIGURATION ===\n');
fprintf('LagH = %d h | m = %d | PACFsum_proj = %.4f\n', ...
        bestLagH, bestm, bestVal);

%% =========================
% SAVE NUMERICAL RESULTS
% =========================
save(fullfile(outdir,'results_optimization_pv.mat'), ...
    'Opti_Statio','LagH_list','m_list', ...
    'bestLagH','bestm','bestVal','execTime');

% --- Save CSV (matrix)
Tmat = array2table(Opti_Statio, ...
    'VariableNames', compose('m_%d', m_list), ...
    'RowNames', compose('LagH_%d', LagH_list));
writetable(Tmat, fullfile(outdir,'PACFsum_projection_matrix_pv.csv'), ...
           'WriteRowNames',true);

% --- Save summary CSV
Summary = table(bestLagH, bestm, bestVal, execTime, ...
    'VariableNames',{'Best_LagH_hours','Best_m_neurons','Best_PACFsum_proj','ExecTime_s'});
writetable(Summary, fullfile(outdir,'Optimization_summary_pv.csv'));

%% =========================
% HEATMAP
% =========================
figure;
imagesc(m_list, LagH_list, Opti_Statio);
set(gca,'YDir','normal');
colorbar;
xlabel('Hidden neurons');
ylabel('LagH (hours)');
title('PACFsum (projection) — PV Corsica');
hold on;
plot(bestm, bestLagH, 'rp', 'MarkerSize', 14, 'MarkerFaceColor','w');
hold off;

saveas(gcf, fullfile(outdir,'PACFsum_heatmap_pv.png'));

%% =========================
% CONTOUR PLOT
% =========================
[X,Y] = meshgrid(m_list, LagH_list);

figure;
contourf(X, Y, Opti_Statio, 30, 'LineColor','none');
colorbar;
xlabel('Hidden neurons');
ylabel('LagH (hours)');
title('PACFsum (projection) — PV Corsica');
hold on;
plot(bestm, bestLagH, 'rp', 'MarkerSize', 14, 'MarkerFaceColor','w');
hold off;

saveas(gcf, fullfile(outdir,'PACFsum_contour_pv.png'));

fprintf('\n=== PV OPTIMIZATION + PLOTS SAVED SUCCESSFULLY ===\n');
