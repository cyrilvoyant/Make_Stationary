%% =========================================================
%  STATISTICAL COMPARISON VS PERSISTENCE
%  Multi-variable / Multi-horizon / Multi-metric
%% =========================================================

clear; clc; close all;

%% ================= VARIABLES =============================
Variables = {'PV','WT','PT','WS','T2M','GHI','GHI_15min'}; 
Horizons  = 1:6;

%% ================= BASE PATH =============================
base_path = 'C:/Users/HP/Downloads/Prof_Cyril Voyant_Proposition_Collaboration/Maklewa';

%% ================= OUTPUT ===============================
Results = {};

%% =========================================================
% ================= LOOP OVER VARIABLES ====================
%% =========================================================
for v = 1:numel(Variables)

    VarName = Variables{v};

    fprintf('\n==============================\n');
    fprintf(' VARIABLE: %s\n', VarName);
    fprintf('==============================\n');

    folder = fullfile(base_path, ...
        sprintf('%s_results_outputs_forecasting',''), ...
        sprintf('%s_results_outputs_forecasting',VarName), ...
        'Metrics');

    % Alternative safer path (if above not working):
    folder = fullfile(base_path, [VarName '_results_outputs_forecasting'], 'Metrics');

    %% ===== STORAGE =====
    Models = {};
    Data = struct();

    %% =====================================================
    % =============== LOAD ALL HORIZONS =====================
    %% =====================================================
    for h = Horizons

        file = fullfile(folder, sprintf('Metrics_%s_H%d.csv', VarName, h));

        if ~exist(file,'file')
            warning('File not found: %s', file);
            continue;
        end

        T = readtable(file);

        % Clean model names (important)
        T.Model = string(T.Model);

        if h == 1
            Models = T.Model;
        end

        for i = 1:numel(Models)

            model = Models(i);

            idx = T.Model == model;

            if ~any(idx)
                continue;
            end

            % Store metrics
            Data.(model).nRMSE(h) = T.nRMSE(idx);
            Data.(model).nMAE(h)  = T.nMAE(idx);
            Data.(model).r2(h)    = T.r2(idx);
            Data.(model).NICE(h)  = T.NICE(idx);
        end
    end

    %% ===== CHECK PERSISTENCE =====
    if ~isfield(Data,'Ref_P')
        warning('Ref_P not found for %s', VarName);
        continue;
    end

    metrics_P = Data.Ref_P;

    %% =====================================================
    % =============== COMPARE MODELS ========================
    %% =====================================================
    modelNames = fieldnames(Data);

    for i = 1:numel(modelNames)

        model = modelNames{i};

        if strcmp(model,'Ref_P')
            continue;
        end

        metrics_model = Data.(model);

        result = compare_model_vs_P(metrics_model, metrics_P);

        Results(end+1,:) = {VarName, model, ...
            result.score, ...
            result.pvals(1), ...
            result.pvals(2), ...
            result.pvals(3), ...
            result.pvals(4), ...
            result.p_global};
    end
end

%% ================= SAVE TABLE ============================
ResultTable = cell2table(Results, ...
    'VariableNames',{'Variable','Model','Score',...
    'p_nRMSE','p_nMAE','p_r2','p_NICE','p_global'});

output_file = fullfile(base_path,'Statistical_Comparison_vs_Persistence.csv');
writetable(ResultTable, output_file);

disp('=== DONE: Statistical comparison completed ===');
disp(['Saved in: ' output_file]);



%% =========================================================
% =============== FUNCTION ================================
%% =========================================================
function result = compare_model_vs_P(metrics_model, metrics_P)

% ===== CLEAN FUNCTION =====
clean = @(a,b) deal(a(isfinite(a)&isfinite(b)), b(isfinite(a)&isfinite(b)));

[a1,b1] = clean(metrics_model.nRMSE, metrics_P.nRMSE);
[a2,b2] = clean(metrics_model.nMAE , metrics_P.nMAE);
[a3,b3] = clean(metrics_model.r2   , metrics_P.r2);
[a4,b4] = clean(metrics_model.NICE , metrics_P.NICE);

% ===== SAFETY =====
if numel(a1) < 3
    warning('Not enough samples for statistical test');
end

% ===== WILCOXON TESTS =====
try
    [p1,h1] = signrank(a1,b1,'tail','left');   % smaller better
catch, p1=1; h1=0; end

try
    [p2,h2] = signrank(a2,b2,'tail','left');
catch, p2=1; h2=0; end

try
    [p3,h3] = signrank(a3,b3,'tail','right');  % larger better
catch, p3=1; h3=0; end

try
    [p4,h4] = signrank(a4,b4,'tail','left');
catch, p4=1; h4=0; end

% ===== SCORE =====
score = h1 + h2 + h3 + h4;


% ===== FISHER COMBINATION =====
pvals = [p1 p2 p3 p4];
pvals(pvals==0) = 1e-10;


chi2 = -2 * sum(log(pvals));
df   = 2 * length(pvals);
p_global = 1 - chi2cdf(chi2, df);

% ===== OUTPUT =====
result.score = score;
result.pvals = pvals;
result.p_global = p_global;

end