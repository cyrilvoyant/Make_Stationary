%% =========================================================
%  DESEASONALIZATION MULTI-CRITERIA COMPARISON
%  (PACF ↓, Entropy ↑, LLE → 0)
%% =========================================================

clear; clc; close all;

%% ================= PATH =============================
file = 'C:/Users/HP/Downloads/Prof_Cyril Voyant_Proposition_Collaboration/Maklewa/final_deseasonalization_code_22_05_2026/results_final_pipeline/Metrics.csv';

T = readtable(file);

Methods = string(T.Signal);

Variables = {'GHI_30min','GHI_1h','PV','WT','PT','WS_30min','WS_1h','T2M_30min','T2M_1h'};

Results = {};

baseline = "RAW";

%% =========================================================
% ================= LOOP OVER VARIABLES =====================
%% =========================================================
for v = 1:numel(Variables)

    Var = Variables{v};

    fprintf('\n==============================\n');
    fprintf(' VARIABLE: %s\n', Var);
    fprintf('==============================\n');

    %% ===== Extract RAW values =====
    idx_raw = Methods == baseline;

    if ~any(idx_raw)
        warning('RAW not found for %s', Var);
        continue;
    end

    raw_PACF    = T.(sprintf('%s_PACF',Var))(idx_raw);
    raw_Entropy = T.(sprintf('%s_Entropy',Var))(idx_raw);
    raw_LLE     = T.(sprintf('%s_LLE',Var))(idx_raw);

    %% =====================================================
    % ================= LOOP MODELS =========================
    %% =====================================================
    for i = 1:numel(Methods)

        model = Methods(i);

        if model == baseline
            continue;
        end

        model_PACF    = T.(sprintf('%s_PACF',Var))(i);
        model_Entropy = T.(sprintf('%s_Entropy',Var))(i);
        model_LLE     = T.(sprintf('%s_LLE',Var))(i);

        %% ===== SCORE =====
        score = 0;

        % PACF ↓
        if model_PACF < raw_PACF
            score = score + 1;
        end

        % Entropy ↑
        if model_Entropy > raw_Entropy
            score = score + 1;
        end

        % LLE → 0
        if model_LLE < raw_LLE
            score = score + 1;
        end
        %% ===== PERCENTAGE IMPROVEMENTS =====
        
        PACF_gain = 100 * (raw_PACF - model_PACF) / abs(raw_PACF);
        
       
        %% ===== STORE =====
%         Results(end+1,:) = {Var, model, ...
%             model_PACF, model_Entropy, model_LLE, ...
%             score};
        Results(end+1,:) = {Var, model, ...
            model_PACF, model_Entropy, model_LLE, ...
            PACF_gain, ...
            score};
    end
end

%% ================= TABLE =============================
% ResultTable = cell2table(Results, ...
%     'VariableNames',{ ...
%     'Variable','Model',...
%     'PACF','Entropy','LLE',...
%     'Score'});
ResultTable = cell2table(Results, ...
    'VariableNames',{ ...
    'Variable','Model',...
    'PACF','Entropy','LLE',...
    'PACF_gain_percent',...
    'Score'});
%% ================= SORT (BEST FIRST) ==================
ResultTable = sortrows(ResultTable, ...
    {'Variable','Score'}, ...
    {'ascend','descend'});

%% ================= SAVE ==============================
out = 'Deseason_Comparison_Final.csv';
writetable(ResultTable,out);

disp('=== DONE ===');
disp(['Saved: ' out]);