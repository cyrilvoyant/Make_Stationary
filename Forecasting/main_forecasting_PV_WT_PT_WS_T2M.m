%% =========================================================
%  main_forecasting_PV_WT_PT_WS_T2M
%% =========================================================

clear; close all; clc;

%% =========================================================
% FORECAST HORIZONS
%% =========================================================
Horizons_hourly = 1:1; %6
%% =========================================================
% METHODS
%% =========================================================
MethodNames = { ...
    'RAW', ...
    'LOESS', ...
    'FOURIER', ...
    'MEDIAN', ...
    'Projection', ...
    'Phase'};

%% =========================================================
% VARIABLES
%% =========================================================
Variables = { ...
    'PV',  true; ...
    'WT',  true; ...
    'PT',  true; ...
    'WS_30min',  true; ...
    'WS_1h',  true; ...
    'T2M_30min',  false; ...
    'T2M_1h', false};

%% =========================================================
% LOOP VARIABLES
%% =========================================================
for v = 1:size(Variables,1)

    VarName = Variables{v,1};
    is_nonnegative = Variables{v,2};
    if contains(VarName,'30min')
        Horizons = 2:2:12; %12
    else
        Horizons = 1:6; %6
    end

    %% =====================================================
    % TRAIN LENGTH
    %% =====================================================
    switch VarName

        case {'PV','WT','PT'}
            TRAIN_LEN = 1*8760; %6

        case {'WS_30min'}
            TRAIN_LEN = 1*8760*2;

        case {'WS_1h'}
            TRAIN_LEN = 1*8760;
        case {'T2M_30min'}

            TRAIN_LEN = 1*8760*2;
    
        case {'T2M_1h'}
            TRAIN_LEN = 1*8760;
            
    end

    fprintf('\n==============================\n');
    fprintf(' VARIABLE: %s\n',VarName);
    fprintf('==============================\n');

    %% =====================================================
    % OUTPUT FOLDERS
    %% =====================================================
    outdir = sprintf('%s_results_outputs_forecasting',VarName);

    if ~exist(outdir,'dir')
        mkdir(outdir);
    end

    if ~exist(fullfile(outdir,'Metrics'),'dir')
        mkdir(fullfile(outdir,'Metrics'));
    end

    if ~exist(fullfile(outdir,'Forecasts'),'dir')
        mkdir(fullfile(outdir,'Forecasts'));
    end

    %% =====================================================
    % LOAD DATA
    %% =====================================================
    fname = sprintf('results_final_pipeline/%s_FULL.csv',VarName);

    Tdes = readtable(fname);

    Xraw = Tdes.Raw(:);

    %% =====================================================
    % RESIDUALS
    %% =====================================================
    Rset = { ...
        Tdes.Raw(:), ...
        Tdes.R_LOESS(:), ...
        Tdes.R_FOURIER(:), ...
        Tdes.R_MEDIAN(:), ...
        Tdes.R_EL_Projection(:), ...
        Tdes.R_EL_Phase(:)};

    %% =====================================================
    % SEASONAL COMPONENTS
    %% =====================================================
    Sset = { ...
        zeros(size(Xraw)), ...
        Tdes.S_LOESS(:), ...
        Tdes.S_FOURIER(:), ...
        Tdes.S_MEDIAN(:), ...
        Tdes.S_EL_Projection(:), ...
        Tdes.S_EL_Phase(:)};

    %% =====================================================
    % GLOBAL CLEAN
    %% =====================================================
    valid = isfinite(Xraw);

    for k = 1:numel(Rset)

        valid = valid & ...
                isfinite(Rset{k}) & ...
                isfinite(Sset{k});

    end

    Xraw = Xraw(valid);

    for k = 1:numel(Rset)

        Rset{k} = Rset{k}(valid);
        Sset{k} = Sset{k}(valid);

    end

    %% =====================================================
    % LOOP HORIZONS
    %% =====================================================
    for Horizon = Horizons

        if contains(VarName,'30min')
            HorizonHours = Horizon/2;
        else
            HorizonHours = Horizon;
        end

        fprintf('--- Horizon %gh ---\n', HorizonHours);

        %% =================================================
        % HYPERPARAMETERS
        %% =================================================
        switch Horizon

            case 1
                Nin = 52; Nh = 220;

            case 2
                Nin = 50; Nh = 230;

            case 3
                Nin = 49; Nh = 220;

            case 4
                Nin = 51; Nh = 170;

            case 5
                Nin = 46; Nh = 230;

            otherwise
                Nin = 72; Nh = 220;

        end

        %% =================================================
        % STORAGE
        %% =================================================
        Global = {};

        Forecast_ELAR = struct();
        Forecast_REF  = struct();

        %% =================================================
        % LOOP METHODS
        %% =================================================
        for m = 1:numel(MethodNames)

            R = Rset{m};
            S = Sset{m};

            %% =============================================
            % TRAIN NORMALIZATION
            %% =============================================
            mu = mean(R(1:TRAIN_LEN),'omitnan');
            sg = std (R(1:TRAIN_LEN),'omitnan');

            if sg < 1e-10
                sg = 1;
            end

            Rz = (R - mu) ./ sg;

            %% =============================================
            % BUILD TRAIN SET
            %% =============================================
            [Xtr,Ytr] = ...
                InputOutput_R( ...
                Rz(1:TRAIN_LEN), ...
                Horizon, ...
                Nin);

            %% =============================================
            % BUILD TEST SET
            %% =============================================
            [Xte,Yte] = ...
                InputOutput_R( ...
                Rz(TRAIN_LEN-Nin+1:end), ...
                Horizon, ...
                Nin);

            %% =============================================
            % TRUE OBSERVATIONS ALIGNMENT
            %% =============================================
            idx_start = TRAIN_LEN + Horizon;

            L = size(Xte,1);

            idx_end = idx_start + L - 1;

            obs = Xraw(idx_start:idx_end);

            Ste = S(idx_start:idx_end);

            %% =============================================
            % ================= ELM =========================
            %% =============================================
            model = ['EL_' MethodNames{m}];

            [B,W] = Train_EL( ...
                Xtr,Ytr,Nh,0.2,0.6,96);% lamdda= 0.2 Nbr_run= 96

            [Rhat,~] = Pred_EL_R( ...
                Xte,Yte,W,B,Nh,mu,sg);

            Rhat = Rhat(:);

            pred = Rhat + Ste;

            if is_nonnegative
                pred(pred < 0) = 0;
            end

            %% ---------- METRICS ----------
            [nRMSE,nMAE,R2,r2,nMBE] = ...
                Erreur(obs,pred);

            [~,~,~,NICE] = ...
                NICE_function(obs,pred,Horizon);

            %% ---------- STORE ----------
            Global(end+1,:) = { ...
                model, ...
                nRMSE,nMAE,R2,r2,nMBE,NICE};

            Forecast_ELAR.(model) = pred;

            %% =============================================
            % ================= AR ==========================
            %% =============================================
            model = ['AR_' MethodNames{m}];

            B_AR = Train_AR(Xtr,Ytr);

            [Rhat,~] = Pred_AR_R( ...
                Xte,Yte,B_AR,mu,sg);

            Rhat = Rhat(:);

            pred = Rhat + Ste;

            if is_nonnegative
                pred(pred < 0) = 0;
            end

            %% ---------- METRICS ----------
            [nRMSE,nMAE,R2,r2,nMBE] = ...
                Erreur(obs,pred);

            [~,~,~,NICE] = ...
                NICE_function(obs,pred,Horizon);

            %% ---------- STORE ----------
            Global(end+1,:) = { ...
                model, ...
                nRMSE,nMAE,R2,r2,nMBE,NICE};

            Forecast_ELAR.(model) = pred;

        end

        %% =================================================
        % REFERENCES
        %% =================================================
        [obs_ref,pP,pCLIM] = ...
            pred_Ref_PV_WT_PT_WS_T2M( ...
            Xraw,Horizon,TRAIN_LEN,is_nonnegative);

        RefNames = {'P','CLIM'};
        RefPreds = {pP,pCLIM};

        for i = 1:numel(RefNames)

            model = ['Ref_' RefNames{i}];

            pred = RefPreds{i}(:);

            %% ---------- METRICS ----------
            [nRMSE,nMAE,R2,r2,nMBE] = ...
                Erreur(obs_ref,pred);

            [~,~,~,NICE] = ...
                NICE_function(obs_ref,pred,Horizon);

            %% ---------- STORE ----------
            Global(end+1,:) = { ...
                model, ...
                nRMSE,nMAE,R2,r2,nMBE,NICE};

            Forecast_REF.(model) = pred;

        end

        %% =================================================
        % SAVE METRICS
        %% =================================================
        GlobalTable = cell2table(Global,...
            'VariableNames',{ ...
            'Model','nRMSE','nMAE', ...
            'R2','r2','nMBE','NICE'});

        writetable( ...
            GlobalTable,...
            fullfile(outdir,'Metrics', ...
            sprintf('Metrics_%s_%gH.csv', ...
            VarName,HorizonHours)));

        %% =================================================
        % SAVE FORECASTS
        %% =================================================

        % ===== EL/AR forecasts saved with aligned obs =====
        save_forecasts( ...
            outdir,HorizonHours, ...
            obs, ...
            Forecast_ELAR, ...
            'EL_AR');

        % ===== Reference forecasts =====
        save_forecasts( ...
            outdir,HorizonHours, ...
            obs_ref, ...
            Forecast_REF, ...
            'REF');

    end
end

disp('=== ALL VARIABLES FINISHED SUCCESSFULLY ===');

%% =========================================================
function [N1,N2,N3,NS] = NICE_function(obs,pred,H)

obs  = obs(:);
pred = pred(:);

valid = isfinite(obs) & isfinite(pred);

obs  = obs(valid);
pred = pred(valid);

if length(obs) <= H+1

    N1 = NaN;
    N2 = NaN;
    N3 = NaN;
    NS = NaN;

    return

end

%% =========================================================
% TRUE PERSISTENCE ERRORS
%% =========================================================
eP = obs(1:end-H) - obs(1+H:end);

%% =========================================================
% MODEL ERRORS
%% =========================================================
e = pred - obs;

% align lengths with persistence error
e = e(1+H:end);

%% =========================================================
% NICE METRICS
%% =========================================================
N1 = sum(abs(e)) ./ sum(abs(eP));

N2 = sqrt(sum(e.^2)) ./ ...
     sqrt(sum(eP.^2));

N3 = (sum(abs(e).^3)).^(1/3) ./ ...
     (sum(abs(eP).^3)).^(1/3);

NS = mean([N1 N2 N3]);

end

%% =========================================================
function save_forecasts(outdir,HorizonHours,obs,F,tag)
N = length(obs);

T = table((1:N)',obs(:), ...
    'VariableNames',{'Time','Observed'});

fn = fieldnames(F);

for k = 1:numel(fn)

    fk = F.(fn{k});
    fk = fk(:);

    if length(fk) >= N
        fk = fk(end-N+1:end);
    else
        fk = [nan(N-length(fk),1); fk];
    end

    T.(fn{k}) = fk;

end

writetable(T, ...
    fullfile(outdir,'Forecasts', ...
    sprintf('Forecasts_%s_%gH.csv',tag,HorizonHours)));

end