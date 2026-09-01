%% =========================================================
%  FORECASTING — GHI HOURLY + 30MIN 
%% =========================================================

clear; close all; clc;

%% =========================================================
% GHI and GHI_15min VARIABLES 
%% =========================================================

VariableList = {'GHI_30min','GHI_1h'};
%% =========================================================
% LOOP VARIABLES
%% =========================================================

for ivar = 1:numel(VariableList)

    VariableName = VariableList{ivar};

    fprintf('\n====================================\n');
    fprintf(' PROCESSING: %s\n',VariableName);
    fprintf('====================================\n');
switch VariableName
    %% =====================================================
    % GHI 30MIN
    %% =====================================================
    case 'GHI_30min'
    
        outdir = ...
            'GHI_30min_results_outputs_forecasting';
    
        TRAIN_LEN = 1*8760*2; #2
    
        %% LOAD DATA
        data = load( ...
            'AJACCIO_station9_30MIN_T2M_GHI.mat');
    
        GHI = data.GHI_30min(:);
    
        %% LOAD CLEAR SKY
        load G_clearsky_h.mat
    
        CS_hourly = G_clearsky_h(:);
    
        %% CONVERT CLEAR SKY TO 30MIN
        CS = repelem(CS_hourly,2);
    
        %% REPEAT CLEAR SKY
        for i = length(CS)+1:length(GHI)
            CS(i) = CS(i-length(CS_hourly)*2);
        end
    
        CS = CS(1:length(GHI));
    
        %% RESULTS FILE
        results_file = ...
            'results_final_pipeline/GHI_30min_FULL.csv';

    %% =====================================================
    % GHI 1H
    %% =====================================================
    case 'GHI_1h'
    
        outdir = ...
            'GHI_1h_results_outputs_forecasting';
    
        TRAIN_LEN = 1*8760; %2
    
        %% LOAD DATA
        data = load( ...
            'AJACCIO_station9_1H_T2M_GHI.mat');
    
        GHI = data.GHI_1h(:);
    
        %% LOAD CLEAR SKY
        load G_clearsky_h.mat
    
        CS = G_clearsky_h(:);
    
        %% REPEAT CLEAR SKY
        for i = 8761:length(GHI)
            CS(i) = CS(i-8760);
        end
    
        CS = CS(1:length(GHI));
    
        %% RESULTS FILE
        results_file = ...
            'results_final_pipeline/GHI_1h_FULL.csv';
end
%% =========================================================
% CREATE OUTPUT FOLDERS
%% =========================================================

sub = {'Metrics','Forecasts'};

for i = 1:numel(sub)

    if ~exist(fullfile(outdir,sub{i}),'dir')

        mkdir(fullfile(outdir,sub{i}));

    end

end

%% =========================================================
% GENERAL PARAMETERS
%% =========================================================

%% =====================================================
% FORECAST HORIZONS
%% =====================================================

if contains(VariableName,'30min')

    Horizons = 2:2:12; %12

else

    Horizons = 1:6; %6

end

MethodNames = { ...
    'RAW', ...
    'LOESS', ...
    'FOURIER', ...
    'MEDIAN', ...
    'Projection', ...
    'Phase'};

%% =========================================================
% LOAD DESEASONALIZATION RESULTS
%% =========================================================

Tdes = readtable(results_file);

GHI = GHI(:);

%% =========================================================
% RESIDUALS
%% =========================================================

Rset = { ...
    GHI(:), ...
    Tdes.R_LOESS(:), ...
    Tdes.R_FOURIER(:), ...
    Tdes.R_MEDIAN(:), ...
    Tdes.R_EL_Projection(:), ...
    Tdes.R_EL_Phase(:)};

%% =========================================================
% SEASONAL COMPONENTS
%% =========================================================

Sset = { ...
    zeros(size(GHI)), ...
    Tdes.S_LOESS(:), ...
    Tdes.S_FOURIER(:), ...
    Tdes.S_MEDIAN(:), ...
    Tdes.S_EL_Projection(:), ...
    Tdes.S_EL_Phase(:)};

%% =========================================================
% GLOBAL CLEAN
%% =========================================================

valid = isfinite(GHI) & isfinite(CS);

for k = 1:numel(Rset)

    valid = valid & ...
            isfinite(Rset{k}) & ...
            isfinite(Sset{k});

end

GHI = GHI(valid);
CS  = CS(valid);

for k = 1:numel(Rset)

    Rset{k} = Rset{k}(valid);
    Sset{k} = Sset{k}(valid);

end

%% =========================================================
% LOOP HORIZONS
%% =========================================================

for Horizon = Horizons

    if contains(VariableName,'30min')
        HorizonHours = Horizon / 2;
    else
        HorizonHours = Horizon;
    end
    
    fprintf('\n=== %s — Horizon %gh ===\n', ...
            VariableName,HorizonHours);
    %% =====================================================
    % HYPERPARAMETERS
    %% =====================================================

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
    %% =====================================================
    % STORAGE
    %% =====================================================

    Global = {};

    Forecast_EL_AR = struct();
    Forecast_REF   = struct();

    %% =====================================================
    % LOOP METHODS
    %% =====================================================

    for m = 1:numel(MethodNames)

        R = Rset{m};
        S = Sset{m};

        %% =================================================
        % NORMALIZATION
        %% =================================================

        mu = mean(R(1:TRAIN_LEN),'omitnan');

        sg = std(R(1:TRAIN_LEN),'omitnan');

        if sg < 1e-10
            sg = 1;
        end

        Rz = (R - mu) ./ sg;

        %% =================================================
        % TRAIN SET
        %% =================================================

        [Xtr,Ytr] = ...
            InputOutput_R( ...
            Rz(1:TRAIN_LEN), ...
            Horizon, ...
            Nin);

        %% =================================================
        % TEST SET
        %% =================================================

        [Xte,Yte] = ...
            InputOutput_R( ...
            Rz(TRAIN_LEN-Nin+1:end), ...
            Horizon, ...
            Nin);

        %% =================================================
        % CLEAN TRAIN
        %% =================================================

        valid_tr = ...
            all(isfinite(Xtr),2) & ...
            isfinite(Ytr);

        Xtr = Xtr(valid_tr,:);
        Ytr = Ytr(valid_tr);

        %% =================================================
        % CLEAN TEST
        %% =================================================

        valid_te = ...
            all(isfinite(Xte),2) & ...
            isfinite(Yte);

        Xte = Xte(valid_te,:);
        Yte = Yte(valid_te);

        %% =================================================
        % ALIGNMENT
        %% =================================================

        idx_start = TRAIN_LEN + Horizon;

        L = size(Xte,1);

        idx_end = idx_start + L - 1;

        obs = GHI(idx_start:idx_end);

        Ste = S(idx_start:idx_end);

        %% =================================================
        % ================= ELM =============================
        %% =================================================

        model = ['EL_' MethodNames{m}];

        [B,W] = Train_EL( ...
            Xtr,Ytr,Nh,0.2,0.6,96);

        [Rhat,~] = Pred_EL_R( ...
            Xte,Yte,W,B,Nh,mu,sg);

        pred = Rhat(:) + Ste(:);

        %% PHYSICAL
        obs(obs < 0) = 0;
        pred(pred < 0) = 0;

        %% METRICS
        [nRMSE,nMAE,R2,r2,nMBE] = ...
            Erreur(obs,pred);

        [~,~,~,NICE] = ...
            NICE_function(obs,pred,Horizon);

        %% STORE
        Global(end+1,:) = { ...
            model,...
            nRMSE,nMAE,R2,r2,nMBE,NICE};

        Forecast_EL_AR.(model) = pred;

        %% =================================================
        % ================= AR ==============================
        %% =================================================

        model = ['AR_' MethodNames{m}];

        B_AR = Train_AR(Xtr,Ytr);

        [Rhat,~] = Pred_AR_R( ...
            Xte,Yte,B_AR,mu,sg);

        pred = Rhat(:) + Ste(:);

        %% PHYSICAL
        obs(obs < 0) = 0;
        pred(pred < 0) = 0;

        %% METRICS
        [nRMSE,nMAE,R2,r2,nMBE] = ...
            Erreur(obs,pred);

        [~,~,~,NICE] = ...
            NICE_function(obs,pred,Horizon);

        %% STORE
        Global(end+1,:) = { ...
            model,...
            nRMSE,nMAE,R2,r2,nMBE,NICE};

        Forecast_EL_AR.(model) = pred;

    end

    %% =====================================================
    % REFERENCES
    %% =====================================================

    [mes,pAR,pSP,pP,pCS,pCLIP,pES,pARTU,pCOMB] = ...
        pred_Ref(GHI,CS,Horizon,TRAIN_LEN);

    RefNames = { ...
        'AR','SP','P','CS', ...
        'CLIPER','ES','ARTU','COMB'};

    RefPreds = { ...
        pAR,pSP,pP,pCS,pCLIP,pES,pARTU,pCOMB};

    for i = 1:numel(RefNames)

        model = ['Ref_' RefNames{i}];

        obs  = mes(:);

        pred = RefPreds{i}(:);

        %% ALIGN
        L = min(length(obs),length(pred));

        obs  = obs(1:L);
        pred = pred(1:L);

        %% PHYSICAL
        obs(obs < 0) = 0;
        pred(pred < 0) = 0;

        %% METRICS
        [nRMSE,nMAE,R2,r2,nMBE] = ...
            Erreur(obs,pred);

        [~,~,~,NICE] = ...
            NICE_function(obs,pred,Horizon);

        %% STORE
        Global(end+1,:) = { ...
            model,...
            nRMSE,nMAE,R2,r2,nMBE,NICE};

        Forecast_REF.(model) = pred;

    end

    %% =====================================================
    % SAVE METRICS
    %% =====================================================

    GlobalTable = cell2table(Global,...
        'VariableNames',{ ...
        'Model','nRMSE','nMAE', ...
        'R2','r2','nMBE','NICE'});

    writetable( ...
    GlobalTable,...
    fullfile(outdir,'Metrics', ...
    sprintf('Metrics_%s_%gH.csv', ...
    VariableName,HorizonHours)));
    %% =====================================================
    % SAVE FORECASTS
    %% =====================================================

    save_forecasts( ...
        outdir,HorizonHours, ...
        obs,...
        Forecast_EL_AR,...
        'EL_AR');

    save_forecasts( ...
        outdir,HorizonHours, ...
        obs,...
        Forecast_REF,...
        'REF');

end
end
disp('=== FINISHED SUCCESSFULLY ===');
%% =========================================================
function save_forecasts(outdir,HorizonHours,obs,F,tag)
% =========================================================
% SAVE FORECASTS WITH STRICT ALIGNMENT
% =========================================================

N = length(obs);

T = table((1:N)',obs(:), ...
    'VariableNames',{'Time','Observed'});

fn = fieldnames(F);

for k = 1:numel(fn)

    fk = F.(fn{k});
    fk = fk(:);

    %% =====================================================
    % STRICT ALIGNMENT
    %% =====================================================

    if length(fk) >= N

        fk = fk(end-N+1:end);

    else

        fk = [nan(N-length(fk),1); fk];

    end

    T.(fn{k}) = fk;

end

%% =========================================================
% WRITE TABLE
%% =========================================================

writetable( ...
    T,...
    fullfile(outdir,'Forecasts', ...
    sprintf('Forecasts_%s_%gH.csv', ...
    tag,HorizonHours)))
end

%% =========================================================
function [N1,N2,N3,NS] = NICE_function(obs,pred,h)
%% =========================================================
% COLUMN VECTORS
%% =========================================================

obs  = obs(:);
pred = pred(:);

%% =========================================================
% REMOVE INVALID VALUES
%% =========================================================

valid = isfinite(obs) & isfinite(pred);

obs  = obs(valid);
pred = pred(valid);

%% =========================================================
% SAFETY CHECK
%% =========================================================

if length(obs) <= h + 1

    N1 = NaN;
    N2 = NaN;
    N3 = NaN;
    NS = NaN;

    return

end

%% =========================================================
% PERSISTENCE ERRORS
%% =========================================================

eP = obs(1:end-h) - obs(1+h:end);

%% =========================================================
% MODEL ERRORS
%% =========================================================

e = pred(1+h:end) - obs(1+h:end);

%% =========================================================
% NICE METRICS
%% =========================================================

N1 = sum(abs(e)) ./ ...
     sum(abs(eP));

N2 = sqrt(sum(e.^2)) ./ ...
     sqrt(sum(eP.^2));

N3 = (sum(abs(e).^3)).^(1/3) ./ ...
     (sum(abs(eP).^3)).^(1/3);

%% =========================================================
% GLOBAL NICE
%% =========================================================

NS = mean([N1 N2 N3]);

end
