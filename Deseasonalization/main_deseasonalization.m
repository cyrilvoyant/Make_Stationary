%% =====================================================
%  FINAL DESEASONALISATION PIPELINE
% =====================================================
clear; clc; close all;

outdir = 'results_final_pipeline';
if ~exist(outdir,'dir'), mkdir(outdir); end

fs = 1;
annee = 2;

%% ================= PARALLEL =================
if isempty(gcp('nocreate'))
    parpool;   % auto workers
end

%% ================= LOAD OPTIMIZED PARAMETERS =================
OptTable = readtable('Best_Optimization_AllVariables.csv');

for i = 1:height(OptTable)
    name = OptTable.Variable{i};
    OptParams.(name).h = OptTable.BestLagH(i);
    OptParams.(name).m = OptTable.BestM(i);
end

%% ================= METRICS =================
LLE = @(x) estimate_LLE(x,24,5,40);
Labels = {'RAW','LOESS','FOURIER','MEDIAN','EL_Proj','EL_Phase'};

%% ================= LOAD DATA =================
T = readtable('Data.xlsx','PreserveVariableNames',true);

DataList = {
    'PV',  T.Date, max(0,T.('Solaire photovoltaïque (MW)')), 8760;
    'WT',  T.Date, max(0,T.('Eolien (MW)')),                 8760;
    'PT',  T.Date, max(0,T.('Production totale (MW)')),      8760;
};

S = load('AJACCIO_station9_15MIN_T2M_FF_GHI.mat');

DataList = [DataList; {
    'WS',        S.time_grid_15, S.FF,           8760*4;
    'GHI_15min', S.time_grid_15, max(0,S.GHI),   8760*4;
    'T2M',       S.time_grid_15, S.T2M,          8760*4;
}];

Tg = readtable('GLO_ajaccio.txt');

DataList = [DataList; {
    'GHI', datetime(num2str(Tg.Var2),'InputFormat','yyyyMMddHH'), max(0,Tg.Var3), 8760;
}];

nVar = size(DataList,1);

%% ================= STORAGE =================
% PACF = struct();
% ENT  = struct();
% LLEm = struct();
PACF_cell = cell(nVar,1);
ENT_cell  = cell(nVar,1);
LLE_cell  = cell(nVar,1);
Names     = cell(nVar,1);
%% ================= MAIN LOOP (PARFOR) =================
parfor id = 1:nVar

    name   = DataList{id,1};
    time   = DataList{id,2};
    series = DataList{id,3};
    H      = DataList{id,4};

    fprintf('\nProcessing %s\n',name);

    horizon = OptParams.(name).h;
    m       = OptParams.(name).m;

    %% ===== BASELINE METHODS =====
    S_loess  = smoothdata(series,'loess',round(H/12));

    t = (1:numel(series))';
    u = mod(t-1,24)/24;
    v = mod(t-1,24*365)/(24*365);
    [S_fourier,~] = crossed_fourier_deseason(series,u,v,3,3,1e-2);

    [S_median,~] = rolling_median_deseason(series,10);

    %% ===== ELM TRAIN / TEST  =====
    [R_ph, yte, S_te2, R_proj, S_te, h, ...
     S_raw, S_r1, S_r2, ~, Model] = ...
     Make_stationary_all_variables(series, annee, horizon, m, H);

    %% ===== TEST WINDOW =====
    N_EL = numel(yte);
    idx_te = (numel(series)-N_EL+1):numel(series);

    %% ===== BUILD TEST SIGNALS =====
    Signals = {
        yte
        series(idx_te)-S_loess(idx_te)
        series(idx_te)-S_fourier(idx_te)
        series(idx_te)-S_median(idx_te)
        R_proj
        R_ph
    };

    %% ===== METRICS =====
    if H==8760*4
        maxLag = 48*4;
    else
        maxLag = 48;
    end

    PACF_loc = zeros(1,numel(Labels));
    ENT_loc  = zeros(1,numel(Labels));
    LLE_loc  = zeros(1,numel(Labels));

    for i = 1:numel(Labels)

        x = Signals{i};

        % PACF
        PACF_loc(i) = sum(abs(parcorr(x,'NumLags',maxLag)));

        % Entropy
        ENT_loc(i) = spectral_entropy(x,fs);

        % LLE (SAFE)
        if length(x) > 3000
            x = x(round(linspace(1,length(x),3000)));
        end

        LLE_loc(i) = LLE(x);

    end

    PACF_cell{id} = PACF_loc;
    ENT_cell{id}  = ENT_loc;
    LLE_cell{id}  = LLE_loc;
    Names{id}     = name;

    %% ===== FULL  =====
    [S_proj_full, S_phase_full] = ...
        apply_elm_full(series, Model, horizon, H);

    R_proj_full  = series - S_proj_full;
    R_phase_full = series - S_phase_full;

    %% ===== SAVE =====
    Deseason = table( ...
        time, series, ...
        S_loess, series-S_loess, ...
        S_fourier, series-S_fourier, ...
        S_median, series-S_median, ...
        S_proj_full, R_proj_full, ...
        S_phase_full, R_phase_full, ...
        'VariableNames',{
        'Time','Raw',...
        'S_LOESS','R_LOESS',...
        'S_FOURIER','R_FOURIER',...
        'S_MEDIAN','R_MEDIAN',...
        'S_EL_Projection','R_EL_Projection',...
        'S_EL_Phase','R_EL_Phase'});

    writetable(Deseason, fullfile(outdir,[name '_FULL.csv']));

end
PACF = struct();
ENT  = struct();
LLEm = struct();

for i = 1:nVar
    PACF.(Names{i}) = PACF_cell{i};
    ENT.(Names{i})  = ENT_cell{i};
    LLEm.(Names{i}) = LLE_cell{i};
end
%% ================= SAVE METRICS =================
SignalNames = fieldnames(PACF);
Metrics = table(Labels','VariableNames',{'Signal'});

for k = 1:numel(SignalNames)
    s = SignalNames{k};
    Metrics.([s '_PACF'])    = PACF.(s)';
    Metrics.([s '_Entropy']) = ENT.(s)';
    Metrics.([s '_LLE'])     = LLEm.(s)';
end

writetable(Metrics, fullfile(outdir,'Metrics.csv'));

disp('=== FINAL PIPELINE DONE  ===');

%% ================= LOCAL FUNCTIONS =================
function H = spectral_entropy(x, fs)
    x = x(:); x = x(isfinite(x));
    if numel(x)<10, H=NaN; return; end
    [Pxx,~] = pwelch(x,[],[],[],fs);
    P = Pxx / sum(Pxx + eps);
    H = -sum(P.*log(P+eps)) / log(numel(P));
end