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
%LLE = @(x) estimate_LLE(x,24,5,40);
Labels = {'RAW','LOESS','FOURIER','MEDIAN','Proj','Phase'};

%% ================= LOAD DATA =================
T = readtable('Data.xlsx','PreserveVariableNames',true);

DataList = {
    'PV',  T.Date, max(0,T.('Solaire photovoltaïque (MW)')), 8760;
    'WT',  T.Date, max(0,T.('Eolien (MW)')),                 8760;
    'PT',  T.Date, max(0,T.('Production totale (MW)')),      8760;
};

S = load('AJACCIO_station9_15MIN_T2M_FF_GHI.mat');


%% =====================================================
% CREATE 30MIN AND 1H AJACCIO DATASETS
%% =====================================================

time15 = S.time_grid_15(:);

GHI15 = S.GHI(:);
T2M15 = S.T2M(:);

%% ================= 30 MIN =================

N30 = floor(length(GHI15)/2)*2;

GHI_30min = mean(reshape(GHI15(1:N30),2,[]),1,'omitnan')';
T2M_30min = mean(reshape(T2M15(1:N30),2,[]),1,'omitnan')';

time_grid_30 = time15(1:2:N30);

GHI_30min(GHI_30min < 0) = 0;

save('AJACCIO_station9_30MIN_T2M_GHI.mat','GHI_30min','T2M_30min','time_grid_30');

%% ================= 1 HOUR =================

N1H = floor(length(GHI15)/4)*4;

GHI_1h = mean(reshape(GHI15(1:N1H),4,[]),1,'omitnan')';
T2M_1h = mean(reshape(T2M15(1:N1H),4,[]),1,'omitnan')';

time_grid_1h = time15(1:4:N1H);

GHI_1h(GHI_1h < 0) = 0;

save('AJACCIO_station9_1H_T2M_GHI.mat','GHI_1h','T2M_1h','time_grid_1h');

disp('=== AJACCIO 30MIN + 1H DATASETS CREATED ===');

%% =====================================================
% LOAD AJACCIO 30MIN DATA
%% =====================================================

S30 = load('AJACCIO_station9_30MIN_T2M_GHI.mat');

%% =====================================================
% LOAD AJACCIO 1H DATA
%% =====================================================

S1H = load('AJACCIO_station9_1H_T2M_GHI.mat');

%% =====================================================
% LOAD BASTIA WIND
%% =====================================================

WB30 = load('Bastia_Wind_30min.mat');

WB60 = load('Bastia_Wind_60min.mat');

%% =====================================================
% APPEND DATA
%% =====================================================

DataList = [DataList; {

    % ================= WIND =================
    'WS_30min', WB30.time_Bastia_30, ...
             WB30.FF_Bastia_30, 8760*2;

    'WS_1h',    WB60.time_Bastia_60, ...
                 WB60.FF_Bastia_60, 8760;
    % ================= GHI ==================
    'GHI_30min', S30.time_grid_30, ...
                  max(0,S30.GHI_30min), 8760*2;

    'GHI_1h',    S1H.time_grid_1h, ...
                  max(0,S1H.GHI_1h), 8760;

    % ================= T2M ==================
    'T2M_30min', S30.time_grid_30, ...
                  S30.T2M_30min, 8760*2;

    'T2M_1h',    S1H.time_grid_1h, ...
                  S1H.T2M_1h, 8760;

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
    samples_per_day = round(H / 365);
    series = series(:);
    time   = time(:);

    fprintf('\nProcessing %s\n',name);

    horizon = OptParams.(name).h;
    m       = OptParams.(name).m;

    %% ===== BASELINE METHODS =====
    
    t = (1:numel(series))';

    u = mod(t-1,samples_per_day) / samples_per_day;
    v = mod(t-1,H) / H;
    [S_fourier,~] = crossed_fourier_deseason( ...
    series,u,v,3,3,1e-2,H);

    window_size = round(24*30*samples_per_day / 24); 

    
    %% ===== MEDIAN TRAIN ONLY =====

    train_signal = series(1:H);
    
    [S_train_median,~] = rolling_median_deseason( ...
                         train_signal,...
                         window_size,...
                         samples_per_day);
    
    S_median = nan(size(series));
    
    Nyear = floor(length(series)/H);
    
    for yy = 1:Nyear
    
        i1 = (yy-1)*H + 1;
        i2 = min(yy*H,length(series));
    
        L = i2 - i1 + 1;
    
        S_median(i1:i2) = S_train_median(1:L);
    
    end
    S_loess = nan(size(series));

    %% ===== LOESS TRAIN ONLY =====

    train_signal = series(1:H);
    
    S_train = smoothdata( ...
        train_signal,...
        'loess',...
        window_size);
    
    S_loess = nan(size(series));
    
    Nyear = floor(length(series)/H);
    
    for yy = 1:Nyear
    
        i1 = (yy-1)*H + 1;
        i2 = min(yy*H,length(series));
    
        L = i2-i1+1;
    
        S_loess(i1:i2) = S_train(1:L);
    
    end
        

    %% ===== ELM TRAIN / TEST  =====
    [R_ph, yte, S_te2, R_proj, S_te, h, ...
     S_raw, S_r1, S_r2, ~, Model] = ...
     Make_stationary_all_variables(series, annee, horizon, m, H,name);

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
    maxLag = 2 * samples_per_day;

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

        tau = round(samples_per_day);

        LLE_loc(i) = estimate_LLE(x,tau,5,40);

    end

    PACF_cell{id} = PACF_loc;
    ENT_cell{id}  = ENT_loc;
    LLE_cell{id}  = LLE_loc;
    Names{id}     = name;

    %% ===== FULL  =====
    [S_proj_full, S_phase_full] = apply_elm_full(series, Model, horizon, H, name);

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
