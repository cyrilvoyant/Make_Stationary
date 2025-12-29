%% =====================================================
%  MAIN — Seasonal extraction via ELM + CLEAN evaluation
% =====================================================
clear; clc; close all;

tGlobal = tic;
outdir = 'results_outputs_final';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% =====================================================
%  PV CORSE
% =====================================================
file = 'Data.xlsx';
colPV = 'Solaire photovoltaïque (MW)';
T = readtable(file,'PreserveVariableNames',true);
series = T.(colPV);
series(series<0) = 0;

annee = 2;
horizon = 24;
Nbre_Hidden = 400;

fprintf('\n=== PV CORSE ===\n');

tPV = tic;   % ⏱ start PV timer
[R_te2, ytrue_te, ~, R_te, ~, h, ...
 S_raw, S_r1, S_r2] = ...
 Make_Stationary(series, annee, horizon, Nbre_Hidden);

PV.ExecTime_s = toc(tPV);   % ⏱ elapsed PV time
fprintf('PV execution time: %.2f seconds\n', PV.ExecTime_s);

% ---- Store signals
PV.raw   = ytrue_te(:);
PV.Rproj = R_te(:);
PV.Rph   = R_te2(:);

% ---- Restore compatibility with plot_results.m
PV.series.raw   = PV.raw;
PV.series.Rproj = PV.Rproj;
PV.series.Rph   = PV.Rph;

%% ---- Metrics (PV)
PV.PACF = [S_raw, S_r1, S_r2];

PV.Entropy.raw   = spectral_entropy(PV.raw,1);
PV.Entropy.proj  = spectral_entropy(PV.Rproj,1);
PV.Entropy.phase = spectral_entropy(PV.Rph,1);

ep = PV.raw(1:end-h) - PV.raw(1+h:end);
PV.NICE.proj  = NICE_metric(PV.Rproj(1:end-h), ep);
PV.NICE.phase = NICE_metric(PV.Rph(1:end-h), ep);

%% =====================================================
%  GHI AJACCIO
% =====================================================
T = readtable('GLO_ajaccio.txt');
series = T.Var3;

fprintf('\n=== GHI AJACCIO ===\n');

tGHI = tic;   % ⏱ start GHI timer
[R_te2, ytrue_te, ~, R_te, ~, h, ...
 S_raw, S_r1, S_r2] = ...
 Make_Stationary(series, annee, horizon, Nbre_Hidden);

GHI.ExecTime_s = toc(tGHI);   % ⏱ elapsed GHI time
fprintf('GHI execution time: %.2f seconds\n', GHI.ExecTime_s);
% ---- Store signals
GHI.raw   = ytrue_te(:);
GHI.Rproj = R_te(:);
GHI.Rph   = R_te2(:);

% ---- Restore compatibility with plot_results.m
GHI.series.raw   = GHI.raw;
GHI.series.Rproj = GHI.Rproj;
GHI.series.Rph   = GHI.Rph;

%% ---- Metrics (GHI)
GHI.PACF = [S_raw, S_r1, S_r2];

GHI.Entropy.raw   = spectral_entropy(GHI.raw,1);
GHI.Entropy.proj  = spectral_entropy(GHI.Rproj,1);
GHI.Entropy.phase = spectral_entropy(GHI.Rph,1);

ep = GHI.raw(1:end-h) - GHI.raw(1+h:end);
GHI.NICE.proj  = NICE_metric(GHI.Rproj(1:end-h), ep);
GHI.NICE.phase = NICE_metric(GHI.Rph(1:end-h), ep);

%% =====================================================
% SAVE RESULTS
% =====================================================
save(fullfile(outdir,'results_PV.mat'),'PV','-v7.3');
save(fullfile(outdir,'results_GHI.mat'),'GHI','-v7.3');

%% =====================================================
% SAVE CLEAN TABLES (CSV)
% =====================================================

% ---- PACF TABLE (RAW + Proj + Phase)
PACF_Table = table( ...
    {'RAW';'R1_Projection';'R2_PhaseOnly'}, ...
    [PV.PACF(1); PV.PACF(2); PV.PACF(3)], ...
    [GHI.PACF(1); GHI.PACF(2); GHI.PACF(3)], ...
    'VariableNames',{'Signal','PV_Corse','GHI_Ajaccio'} ...
);
writetable(PACF_Table, fullfile(outdir,'PACFsum_PV_GHI.csv'));

% ---- NICE TABLE (no RAW by definition)
NICE_Table = table( ...
    {'R1_Projection';'R2_PhaseOnly'}, ...
    [PV.NICE.proj; PV.NICE.phase], ...
    [GHI.NICE.proj; GHI.NICE.phase], ...
    'VariableNames',{'Signal','PV_Corse','GHI_Ajaccio'} ...
);
writetable(NICE_Table, fullfile(outdir,'NICE_PV_GHI.csv'));

% ---- ENTROPY TABLE
Entropy_Table = table( ...
    {'RAW';'R1_Projection';'R2_PhaseOnly'}, ...
    [PV.Entropy.raw; PV.Entropy.proj; PV.Entropy.phase], ...
    [GHI.Entropy.raw; GHI.Entropy.proj; GHI.Entropy.phase], ...
    'VariableNames',{'Signal','PV_Corse','GHI_Ajaccio'} ...
);
writetable(Entropy_Table, fullfile(outdir,'Entropy_PV_GHI.csv'));
TotalTime_s = toc(tGlobal);
fprintf('\nTOTAL execution time: %.2f seconds\n', TotalTime_s);

Time_Table = table( ...
    PV.ExecTime_s, ...
    GHI.ExecTime_s, ...
    TotalTime_s, ...
    'VariableNames',{'PV_seconds','GHI_seconds','Total_seconds'} ...
);

writetable(Time_Table, fullfile(outdir,'ExecutionTime.csv'));

fprintf('\n=== CLEAN EVALUATION DONE SUCCESSFULLY ===\n');

%% =====================================================
% LOCAL FUNCTIONS
% =====================================================
function H = spectral_entropy(x, fs)
    if nargin<2, fs = 1; end
    x = x(:); x = x(isfinite(x));
    if numel(x) < 10, H = NaN; return; end
    [Pxx,~] = pwelch(x,[],[],[],fs);
    P = Pxx / sum(Pxx + eps);
    H = -sum(P.*log(P+eps)) / log(numel(P));
end

function n = NICE_metric(e, ep)
    e  = e(:); 
    ep = ep(:);
    n = mean([ ...
        mean(abs(e)) / mean(abs(ep)), ...
        rms(e) / rms(ep), ...
        nthroot(mean(abs(e).^3),3) / nthroot(mean(abs(ep).^3),3) ...
    ]);
end
