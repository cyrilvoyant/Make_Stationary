%% ========================================================= 
%  Uses results from: results_outputs_final/ 
%  To PLOT RESULTS
% =========================================================
clear; clc; close all;

outdir = 'results_outputs_final';

load(fullfile(outdir,'results_PV.mat'),'PV');
load(fullfile(outdir,'results_GHI.mat'),'GHI');

%% =========================================================
% 1) PACFsum BAR PLOTS
%     Measures remaining linear dependence
%     Lower PACFsum = better deseasonalisation
%% =========================================================
figure;
bar(PV.PACF)
set(gca,'XTickLabel',{'Raw','Projection','Phase-only'})
ylabel('PACFsum')
title('PACFsum comparison – PV Corsica')
grid on
saveas(gcf, fullfile(outdir,'PACFsum_PV.png'));

figure;
bar(GHI.PACF)
set(gca,'XTickLabel',{'Raw','Projection','Phase-only'})
ylabel('PACFsum')
title('PACFsum comparison – GHI Ajaccio')
grid on
saveas(gcf, fullfile(outdir,'PACFsum_GHI.png'));

%% =========================================================
% 2) TIME SERIES OVERLAY (RAW / RESIDUALS)
%    
%% =========================================================
N = 500;   % first samples for readability

figure;
%plot(PV.raw(1:N),'k'); hold on
plot(PV.Rproj(1:N),'r'); hold on
plot(PV.Rph(1:N),'b')
legend('Residual (projection)','Residual (phase-only)')
title('PV Corsica – Residuals')
xlabel('Time (hours)')
ylabel('Power (MW)')
grid on
saveas(gcf, fullfile(outdir,'Timeseries_PV_residuals.png'));

figure;
%plot(GHI.raw(1:N),'k'); hold on
plot(GHI.Rproj(1:N),'r'); hold on
plot(GHI.Rph(1:N),'b')
legend('Residual (projection)','Residual (phase-only)','Location','best')
title('GHI Ajaccio – Residuals')
xlabel('Time (hours)')
ylabel('Irradiance (W/m²)')
grid on
saveas(gcf, fullfile(outdir,'Timeseries_GHI_residuals.png'));

%% =========================================================
% 3) PACF SHAPES 
%    Confirms removal of diurnal/annual memory
%% =========================================================
maxLag = 48;

figure;
subplot(3,1,1)
parcorr(PV.raw,'NumLags',maxLag)
title('PV – Raw'); ylabel('PACF')

subplot(3,1,2)
parcorr(PV.Rproj,'NumLags',maxLag)
title('PV – Residual (projection)'); ylabel('PACF')

subplot(3,1,3)
parcorr(PV.Rph,'NumLags',maxLag)
title('PV – Residual (phase-only)')
xlabel('Lag (hours)'); ylabel('PACF')

saveas(gcf, fullfile(outdir,'PACF_shapes_PV.png'));

figure;
subplot(3,1,1)
parcorr(GHI.raw,'NumLags',maxLag)
title('GHI – Raw'); ylabel('PACF')

subplot(3,1,2)
parcorr(GHI.Rproj,'NumLags',maxLag)
title('GHI – Residual (projection)'); ylabel('PACF')

subplot(3,1,3)
parcorr(GHI.Rph,'NumLags',maxLag)
title('GHI – Residual (phase-only)')
xlabel('Lag (hours)'); ylabel('PACF')

saveas(gcf, fullfile(outdir,'PACF_shapes_GHI.png'));

%% =========================================================
% 4) POWER SPECTRUM
%    Shows removal of 24h / 365d harmonics
%% =========================================================
fs = 1; % hourly sampling

figure;
%pwelch(PV.raw,[],[],[],fs); hold on
pwelch(PV.Rproj,[],[],[],fs); hold on
pwelch(PV.Rph,[],[],[],fs);
legend('Residual (projection)','Residual (phase-only)','Location','best')
title('Power spectrum – PV Corsica')
grid on
saveas(gcf, fullfile(outdir,'Spectrum_PV.png'));

figure;
%pwelch(GHI.raw,[],[],[],fs); hold on
pwelch(GHI.Rproj,[],[],[],fs); hold on
pwelch(GHI.Rph,[],[],[],fs);
legend('Residual (projection)','Residual (phase-only)','Location','best')
title('Power spectrum – GHI Ajaccio')
grid on
saveas(gcf, fullfile(outdir,'Spectrum_GHI.png'));

%% =========================================================
% 5) SPECTRAL ENTROPY
%     Higher entropy = less periodic structure
%% =========================================================
figure;
bar([PV.Entropy.raw, PV.Entropy.proj, PV.Entropy.phase])
set(gca,'XTickLabel',{'Raw','Projection','Phase-only'})
ylabel('Spectral entropy')
title('Spectral entropy – PV Corsica')
grid on
saveas(gcf, fullfile(outdir,'Entropy_PV.png'));

figure;
bar([GHI.Entropy.raw, GHI.Entropy.proj, GHI.Entropy.phase])
set(gca,'XTickLabel',{'Raw','Projection','Phase-only'})
ylabel('Spectral entropy')
title('Spectral entropy – GHI Ajaccio')
grid on
saveas(gcf, fullfile(outdir,'Entropy_GHI.png'));

%% =========================================================
% 6) NICE METRIC (FORECAST RELEVANCE)
%    Applied ONLY to residuals (not Raw)
%    < 1 means better than persistence
%% =========================================================
figure;
bar([PV.NICE.proj, PV.NICE.phase])
set(gca,'XTickLabel',{'Projection','Phase-only'})
ylabel('NICE score')
title('Forecast relevance – PV Corsica')
grid on
saveas(gcf, fullfile(outdir,'NICE_PV.png'));

figure;
bar([GHI.NICE.proj, GHI.NICE.phase])
set(gca,'XTickLabel',{'Projection','Phase-only'})
ylabel('NICE score')
title('Forecast relevance – GHI Ajaccio')
grid on
saveas(gcf, fullfile(outdir,'NICE_GHI.png'));

fprintf('\n=== ALL FINAL FIGURES GENERATED SUCCESSFULLY ===\n');
