%% =========================================================
%  POST-PROCESSING GHI and GHI_15min VARIABLES 
%% =========================================================

clear; close all; clc;

%% ================= VARIABLES =============================
Variables = {'GHI_30min','GHI_1h'};
Horizons_hourly = 1:6;

% ===== VIOLIN (4 metrics)
MetricsNames_violin  = {'nRMSE','nMAE'}; %,'r2','nMBE'
MetricsLabels_violin = {'nRMSE','nMAE'}; % ,'r^2','nMBE'

% ===== BAR (6 metrics)
MetricsNames_bar  = {'nRMSE','nMAE'}; %,'R2','NICE','r2','nMBE'
MetricsLabels_bar = {'nRMSE','nMAE'}; % ,'R^2','NICE' ,'r^2','nMBE'

%% =========================================================
for v = 1:numel(Variables)

    VarName = Variables{v};
    if contains(VarName,'30min')
        Horizons = 2:2:12;
    else
        Horizons = 1:6;
    end
    fprintf('\n==============================\n');
    fprintf(' POST-PROCESSING: %s\n',VarName);
    fprintf('==============================\n');

    %% PATHS
    baseDir     = sprintf('%s_results_outputs_forecasting',VarName);
    figDir      = fullfile(baseDir,'Figures');
    ForecastDir = fullfile(baseDir,'Forecasts');
    MetricDir   = fullfile(baseDir,'Metrics');

    if ~exist(figDir,'dir'), mkdir(figDir); end

    %% =====================================================
    % DETECT MODELS
    %% =====================================================
    ModelNames = {};
    
    for H = Horizons
        if contains(VarName,'30min')
            HorizonHours = H/2;
        else
            HorizonHours = H;
        end
        
        Ttmp = readtable(fullfile(ForecastDir,...
            sprintf('Forecasts_EL_AR_%gH.csv',HorizonHours)));

        vars = setdiff(Ttmp.Properties.VariableNames,...
            {'Time','Observed'});

        ModelNames = union(ModelNames, vars);
    end

    ModelNames = ModelNames(:)';
    ModelNames = [ModelNames , ...
    {'Ref_AR','Ref_P','Ref_CS'}];
    fprintf('Detected %d models\n',numel(ModelNames));

    %% =====================================================
    % VIOLIN PLOTS 
    %% =====================================================
    for imodel = 1:numel(ModelNames)

        ModelName = ModelNames{imodel};
        figure('Color','w','Position',[100 100 1400 900]);

        for im = 1:numel(MetricsNames_violin)
            subplot(2,1,im); hold on

            Pcell = cell(numel(Horizons),1);
            Mcell = cell(numel(Horizons),1);

            for ih = 1:numel(Horizons)

                H = Horizons(ih);

                if contains(VarName,'30min')
                    HorizonHours = H/2;
                else
                    HorizonHours = H;
                end
                
                TmEL = readtable(fullfile(ForecastDir,...
                    sprintf('Forecasts_EL_AR_%gH.csv',HorizonHours)));
                
                TmREF = readtable(fullfile(ForecastDir,...
                    sprintf('Forecasts_REF_%gH.csv',HorizonHours)));

                if startsWith(ModelName,'Ref_')

                    obs   = TmREF.Observed(:);
                    predM = TmREF.(ModelName)(:);
                
                else
                
                    obs   = TmEL.Observed(:);
                    predM = TmEL.(ModelName)(:);
                
                end
                predP = TmREF.Ref_P(:);
                
                % ==========================================
                % ALIGN LENGTHS
                % ==========================================
                L = min([length(obs), length(predM), length(predP)]);
                
                obs   = obs(1:L);
                predM = predM(1:L);
                predP = predP(1:L);
                
                % ==========================================
                % REMOVE INVALID VALUES
                % ==========================================
                valid = isfinite(obs) & ...
                        isfinite(predM) & ...
                        isfinite(predP);
                
                obs   = obs(valid);
                predM = predM(valid);
                predP = predP(valid);

                DM = daily_metrics_from_series(obs, predM);
                DP = daily_metrics_from_series(obs, predP);

                metric = MetricsNames_violin{im};

                Mcell{ih} = DM.(metric);
                Pcell{ih} = DP.(metric);
            end

            %% MATRIX
            maxN = max(cellfun(@numel,Mcell));
            maxN = max(maxN, max(cellfun(@numel,Pcell)));

            Mmat = nan(maxN,numel(Horizons));
            Pmat = nan(maxN,numel(Horizons));

            for ih=1:numel(Horizons)
                mv = Mcell{ih};
                pv = Pcell{ih};

                if ~isempty(mv)
                    Mmat(end-numel(mv)+1:end,ih) = mv;
                end
                if ~isempty(pv)
                    Pmat(end-numel(pv)+1:end,ih) = pv;
                end
            end

            %% VIOLIN
            hV = daviolinplot({Pmat,Mmat},...
                'violin','full',...
                'violinmin',0, ...
                'colors',[0.7 0.7 0.7; 0 0.447 0.741],...
                'xtlabels',string(Horizons_hourly));

            xlabel('Forecast horizon','FontWeight','bold')
            ylim([0 inf])
            ylabel(MetricsLabels_violin{im})
            title(MetricsLabels_violin{im})

            %% 
            hold on
            h1 = plot(nan,nan,'s','MarkerFaceColor',[0.7 0.7 0.7],'MarkerEdgeColor','k');
            h2 = plot(nan,nan,'s','MarkerFaceColor',[0 0.447 0.741],'MarkerEdgeColor','k');

            legend([h1 h2],{'Persistence',ModelName},...
                'Interpreter','none','Location','best')

        end

        sgtitle(sprintf('%s — %s',VarName,ModelName),...
            'Interpreter','none','FontSize',14,'FontWeight','bold')

        saveas(gcf, fullfile(figDir,...
            sprintf('%s_Violin_%s.png',VarName,ModelName)));
    end

    %% =====================================================
    % BAR PLOTS 
    %% =====================================================
    for H = Horizons

        if contains(VarName,'30min')
            HorizonHours = H/2;
        else
            HorizonHours = H;
        end
        
        T = readtable(fullfile(MetricDir,...
            sprintf('Metrics_%s_%gH.csv',VarName,HorizonHours)));
        idx_keep = startsWith(T.Model,'EL_') | ...
                   startsWith(T.Model,'AR_');

        T = T(idx_keep,:);

        figure('Color','w','Position',[100 100 1400 700]);

        cmap = lines(height(T));

        for im = 1:numel(MetricsNames_bar)

            subplot(2,1,im)

            vals = T.(MetricsNames_bar{im});
            b = bar(vals,'FaceColor','flat');

            for k=1:length(vals)
                b.CData(k,:) = cmap(k,:);
            end

            %% 
            ax = gca;
            ax.XTick = 1:numel(T.Model);
            ax.XTickLabel = T.Model;
            ax.TickLabelInterpreter = 'none'; 
            ax.XTickLabelRotation = 30;

            ax.FontSize = 10;

            ylabel(MetricsLabels_bar{im})
            title(MetricsLabels_bar{im})
            grid on
        end

        sgtitle(sprintf('%s-Global Metrics (H=%gh)',VarName,HorizonHours),...
            'Interpreter','none',...
            'FontSize',14,'FontWeight','bold')

        if contains(VarName,'30min')
            HorizonHours = H/2;
        else
            HorizonHours = H;
        end
        
        saveas(gcf, fullfile(figDir,...
            sprintf('%s_Bar_%gH.png',VarName,HorizonHours)));
    end
%% =====================================================
    % SCATTER 
    %% =====================================================
    for H = Horizons

        if contains(VarName,'30min')
            HorizonHours = H/2;
        else
            HorizonHours = H;
        end
        
        TmEL = readtable(fullfile(ForecastDir,...
            sprintf('Forecasts_EL_AR_%gH.csv',HorizonHours)));
        
        TmREF = readtable(fullfile(ForecastDir,...
            sprintf('Forecasts_REF_%gH.csv',HorizonHours))); 

       obs = TmEL.Observed(:);

        figure('Color','w','Position',[100 100 1400 900]);

        for k = 1:numel(ModelNames)

            if startsWith(ModelNames{k},'Ref_')

                pred = TmREF.(ModelNames{k})(:);
            
            else
            
                pred = TmEL.(ModelNames{k})(:);
            
            end
                        
            obs0 = obs(:);

            L = min(length(obs0),length(pred));
            
            o = obs0(1:L);
            p = pred(1:L);
            
            valid = isfinite(o) & isfinite(p);
            
            o = o(valid);
            p = p(valid);

            subplot(4,4,k)
            scatter(o,p,6,'filled'); hold on

            mn = min([o;p]);
            mx = max([o;p]);
            
            plot([mn mx],[mn mx],'k--','LineWidth',1.2)
            axis square
            grid on
            title(ModelNames{k},'Interpreter','none')
        end

        sgtitle(sprintf('%s Scatter H=%gh',VarName,HorizonHours),'Interpreter','none')

        if contains(VarName,'30min')
            HorizonHours = H/2;
        else
            HorizonHours = H;
        end
        
        saveas(gcf, fullfile(figDir,...
            sprintf('%s_Scatter_%gH.png',VarName,HorizonHours)));
    end
end

disp('=== ALL VARIABLES PLOTTED ===')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = daily_metrics_from_series(obs,pred)

% ======================================================
% DAILY METRICS COMPUTATION

% DETECT TEMPORAL RESOLUTION
% ======================================================

if length(obs) > 30000
    samples_per_day = 48;
else
    samples_per_day = 24;
end

obs  = obs(:);
pred = pred(:);

%% ======================================================
% REMOVE INVALID VALUES
%% ======================================================
idx = isfinite(obs) & isfinite(pred);

obs  = obs(idx);
pred = pred(idx);

%% ======================================================
% MINIMUM LENGTH CHECK
%% ======================================================
if numel(obs) < 2*samples_per_day

    D.nRMSE = [];
    D.nMAE  = [];
    D.nMBE  = [];
    D.r2    = [];

    return

end

%% ======================================================
% NUMBER OF COMPLETE DAYS
%% ======================================================
Nd = floor(numel(obs)/samples_per_day);

if Nd < 2

    D.nRMSE = [];
    D.nMAE  = [];
    D.nMBE  = [];
    D.r2    = [];

    return

end

%% ======================================================
% RESHAPE INTO DAYS
%% ======================================================
obs  = reshape(obs(1:samples_per_day*Nd), ...
               samples_per_day,Nd);

pred = reshape(pred(1:samples_per_day*Nd), ...
               samples_per_day,Nd);

%% ======================================================
% INIT
%% ======================================================
D.nRMSE = nan(Nd,1);
D.nMAE  = nan(Nd,1);
D.nMBE  = nan(Nd,1);
D.r2    = nan(Nd,1);

%% ======================================================
% LOOP DAYS
%% ======================================================
for d = 1:Nd

    o = obs(:,d);
    p = pred(:,d);

    %% --------------------------------------------------
    % DAILY ERROR
    %% --------------------------------------------------
    e = p - o;

    %% --------------------------------------------------
    % NORMALIZATION
    global_scale = mean(obs(:));

    if global_scale < 1e-6
        global_scale = 1;
    end
        
    D.nRMSE(d) = sqrt(mean(e.^2)) / global_scale;
    
    D.nMAE(d) = mean(abs(e)) / global_scale;
    
    D.nMBE(d) = mean(e) / global_scale;

    %% --------------------------------------------------
    % PEARSON r²
    %% --------------------------------------------------
    if std(o) > 1e-10 && std(p) > 1e-10

        C = corrcoef(o,p);

        D.r2(d) = C(1,2)^2;

    end

end

%% ======================================================
% REMOVE NaN
%% ======================================================
D.nRMSE = D.nRMSE(isfinite(D.nRMSE));
D.nMAE  = D.nMAE(isfinite(D.nMAE));
D.nMBE  = D.nMBE(isfinite(D.nMBE));
D.r2    = D.r2(isfinite(D.r2));

end