%% =========================================================
% GLOBAL OPTIMIZATION 
% LagH × m 
% lambda fixed = 10^(-4)
%% =========================================================
clear; clc; close all;

outdir = 'results_optimization_all_variables';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% ================= LOAD DATA =================

T = readtable('Data.xlsx','PreserveVariableNames',true);

DataList = {
    'PV',  max(0,T.('Solaire photovoltaïque (MW)')),  false;
    'WT',  max(0,T.('Eolien (MW)')),                  false;
    'PT',  max(0,T.('Production totale (MW)')),       false;
};

S = load('AJACCIO_station9_30MIN_T2M_GHI.mat');

DataList = [DataList; {
    'GHI_30min', max(0,S.GHI_30min),   true;
    'T2M_30min',       S.T2M_30min,          true;
}];

S = load('AJACCIO_station9_1H_T2M_GHI.mat');

DataList = [DataList; {
    'GHI_1h', max(0,S.GHI_1h),   false;
    'T2M_1h',       S.T2M_1h,    false;
}];

S = load('Bastia_Wind_30min.mat');

DataList = [DataList; {
    'WS_30min', S.FF_Bastia_30,   true;
}];

S = load('Bastia_Wind_60min.mat');

DataList = [DataList; {
    'WS_1h', S.FF_Bastia_60,   false;
}];


%% ================= GRID =================

LagH_base   = 24:24:72;
m_list      = 200:200:1600;

AllBest = [];

%% ================= LOOP VARIABLES =================

for v = 1:size(DataList,1)

    VarName = DataList{v,1};
    series  = DataList{v,2};
    is30min = DataList{v,3};

    fprintf('\n==============================\n');
    fprintf('Variable: %s\n', VarName);
    fprintf('==============================\n');

    % -------- Resolution handling
    if is30min
        H = 8760*2;
        LagH_list = LagH_base*2;
    else
        H = 8760;
        LagH_list = LagH_base;
    end

    nYears = floor(length(series)/H);
    if nYears < 2
        warning('Not enough years for %s',VarName);
        continue
    end

    annee = nYears-1;

    nL = numel(LagH_list);
    nM = numel(m_list);
    
    PACF_proj = nan(nL,nM);

    %% ================= GRID SEARCH =================
    for i = 1:nL

        LagH = LagH_list(i);
        fprintf(' LagH = %d\n',LagH);

        parfor j = 1:nM

            m = m_list(j);
        
            try
                [~,~,~,~,~,~,~,S_r1,~] = ...
                    Make_stationary_all_variables(series, annee, LagH, m, H);
        
                PACF_proj(i,j) = S_r1;
            catch
                PACF_proj(i,j) = NaN;
            end
        end
    end
    %% ================= FIND BEST (PROJECTION ONLY) =================
    M = PACF_proj;
    M(~isfinite(M)) = Inf;

    [bestVal, idx] = min(M(:));
    [iL,iM] = ind2sub(size(M),idx);

    bestLagH = LagH_list(iL);
    bestm    = m_list(iM);

    fprintf('\n BEST (Projection)\n');
    fprintf('LagH=%d | m=%d | PACF_proj=%.4f\n',...
        bestLagH,bestm,bestVal);

    %% ================= RE-EVALUATE BEST CONFIG =================
    [~,~,~,~,~,~,S_raw,S_r1,S_r2] = ...
        Make_stationary_all_variables(series, annee, bestLagH, bestm, H);

    fprintf('RAW=%.4f | Phase-only=%.4f\n',S_raw,S_r2);

    %% ================= SAVE RESULTS =================
    AllBest = [AllBest;
        {VarName,bestLagH,bestm,S_raw,S_r1,S_r2}
    ];

    %% ================= CONTOUR PLOT =================

    %% ================= CONTOUR PLOT =================

    Mplot = PACF_proj;
    [X,Y] = meshgrid(m_list,LagH_list);
    
    figure('Color','w');
    contourf(X,Y,Mplot,30,'LineColor','none');
    colormap(parula);
    colorbar;
    hold on;
    
    plot(bestm,bestLagH,'rp',...
        'MarkerSize',14,'MarkerFaceColor','w');
    
    xlabel('Hidden neurons');
    ylabel('LagH');
    title(['PACFsum Projection-' VarName], 'Interpreter','none');
    
    set(gca,'FontSize',12,'LineWidth',1);
    
    exportgraphics(gcf,...
        fullfile(outdir,['Contour_' VarName '.png']),...
        'Resolution',300);
    
    hold off;
   end
%% ================= SAVE CSV =================

BestTable = cell2table(AllBest,...
    'VariableNames',{
    'Variable','BestLagH','BestM',...
    'PACF_RAW','PACF_Projection','PACF_PhaseOnly'});

writetable(BestTable,...
    fullfile(outdir,'Best_Optimization_AllVariables.csv'));

disp('=== OPTIMIZATION FINISHED ===');
