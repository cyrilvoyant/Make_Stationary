%% ================= LOAD DATA =================
clear; clc; close all;

outdir = 'deseasonalisation_plot_results';
if ~exist(outdir,'dir'), mkdir(outdir); end

T = readtable('C:/Users/HP/Downloads/Prof_Cyril Voyant_Proposition_Collaboration/Maklewa/final_deseasonalization_code_22_05_2026/results_final_pipeline/Metrics.csv');

Methods = T.Signal;

% Method names
Methods = strrep(Methods,'EL_Proj','EL-Proj');
Methods = strrep(Methods,'EL_Phase','EL-Phase');

%% ================= VARIABLES =================
Vars = {'GHI_30min','GHI_1h','PV','WT','PT','WS_30min','WS_1h','T2M_30min','T2M_1h'};
Vars_display = {'GHI30','GHI1h','PV','WT','PT','WS30','WS1h','T2M30','T2M1h'};

MetricsNames = {'PACFsum','Entropy','LLE'};

nVar = numel(Vars);
nMet = numel(Methods);

%% ================= COLORS =================
colors = lines(nMet);

%% ================= FIGURE =================
figure('Color','w','Position',[100 50 1400 1800]);

for v = 1:nVar

    var = Vars{v};

    % Extract data
    PACF = T.([var '_PACF']);
    ENT  = T.([var '_Entropy']);
    LLE  = T.([var '_LLE']);

    DATA = {PACF, ENT, LLE};

    for m = 1:3

        subplot(nVar,3,(v-1)*3 + m)

        b = bar(DATA{m},'FaceColor','flat');

        % Assign colors
        for k = 1:nMet
            b.CData(k,:) = colors(k,:);
        end

        xticks(1:nMet)

        % ===== SHOW X LABELS ONLY LAST ROW =====
        if v == nVar
            xticklabels(Methods)
            xtickangle(45)
        else
            xticklabels([])   % remove labels
        end

        % ===== TITLES ONLY FIRST ROW =====
        if v == 1
            title(MetricsNames{m},'FontWeight','bold')
        end

        % ===== Y LABEL ONLY FIRST COLUMN =====
        if m == 1
            ylabel(Vars_display{v},'FontWeight','bold')
        end

        grid on
        set(gca,'FontSize',10)

    end
end

%% ================= SAVE =================
exportgraphics(gcf, fullfile(outdir,'ALL_VARIABLES_BARPLOT.png'), 'Resolution',300);

disp('Figure saved');

