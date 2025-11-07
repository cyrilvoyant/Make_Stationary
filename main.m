%% LOAD (index horaire, sans timezone) // PV Corse
clear all;
file = 'Data.xlsx';
colPV = 'Solaire photovoltaïque (MW)';

T = readtable(file, 'PreserveVariableNames', true);
pv = T.(colPV);
pv(pv < 0) = 0;  % clamp simple
series = pv;
horizon = 24; %ELM construit avec "horizon" lag
Nbre_Hidden = 500; %neurones cachés
annee = 2; % année de test
%2ans, 1 an pour la desais et l'autre pour le test
[desais, Raw, Seasonal_Trend,desais_Proj, Seasonal_Trend_Proj, Lyap,Coeff_Statio_Raw,Coeff_Statio_Proj,Coeff_Statio] = Make_Stationary(series,annee,horizon,Nbre_Hidden);

%% GHI Ajaccio

clear all;
T = readtable('GLO_ajaccio.txt');
series=T.Var3;
horizon = 24; %ELM construit avec "horizon" lag
Nbre_Hidden = 500; %neurones cachés
%rng(42);
annee = 6; % année de test
%2ans, 1 an pour la desais et l'autre pour le test
[desais, Raw, Seasonal_Trend,desais_Proj, Seasonal_Trend_Proj, Lyap,Coeff_Statio_Raw,Coeff_Statio_Proj,Coeff_Statio] = Make_Stationary(series,annee,horizon,Nbre_Hidden);

%% Optimisation horizon vs Nbre_Hidden exemple Avec Data GHI Ajaccio

clear all;
T = readtable('GLO_ajaccio.txt');
series=T.Var3;
annee = 6;
Opti_Statio = nan(10,40);
for horizon = 1:1:10
    parfor Nbre_Hidden = 1:1:40
        LagH = 5*horizon;     % 4,8,...,40
        m    = 50*Nbre_Hidden;  % 100,...,1000
        
            [~,~,~,~,~,~,~,S_r1,~] = Make_Stationary(series, annee, LagH, m);
            Opti_Statio(horizon, Nbre_Hidden) = S_r1;   % critère = PACFsum projection
    end
end

% Analyse du résultat de la grille

M = Opti_Statio;                      % 10x10, damier NaN/valeurs (indices impairs remplis)
M(~isfinite(M)) = Inf;                % NaN -> Inf pour la recherche du min

[bestVal, linIdx] = min(M(:));
if ~isfinite(bestVal)
    error('Aucune valeur finie dans Opti_Statio.');
end
[bestH_idx, bestN_idx] = ind2sub(size(M), linIdx);

bestLagH = 5 * bestH_idx;             % horizons testés: 5,10,...,50
bestM    = 50 * bestN_idx;           % neurones testés: 200,400,...,2000
fprintf('\n=== BEST ===\nLagH=%d h | m=%d | PACFsum=%.4f\n', bestLagH, bestM, bestVal);


% Sous-grille utilisée (indices impairs)
hIdx = 1:1:10;  mIdx = 1:1:40;
Mred = Opti_Statio(hIdx, mIdx);

Hvals = 5 * hIdx;                     % Y (heures)
Mvals = 50 * mIdx;                   % X (neurones)
[X,Y] = meshgrid(Mvals, Hvals);

% Surface 3D
figure('Color','w');
surf(X, Y, Mred, 'EdgeColor','none'); shading interp; colormap(parula); colorbar;
xlabel('m (neurones cachés)'); ylabel('LagH (heures)'); zlabel('PACFsum (↓ mieux)');
title('Optimisation LagH–m (sous-grille impaire)');
view(135,30); grid on; box on;

% Marqueur du minimum (projeter sur la sous-grille si dedans)
hold on;
if ismember(bestH_idx,hIdx) && ismember(bestN_idx,mIdx)
    [~,iH] = ismember(bestH_idx,hIdx);
    [~,iM] = ismember(bestN_idx,mIdx);
    plot3(Mvals(iM), Hvals(iH), Mred(iH,iM), 'rp', 'MarkerFaceColor','w', 'MarkerSize',12);
end
hold off;

% Carte de contours
figure('Color','w');
contourf(X, Y, Mred, 20, 'LineColor','none'); colormap(parula); colorbar;
xlabel('m (neurones cachés)'); ylabel('LagH (heures)');
title('Carte PACFsum (plus sombre = meilleur)'); grid on; box on;