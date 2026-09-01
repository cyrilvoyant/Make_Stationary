function [R_te2, ytrue_te, S_te2, R_te, S_te, h, ...
          S_raw, S_r1, S_r2, best_alpha1] = ...
          Make_stationary_all_variables(series, annee, LagH, m, H)

% =========================================================================
% MAKE_STATIONARY_ALL_VARIABLES — Décomposition saisonnière avec ELM 
%
% OBJECTIF
%   Extraire la composante saisonnière S(t) d'une série énergétique ou
%   météorologique (PV, WT, PT, GHI, WS, T2M)
%   sur deux années consécutives :
%       - 1 an TRAIN
%       - 1 an TEST
%
% Deux approches :
%   (1) Phase-only  : saison pure basée uniquement sur cycles jour/année
%   (2) Projection  : modèle complet (lags+phases) puis projection cyclique
%
% ADAPTATION AUTOMATIQUE
%   - Détection statistique des signaux "solaire-like"
%   - Application conditionnelle des contraintes physiques
%   - Rescaling α uniquement si physiquement justifié
%
% STATIONNARITÉ
%   Mesurée par :
%       PACFsum = somme des |PACF(k)|, k=1..48
%   Plus PACFsum est faible → plus la série est stationnaire.
%
%
% =========================================================================


%% ===================== 0) Vérifications =====================

need = 2*H;
if numel(series) < need
    error('Pas assez de données.');
end

%% ===================== 1) Séparation TRAIN/TEST =====================

series = series(H*(annee-1)+1 : H*(annee+1));

series_train = series(1:H);
series_test  = series(H+1:end);

mu_train = mean(series_train,'omitnan');
sd_train = std(series_train,'omitnan');

if sd_train <= eps || ~isfinite(sd_train)
    error('Ecart-type invalide.');
end

Z_train = (series_train - mu_train)/sd_train;
Z_test  = (series_test  - mu_train)/sd_train;

%% ===================== 2) Détection type de variable =====================

isNonNegative = min(series) >= 0;
zeroRatio = mean(series < 0.02*max(series));
isSolarLike = isNonNegative && zeroRatio > 0.25;

% Interprétation :
%   isSolarLike = TRUE si :
%       - variable non négative
%       - forte proportion de zéros (comportement jour/nuit)
%
%   Typiquement :
%       TRUE  → GHI, PV
%       FALSE → WS, T2M, WT, PT
%
% Cette détection évite d'imposer des contraintes physiques
% inadaptées aux variables non solaires.


%% ===================== 3) Choix h via PACF =====================
% h = horizon court déterminé automatiquement via PACF sur TRAIN.
%
% Logique :
%   On cherche le premier lag où la dépendance devient non significative.
%
% Ce h sert à :
%   - Décaler la cible
%   - Éviter fuite d'information
%   - Stabiliser l’apprentissage

maxLag_h = 24;
h = 1;

try
    [pacf,~,bounds] = parcorr(Z_train,'NumLags',maxLag_h);
    p = pacf(2:end);
    seuil = max(abs(bounds));
    idx = find(abs(p)<seuil,1,'first');
    if isempty(idx)
        [~,idx] = min(abs(p));
    end
    h = idx;
catch
    h = 1;
end

%% ===================== 4) Construction features =====================

idx_tr = (LagH+1):(H-h);
idx_te = (LagH+1):(H-h);

Xlags_tr = zeros(numel(idx_tr), LagH+1);
Xlags_te = zeros(numel(idx_te), LagH+1);

for i=0:LagH
    Xlags_tr(:,i+1) = Z_train(idx_tr - i);
    Xlags_te(:,i+1) = Z_test(idx_te - i);
end

t_tr = idx_tr + h - 1;
t_te = idx_te + h - 1;
% ----------- Resolution-sure phases -----------

if H == 8760*4
    Nd = 96;           % 15min
    Ny = 365*96;
else
    Nd = 24;           % hourly
    Ny = 365*24;
end
%------------------------------------------------%

phi_d_tr = mod(t_tr,Nd)/Nd;
phi_d_te = mod(t_te,Nd)/Nd;

phi_y_tr = mod(t_tr,Ny)/Ny;
phi_y_te = mod(t_te,Ny)/Ny;

Xphase_tr = [cos(2*pi*phi_d_tr(:)) sin(2*pi*phi_d_tr(:)) ...
             cos(2*pi*phi_y_tr(:)) sin(2*pi*phi_y_tr(:))];

Xphase_te = [cos(2*pi*phi_d_te(:)) sin(2*pi*phi_d_te(:)) ...
             cos(2*pi*phi_y_te(:)) sin(2*pi*phi_y_te(:))];

y_tr = Z_train(idx_tr + h);
y_te = Z_test(idx_te + h);

%% ===================== 5) ELM phase-only =====================
% Multi-start ELM :
%   Les poids cachés étant aléatoires,
%   on teste K initialisations différentes.
%
% Sélection basée sur :
%   Corrélation de Spearman.
%
% On retient la configuration maximisant
% la cohérence cyclique sur TRAIN.
lambda=10^(-4);
K = 500;
d_S = size(Xphase_tr,2);
best_rho = -Inf;

for k=1:K
    W = -1+2*rand(d_S,m);
    b = -1+2*rand(1,m);

    Htr = tanh(Xphase_tr*W + b);
    [beta,beta0] = ols_fit(Htr,y_tr,lambda);

    S_tr_Z = Htr*beta+ beta0;
    S_tr_phys = S_tr_Z*sd_train + mu_train;
    ytrue_tr = y_tr*sd_train + mu_train;

    M = isfinite(S_tr_phys) & isfinite(ytrue_tr);

    if nnz(M)<10, continue; end

    rho = corr(S_tr_phys(M),ytrue_tr(M),...
               'Type','Spearman','Rows','pairwise');

    if isfinite(rho) && rho>best_rho
        best_rho=rho; bestW=W; bestb=b; bestbeta=beta; bestbeta0=beta0;
    end
end

Htr = tanh(Xphase_tr*bestW + bestb);
Hte = tanh(Xphase_te*bestW + bestb);

S_tr2 = (Htr*bestbeta + bestbeta0)*sd_train + mu_train;
S_te2 = (Hte*bestbeta + bestbeta0)*sd_train + mu_train;

%% ===================== 6) ELM projection =====================
% Projection :
%   On entraîne un modèle complet (lags + phases),
%   puis on annule les lags pour extraire
%   uniquement la composante cyclique.
%
% Cela permet de séparer :
%       dynamique court terme
%       saisonnalité déterministe

d_C = size([Xlags_tr Xphase_tr],2);
best_rho = -Inf;

for k=1:K
    W=-1+2*rand(d_C,m);
    b=-1+2*rand(1,m);

    Htr_full = tanh([Xlags_tr Xphase_tr]*W+b);
    [beta,beta0] = ols_fit(Htr_full,y_tr,lambda);

    Hproj = tanh([zeros(size(Xlags_tr)) Xphase_tr]*W+b);
    S_tr_Z = Hproj*beta + beta0;

    S_tr_phys = S_tr_Z*sd_train + mu_train;
    ytrue_tr  = y_tr*sd_train + mu_train;

    M = isfinite(S_tr_phys) & isfinite(ytrue_tr);
    if nnz(M)<10, continue; end

    rho = corr(S_tr_phys(M),ytrue_tr(M),...
               'Type','Spearman','Rows','pairwise');

    if isfinite(rho) && rho>best_rho
        best_rho=rho; bestW=W; bestb=b; bestbeta=beta; bestbeta0=beta0;
    end
end

Hproj_tr = tanh([zeros(size(Xlags_tr)) Xphase_tr]*bestW+bestb);
Hproj_te = tanh([zeros(size(Xlags_te)) Xphase_te]*bestW+bestb);

S_tr = (Hproj_tr*bestbeta + bestbeta0)*sd_train + mu_train;
S_te = (Hproj_te*bestbeta + bestbeta0)*sd_train + mu_train;

%% ===================== 7) Contraintes adaptatives =====================
% Contraintes physiques appliquées UNIQUEMENT
% si comportement solaire détecté.
%
% Cela évite :
%   - biaiser température
%   - écraser vent
%   - forcer positivité inutile

ytrue_tr = y_tr*sd_train + mu_train;
ytrue_te = y_te*sd_train + mu_train;

if isSolarLike
    
    % --- Clip physique ---
    ytrue_tr = max(0,ytrue_tr);
    ytrue_te = max(0,ytrue_te);
    
    % --- Masque faible intensité ---
    seuil = 0.05*max([ytrue_tr; ytrue_te]);
    ytrue_tr(ytrue_tr<seuil)=0;
    ytrue_te(ytrue_te<seuil)=0;
    
    S_tr(ytrue_tr==0)=0;
    S_te(ytrue_te==0)=0;
    S_tr2(ytrue_tr==0)=0;
    S_te2(ytrue_te==0)=0;
    
    % --- Rescaling α uniquement pour solaire ---
    A=(1:300)'/100;
    E=ytrue_tr(:)'-S_tr(:)'.*A;
    [~,id]=min(mean(abs(E),2,'omitnan'));
    best_alpha1=A(id);
    
    S_tr=best_alpha1*S_tr;
    S_te=best_alpha1*S_te;
    
    E2 = ytrue_tr(:)' - S_tr2(:)'.*A;
    [~,id2] = min(mean(abs(E2),2,'omitnan'));
    alpha2 = A(id2);
    
    S_tr2 = alpha2*S_tr2;
    S_te2 = alpha2*S_te2;

    
else
    
    % Pas de contrainte physique
    best_alpha1 = 1;
    
end

%% ===================== 8) Résidus =====================

R_te  = ytrue_te - S_te;
R_te2 = ytrue_te - S_te2;

%% ===================== 9) PACF =====================

if H==8760*4
    maxLag=48*4;
else
    maxLag=48;
end

pacf_raw = parcorr(ytrue_te,'NumLags',maxLag);
pacf_r1  = parcorr(R_te,'NumLags',maxLag);
pacf_r2  = parcorr(R_te2,'NumLags',maxLag);

PACFsum=@(p) nansum(abs(p(2:end)));

S_raw=PACFsum(pacf_raw);
S_r1 =PACFsum(pacf_r1);
S_r2 =PACFsum(pacf_r2);
fprintf('\n=== PACFsum (plus petit = plus stationnaire) ===\n');
fprintf(' RAW         : %.4f\n', S_raw);
fprintf(' Projection  : %.4f\n', S_r1);
fprintf(' Phase-only  : %.4f\n', S_r2);

% if isSolarLike
%     fprintf(' Variable detectee comme SOLAIRE\n');
% else
%     fprintf(' Variable detectee comme NON-SOLAIRE\n');
% end

end




