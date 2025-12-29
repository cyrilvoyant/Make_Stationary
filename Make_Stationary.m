function [R_te2, ytrue_te, S_te2, R_te, S_te, h,S_raw,S_r1,S_r2,best_alpha1] = Make_Stationary(series, annee, LagH, m)
% MAKE_STATIONARY — Décomposition saisonnière par ELM (2 approches) et évaluation simple de la stationnarité.
%
% Objectif
%   Extraire une composante saisonnière S(t) d’une série cyclique horaire sur 2 ans (année TRAIN, année TEST),
%   en utilisant deux variantes ELM :
%     (1) Projection : entraînement sur [lags + phases], puis projection sur [0 + phases] pour isoler S.
%     (2) Phase-only : entraînement et inférence uniquement sur les phases (jour/année).
%   Puis, remettre S à l’échelle (alpha*), imposer des contraintes physiques simples, et quantifier la
%   “stationnarité cyclique” via un critère PACFsum (plus petit = plus stationnaire) calculé sur TEST.
%
% Hypothèses / conventions
%   - Séries horaires avec 2 années complètes (H = 8760 h/an), 50% pour %  train et 50% pour train
%   - Standardisation sur TRAIN uniquement (pas de fuite, on reste causal).
%   - Choix h (horizon court) par PACF(Z_train) borné à 24 h.
%   - ELM caché tanh, poids/biais aléatoires, sortie estimée via ols_fit (Ridge).
%   - Multi-start K pour sélectionner les poids par corrélation de Spearman (TRAIN, jour uniquement).
%   - Contraintes physiques : ytrue >= 0 ; S = 0 lorsque ytrue <= 0 ; clip S >= 0.
%   - Remise à l’échelle α* par minimisation MBE(ytrue_tr, α S_tr) (TRAIN), appliquée ensuite à TEST.
%   - Stationnarité : PACFsum = sum_k |PACF(k)|, k=1..48, calculé sur TEST.
%
% Entrées
%   series   : vecteur (>= 2*8760) des valeurs horaires brutes.
%   annee    : entier ≥ 1 ; fenêtre 2 ans = [ (annee-1) .. (annee) ] en indices année calée.
%   LagH     : nb de retards utilisés dans Xlags (p.ex. 24).
%   m        : nb de neurones cachés (p.ex. 100).
%
% Sorties
%   R_te     : résidu TEST de l’approche (1) Projection   = ytrue_te - S_te
%   S_te     : saison TEST      (1) Projection (après α*)
%   R_te2    : résidu TEST de l’approche (2) Phase-only   = ytrue_te - S_te2
%   S_te2    : saison TEST      (2) Phase-only (après α*)
%   ytrue_te : vérité TEST en unités physiques (>=0)
%   h        : horizon court (1..24) choisi via PACF(Z_train)
%   S_raw    : PACFsum du signal brut TEST  (masque jour)
%   S_r1     : PACFsum du résidu R_te       (masque jour)  — critère “stationnarité” pour l’approche 1
%   S_r2     : PACFsum du résidu R_te2      (masque jour)  — critère “stationnarité” pour l’approche 2
%
% Détails des deux approches
%   (1) Projection :
%       - Entraînement : H_tr = tanh([Xlags_tr, Xphase_tr]*W + b), beta = ols_fit(H_tr, y_tr)
%       - Projection :  H_Sproj = tanh([0, Xphase]*W + b)  →  S_Z = H_Sproj*beta  → S en unités physiques
%   (2) Phase-only :
%       - Entraînement/Inférence uniquement sur Xphase_* avec sélection multi-start par Spearman(train, jour)
%
% Critère de stationnarité (rapide, robuste, cyclique)
%   PACFsum(x) = sum_{k=1..48} |PACF_x(k)| sur TEST en mettant NaN la nuit (|x| < 2% max|ytrue_te|).
%   Interprétation : plus PACFsum ↓, plus la dépendance cyclique résiduelle est faible (série “stationnaire”).
%
%
% Complexité (ordre de grandeur)
%   - ELM : O(K * m * N) par approche (N ~ nb d’échantillons valides).
%
% Exemple d’appel
%   [R2, yte, S2, R1, S1, hh, Sraw, Sr1, Sr2] = Make_Stationary(x, 2, 24, 200);
%   % comparer Sr1/Sr2 vs Sraw ; plus petit = mieux
%
% Auteur / Notes
%   - Conçu pour séries solaires/énergétiques périodique, sans clear-sky, avec contraintes physiques minimales.
%   - Style “frugal”: pas d’itératifs lourds, pas de modèles externes, métrique simple et lisible.

%% 0) Paramètres et garde-fous — structure minimale des données et des hyperparamètres
% - Exige 2 années horaires complètes (2 × 8760 pts).
% - LagH : nombre de retards utilisés dans le modèle complet (par défaut 24 h).
% - m    : nombre de neurones cachés ELM (par défaut 100).
% - Vérifie existence/validité des données et empêche l’exécution si la variance TRAIN est nulle.
H = 8760;                   % heures par an
need = 2*H;
if nargin < 3 || isempty(LagH), LagH = 24; end
if nargin < 4 || isempty(m),    m    = 100; end
if numel(series) < need
    error('Pas assez de points: il faut au moins %d échantillons horaires.', need);
end

%% 1) Extraction de la fenêtre 2 ans, normalisation TRAIN-only et choix de l’horizon court h via PACF
% - Sélectionne l’année TRAIN + l’année TEST consécutive.
% - Normalise uniquement sur TRAIN (pas de fuite d’information).
% - Détermine h (1–24 h) comme premier lag dont le PACF rentre dans les bornes 95 %, sinon lag au PACF minimal.
series = series(H*(annee-1)+1 : H*(annee+1));
series_train = series(1:H);
series_test  = series(H+1:end);

mu_train = mean(series_train, 'omitnan');
sd_train = std(series_train,  'omitnan');
if ~isfinite(sd_train) || sd_train <= eps
    error('Ecart-type train nul/non défini (%.3g).', sd_train);
end

Z_train = (series_train - mu_train) ./ sd_train;
Z_test  = (series_test  - mu_train) ./ sd_train;

maxLag = 24;
h = 1;
try
    [pacf,~,bounds] = parcorr(Z_train, 'NumLags', maxLag);
    p = pacf(2:end); bb = max(abs(bounds));
    idx = find(abs(p) < bb, 1, 'first');
    if ~isempty(idx), h = idx; else [~,h] = min(abs(p)); end
catch
    h = 1;
end

%% 2) Construction des features : retards (lags) et encodages de phases (jour/année)
% - Génère Xlags à partir de LagH retards sur Z_train/Z_test.
% - Encode les phases diurnes et annuelles via (cos,sin) pour préserver la circularité.
% - Définit les indices TRAIN/TEST post-h afin d’aligner correctement cibles et caractéristiques.
idx_tr = (LagH+1) : (H - h);
idx_te = (LagH+1) : (H - h);

Xlags_tr = zeros(numel(idx_tr), LagH+1);
Xlags_te = zeros(numel(idx_te), LagH+1);
for i = 0:LagH
    Xlags_tr(:, i+1) = Z_train(idx_tr - i);
    Xlags_te(:, i+1) = Z_test( idx_te - i);
end

t_tr = idx_tr + h - 1;     
t_te = idx_te + h - 1;

day_tr = floor(t_tr/24) + 1; day_tr(day_tr > 365) = day_tr(day_tr > 365) - 365;
day_te = floor(t_te/24) + 1; day_te(day_te > 365) = day_te(day_te > 365) - 365;

phi_d_tr = mod(t_tr, 24) / 24;  phi_d_te = mod(t_te, 24) / 24;
phi_y_tr = (day_tr - 1) / 365;  phi_y_te = (day_te - 1) / 365;

phi_d_tr = phi_d_tr(:); phi_y_tr = phi_y_tr(:);
phi_d_te = phi_d_te(:); phi_y_te = phi_y_te(:);

Xphase_tr = [cos(2*pi*phi_d_tr), sin(2*pi*phi_d_tr), ...
             cos(2*pi*phi_y_tr), sin(2*pi*phi_y_tr)];
Xphase_te = [cos(2*pi*phi_d_te), sin(2*pi*phi_d_te), ...
             cos(2*pi*phi_y_te), sin(2*pi*phi_y_te)];

y_tr = Z_train(idx_tr + h);
y_te = Z_test( idx_te  + h);

%% 3) Construction des matrices d’entrée pour les deux modèles ELM
% - Modèle complet (C) : concatène lags + phases → capte dynamique + cyclique.
% - Modèle saison (S_proj) : même architecture mais lags mis à zéro → isole la part purement cyclique.
% - d_C et d_S = dimensions effectives des deux espaces d’entrée.
X_tr_C = [Xlags_tr, Xphase_tr];
X_te_C = [Xlags_te, Xphase_te];
d_C = size(X_tr_C, 2);

d_S = size(Xphase_tr, 2);

X_tr_S_proj = [zeros(size(Xlags_tr)), Xphase_tr];
X_te_S_proj = [zeros(size(Xlags_te)), Xphase_te];

%% 5) ELM Saisonnalité (S) séparé — multi-start avec sélection Spearman (TRAIN)
% Objectif : estimer la composante strictement cyclique en apprenant uniquement
% sur les features de phase (SANS lags).  
% → On lance K initialisations (W,b) aléatoires.  
% → Pour chaque run : apprentissage ridge (ols_fit), reconstruction en MW, clip physique.  
% → Critère de sélection : corrélation de Spearman sur TRAIN (jour uniquement).  
% Le triplet (W_S , b_S , beta_S) maximisant Spearman est retenu.
K = 1000;                         % nb d'initialisations
best_rho = -Inf;
bestW_S = []; bestb_S = []; bestbeta_S = [];

rho_all  = -Inf(K,1);
W_all    = cell(K,1);
b_all    = cell(K,1);
beta_all = cell(K,1);

parfor k = 1:K
    % Tirage des poids cachés
    Wk = -1 + 2*rand(d_S, m);
    bk = -1 + 2*rand(1, m);

    % Apprentissage sur TRAIN
    Hk_tr = tanh(Xphase_tr*Wk + bk);
    betak = ols_fit(Hk_tr, y_tr);          % ridge via

    % Sortie TRAIN (Z), passage en unités physiques + clip
    S_tr_Z2_k = Hk_tr * betak;
    S_tr_k    = max(0, S_tr_Z2_k*sd_train + mu_train);

    % Vérité TRAIN (physique, mesure) pour le critère
    ytrue_tr_k = y_tr*sd_train + mu_train;

    % Masque "jours" et valeurs finies (évite pollution des nuits)
    M = (ytrue_tr_k > 0.05 * max(ytrue_tr_k, [], 'omitnan')) & isfinite(S_tr_k) & isfinite(ytrue_tr_k);
    if nnz(M) < 10
        continue
    end

    % Critère : Spearman sur TRAIN (robuste à l'échelle)
    rho = corr(S_tr_k(M), ytrue_tr_k(M), 'Type','Spearman', 'Rows','pairwise');
    if isfinite(rho) 
        rho_all(k)  = rho;
        W_all{k}    = Wk;
        b_all{k}    = bk;
        beta_all{k} = betak;
    end
end

[best_rho, kbest] = max(rho_all);
bestW_S    = W_all{kbest};
bestb_S    = b_all{kbest};
bestbeta_S = beta_all{kbest};

% Sécurité : si aucun rho valide n'a été trouvé, on fait au moins une init
if isempty(bestW_S)
    bestW_S = -1 + 2*rand(d_S, m);
    bestb_S = -1 + 2*rand(1, m);
    H_tr_S  = tanh(Xphase_tr*bestW_S + bestb_S);
    bestbeta_S = ols_fit(H_tr_S, y_tr);
else
    H_tr_S = tanh(Xphase_tr*bestW_S + bestb_S);
end

% Sorties finales (Z) avec les meilleurs poids
H_te_S  = tanh(Xphase_te*bestW_S + bestb_S);
beta_S  = bestbeta_S;                 % << demandé : renvoyer le "meilleur" beta_S
S_tr_Z2 = H_tr_S * beta_S;
S_te_Z2 = H_te_S * beta_S;

%% 6) Approche 1 : Saison par projection (C → S_proj) — multi-start + Spearman (TRAIN)
% Objectif : estimer la saisonnalité en réutilisant l’ELM complet (lags+phases),
% puis en projetant ses poids sur une entrée où les lags sont annulés. Procédure :
% 1) Multi-start : tirage aléatoire (W_C , b_C) sur le modèle complet.
% 2) Fit ridge (ols_fit) sur TRAIN avec les features complètes.
% 3) Projection : évaluer le même modèle sur X_tr_S_proj (lags=0) → composante cyclique.
% 4) Reconstruction en MW + clip physique.
% 5) Sélection du run via Spearman sur TRAIN (jours uniquement).
% Le triplet gagnant (W_C , b_C , beta_C) fournit la saisonnalité projetée.
K = 1000; 
best_rho_C = -Inf;
bestW_C = []; bestb_C = []; bestbeta_C = [];

rho_all_C  = -Inf(K,1);
W_all_C    = cell(K,1);
b_all_C    = cell(K,1);
beta_all_C = cell(K,1);

parfor k = 1:K
    % Tirage des poids cachés du modèle complet (lags + phases)
    Wk = -1 + 2*rand(d_C, m);
    bk = -1 + 2*rand(1, m);

    % Fit sur TRAIN (modèle complet)
    Hk_tr = tanh(X_tr_C*Wk + bk);
    betak = ols_fit(Hk_tr, y_tr);                    % ridge via ta version

    % Projection "saison" sur TRAIN avec ces poids
    Hk_Sproj_tr = tanh(X_tr_S_proj*Wk + bk);
    S_tr_Z_k    = Hk_Sproj_tr * betak;               % en Z
    S_tr_k      = max(0, S_tr_Z_k*sd_train + mu_train);

    % Vérité TRAIN (physique) pour le critère
    ytrue_tr_k  = y_tr*sd_train + mu_train;

    % Masque "jours" (évite pollution des nuits) + finitudes
    M = (ytrue_tr_k > 0.05 * max(ytrue_tr_k, [], 'omitnan')) & isfinite(S_tr_k) & isfinite(ytrue_tr_k);
    if nnz(M) < 10
        continue
    end

    % Critère : Spearman sur TRAIN
    rho = corr(S_tr_k(M), ytrue_tr_k(M), 'Type','Spearman', 'Rows','pairwise');
    if isfinite(rho) 
        rho_all_C(k)  = rho;
        W_all_C{k}    = Wk;
        b_all_C{k}    = bk;
        beta_all_C{k} = betak;
    end
end

[best_rho_C, kbest] = max(rho_all_C);
bestW_C     = W_all_C{kbest};
bestb_C     = b_all_C{kbest};
bestbeta_C  = beta_all_C{kbest};

% Fallback si aucune init valide
if isempty(bestW_C) || isempty(bestb_C) || isempty(bestbeta_C) || ~isfinite(best_rho_C)
    bestW_C = -1 + 2*rand(d_C, m);
    bestb_C = -1 + 2*rand(1, m);
    H_tr_C  = tanh(X_tr_C*bestW_C + bestb_C);
    bestbeta_C = ols_fit(H_tr_C, y_tr);
end

% Matrices finales avec les meilleurs poids (TRAIN/TEST)
H_S_proj_tr = tanh(X_tr_S_proj*bestW_C + bestb_C);
H_S_proj_te = tanh(X_te_S_proj*bestW_C + bestb_C);

% Sorties Z (comme avant) + mise à dispo des meilleurs paramètres
beta_C = bestbeta_C;        % meilleur beta_C
W_C    = bestW_C;           % meilleurs poids
b_C    = bestb_C;

S_tr_Z = H_S_proj_tr * beta_C;
S_te_Z = H_S_proj_te * beta_C;


% Passage en MW et clip >=0
S_tr  = max(0, S_tr_Z  * sd_train + mu_train);
S_te  = max(0, S_te_Z  * sd_train + mu_train);
S_tr2 = max(0, S_tr_Z2 * sd_train + mu_train);
S_te2 = max(0, S_te_Z2 * sd_train + mu_train);

%% 7) Vérités (MW) et contraintes physiques
% - Dénormalise les cibles en unités physiques (ytrue_TR/TE).
% - Contrainte physique : Y ≥ 0.
% - Contrainte de cohérence : S = 0 lorsque Y ≤ 0 (nuits/masques) pour chaque approche.
% - Re-clip final S ≥ 0 pour éliminer toute valeur négative résiduelle.
ytrue_tr = Z_train(idx_tr + h)*sd_train + mu_train;
ytrue_te = Z_test( idx_te  + h)*sd_train + mu_train;

% Y >= 0
ytrue_tr(ytrue_tr < 0) = 0;
ytrue_te(ytrue_te < 0) = 0;

% --- Seuil relatif : 5% du max global (TRAIN + TEST) → robuste multi-variables ---
max_global = max([ytrue_tr; ytrue_te], [], 'omitnan');
seuil_nuit = 0.05 * max_global;   % 5% du maximum observé

% Forcer ytrue à 0 en dessous du seuil (bruit + crépuscule/calme)
ytrue_tr(ytrue_tr < seuil_nuit) = 0;
ytrue_te(ytrue_te < seuil_nuit) = 0;

% --- Contrainte de cohérence : S = 0 partout où ytrue == 0 ---
t1 = (ytrue_tr == 0);
t2 = (ytrue_te == 0);

S_tr(t1)  = 0;  S_te(t2)  = 0;
S_tr2(t1) = 0;  S_te2(t2) = 0;

% Ré-clip physique simple (cohérence Y>=0)
S_tr  = max(0, S_tr);
S_te  = max(0, S_te);
S_tr2 = max(0, S_tr2);
S_te2 = max(0, S_te2);

%%  Remise à l’échelle (MBE sur TRAIN) conservation energie totale
% Objectif : ajuster l’amplitude de la saisonnalité S_tr (et S_tr2) par un scalaire α.
% Méthode : recherche discrète α ∈ [0.01, 3.00], pas 0.01.
% Pour chaque α :
%     erreur(α) = MBE( ytrue_tr , α * S_tr )
% On retient l’α minimisant la valeur absolue de MBE, puis on applique la même mise à l’échelle au TEST.
A = (1:300)'/100;                                % 0.01:0.01:3.00
E = ytrue_tr(:)' - S_tr(:)' .* A;           % matrice (300 x N)
[~, idx] = min(mean(abs(E), 2, 'omitnan'));
best_alpha1 = A(idx);

S_tr = best_alpha1 * S_tr;
S_te = best_alpha1 * S_te;

A = (1:300)'/100;                                % 0.01:0.01:3.00
E = ytrue_tr(:)' - S_tr2(:)' .* A;          % matrice (300 x N)
[~, idx] = min(mean(abs(E), 2, 'omitnan'));
best_alpha2 = A(idx);

S_tr2 = best_alpha2 * S_tr2;
S_te2 = best_alpha2 * S_te2;

%% 8) Résidus
% Calcul des séries résiduelles :
%   R_tr  = ytrue_tr  − S_tr     (approche 1)
%   R_te  = ytrue_te  − S_te
%   R_tr2 = ytrue_tr  − S_tr2    (approche 2)
%   R_te2 = ytrue_te  − S_te2
% Les résidus sont ensuite exploités pour évaluer la stationnarité.
R_tr  = ytrue_tr - S_tr;  
R_te  = ytrue_te - S_te;
R_tr2 = ytrue_tr - S_tr2;
R_te2 = ytrue_te - S_te2;

%% 9) PACFsum – stationnarité cyclique
% Critère simple et robuste : somme des amplitudes PACF, hors lag 0, après masquage des nuits.
% Interprétation :
%   – Plus la série est stationnaire, plus la dépendance résiduelle (PACF) s’effondre rapidement.
%   – PACFsum = somme |PACF(lag)| sur les 48 premiers lags → mesure globale de “l’aire sous la
%     courbe des dépendances”. 
% Sens physique :
%   – PACF élevé = présence de structure déterministe répétitive (cycle jour/nuit, hystérésis nuageuse…)
%   – PACFsum faible = série “blanchie”, donc saisonnalité correctement retirée.
% Ce n’est pas une métrique statistique “officielle”, mais un indicateur cohérent
% pour des séries cycliques irradiance/puissance lorsque l’objectif est de minimiser
% la persistance structurelle résiduelle.

%thr  = 0.05 * max(abs(ytrue_te),[],'omitnan');   % seuil 5%
%mask = abs(ytrue_te) >= thr;                     % jour uniquement

%clean = @(x) (x .* mask);                        % NaN sur la nuit

maxLag = 48;

% PACF masqué
pacf_raw = parcorr(ytrue_te, 'NumLags', maxLag);
pacf_r1  = parcorr(R_te,     'NumLags', maxLag);
pacf_r2  = parcorr(R_te2,    'NumLags', maxLag);

% Somme des amplitudes PACF (hors lag0 = pacf(1))
PACFsum = @(p) nansum(abs(p(2:maxLag+1)));

S_raw = PACFsum(pacf_raw);
S_r1  = PACFsum(pacf_r1);
S_r2  = PACFsum(pacf_r2);

fprintf('\n=== PACFsum (plus petit = plus stationnaire) ===\n');
fprintf(' RAW : %.3f\n', S_raw);
fprintf(' R_1 : %.3f\n', S_r1);
fprintf(' R_2 : %.3f\n', S_r2);

end
