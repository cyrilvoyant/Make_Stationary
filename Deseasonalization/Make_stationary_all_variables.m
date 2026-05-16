function [R_te2, ytrue_te, S_te2, R_te, S_te, h, ...
          S_raw, S_r1, S_r2, best_alpha1, Model] = ...
          Make_stationary_all_variables(series, annee, LagH, m, H)

Model = struct();
best_alpha1 = 1;   

%% ===================== DATA =====================
series = series(:);
series = series(H*(annee-1)+1 : H*(annee+1));

series_train = series(1:H);
series_test  = series(H+1:end);

mu_train = mean(series_train,'omitnan');
sd_train = std(series_train,'omitnan');

if sd_train <= eps || ~isfinite(sd_train)
    sd_train = 1; 
end

Z_train = (series_train - mu_train)/sd_train;
Z_test  = (series_test  - mu_train)/sd_train;

Z_train(~isfinite(Z_train)) = 0;
Z_test(~isfinite(Z_test))   = 0;

%% ===================== TYPE =====================
isNonNegative = min(series) >= 0;
zeroRatio = mean(series < 0.02*max(series));
isSolarLike = isNonNegative && zeroRatio > 0.25;

%% ===================== H =====================
h = 1;
Model.h = h;

%% ===================== FEATURES =====================
idx_tr = (LagH+1):(H-h);
idx_te = (LagH+1):(H-h);

% LAGS
Xlags_tr = zeros(numel(idx_tr), LagH+1);
Xlags_te = zeros(numel(idx_te), LagH+1);

for i=0:LagH
    Xlags_tr(:,i+1) = Z_train(idx_tr - i);
    Xlags_te(:,i+1) = Z_test(idx_te - i);
end

% PHASES
if H == 8760*4
    Nd = 96; Ny = 365*96;
else
    Nd = 24; Ny = 365*24;
end

t_tr = idx_tr + h - 1;
t_te = idx_te + h - 1;

phi_d_tr = mod(t_tr,Nd)/Nd;
phi_y_tr = mod(t_tr,Ny)/Ny;

phi_d_te = mod(t_te,Nd)/Nd;
phi_y_te = mod(t_te,Ny)/Ny;

Xphase_tr = [cos(2*pi*phi_d_tr(:)) sin(2*pi*phi_d_tr(:)) ...
             cos(2*pi*phi_y_tr(:)) sin(2*pi*phi_y_tr(:))];

Xphase_te = [cos(2*pi*phi_d_te(:)) sin(2*pi*phi_d_te(:)) ...
             cos(2*pi*phi_y_te(:)) sin(2*pi*phi_y_te(:))];

y_tr = Z_train(idx_tr + h);
y_te = Z_test(idx_te + h);

%% ===================== ELM PHASE =====================
lambda = 1e-4;
K = 200;
best_err = Inf;

for k=1:K
    W = -1+2*rand(size(Xphase_tr,2),m);
    b = -1+2*rand(1,m);

    Htr = tanh(Xphase_tr*W + b);

    [beta,beta0] = ols_fit(Htr,y_tr,lambda);

    S_tr = (Htr*beta + beta0)*sd_train + mu_train;
    ytrue_tr = y_tr*sd_train + mu_train;

    err = mean((S_tr - ytrue_tr).^2,'omitnan');

    if isfinite(err) && err < best_err
        best_err = err;
        bestW = W; bestb = b;
        bestbeta = beta; bestbeta0 = beta0;
    end
end

Hte = tanh(Xphase_te*bestW + bestb);
S_te2 = (Hte*bestbeta + bestbeta0)*sd_train + mu_train;

Model.phase.W = bestW;
Model.phase.b = bestb;
Model.phase.beta = bestbeta;
Model.phase.beta0 = bestbeta0;

%% ===================== ELM PROJECTION =====================
best_err = Inf;

for k=1:K
    W = -1+2*rand(size([Xlags_tr Xphase_tr],2),m);
    b = -1+2*rand(1,m);

    Hfull = tanh([Xlags_tr Xphase_tr]*W + b);

    [beta,beta0] = ols_fit(Hfull,y_tr,lambda);

    Hproj = tanh([zeros(size(Xlags_tr)) Xphase_tr]*W + b);

    S_tr = (Hproj*beta + beta0)*sd_train + mu_train;
    ytrue_tr = y_tr*sd_train + mu_train;

    err = mean((S_tr - ytrue_tr).^2,'omitnan');

    if isfinite(err) && err < best_err
        best_err = err;
        bestW = W; bestb = b;
        bestbeta = beta; bestbeta0 = beta0;
    end
end

Hproj_te = tanh([zeros(size(Xlags_te)) Xphase_te]*bestW + bestb);
S_te = (Hproj_te*bestbeta + bestbeta0)*sd_train + mu_train;

Model.proj.W = bestW;
Model.proj.b = bestb;
Model.proj.beta = bestbeta;
Model.proj.beta0 = bestbeta0;

Model.mu = mu_train;
Model.sd = sd_train;

%% ===================== ALPHA =====================
ytrue_te = y_te*sd_train + mu_train;

A = (1:200)'/100;

% Sécurisation alpha projection
E = abs(ytrue_te(:)' - S_te(:)'.*A);
if all(isfinite(E(:)))
    [~,id] = min(mean(E,2,'omitnan'));
    if ~isempty(id) && isfinite(id)
        alpha_proj = A(id);
    else
        alpha_proj = 1;
    end
else
    alpha_proj = 1;
end

% Sécurisation alpha phase
E2 = abs(ytrue_te(:)' - S_te2(:)'.*A);
if all(isfinite(E2(:)))
    [~,id2] = min(mean(E2,2,'omitnan'));
    if ~isempty(id2) && isfinite(id2)
        alpha_phase = A(id2);
    else
        alpha_phase = 1;
    end
else
    alpha_phase = 1;
end

S_te  = alpha_proj  * S_te;
S_te2 = alpha_phase * S_te2;

Model.alpha_proj  = alpha_proj;
Model.alpha_phase = alpha_phase;

best_alpha1 = alpha_proj;

%% ===================== RESIDUALS =====================
R_te  = ytrue_te - S_te;
R_te2 = ytrue_te - S_te2;

%% ===================== CLEAN =====================
R_te(~isfinite(R_te))   = 0;
R_te2(~isfinite(R_te2)) = 0;

%% ===================== PACF =====================
maxLag = 48;
PACFsum = @(p) nansum(abs(p(2:end)));

try
    S_raw = PACFsum(parcorr(ytrue_te,'NumLags',maxLag));
    S_r1  = PACFsum(parcorr(R_te,'NumLags',maxLag));
    S_r2  = PACFsum(parcorr(R_te2,'NumLags',maxLag));
catch
    S_raw = NaN; S_r1 = NaN; S_r2 = NaN;
end

end