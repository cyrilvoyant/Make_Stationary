function [S_proj_full, S_phase_full] = apply_elm_full(series, Model, LagH, H)

series = series(:);
N = numel(series);

%% ===================== NORMALISATION =====================
mu = Model.mu;
sd = Model.sd;

Z = (series - mu)/sd;
Z(~isfinite(Z)) = 0;

%% ===================== h =====================
h = Model.h;

%% ===================== FEATURES =====================
idx = (LagH+1):(N-h);

% --- LAGS ---
Xlags = zeros(numel(idx), LagH+1);
for i = 0:LagH
    Xlags(:,i+1) = Z(idx - i);
end

% --- PHASES ---
if H == 8760*4
    Nd = 96; Ny = 365*96;
else
    Nd = 24; Ny = 365*24;
end

t = idx + h - 1;

phi_d = mod(t,Nd)/Nd;
phi_y = mod(t,Ny)/Ny;

Xphase = [cos(2*pi*phi_d(:)) sin(2*pi*phi_d(:)) ...
          cos(2*pi*phi_y(:)) sin(2*pi*phi_y(:))];

%% ===================== PHASE =====================
W = Model.phase.W;
b = Model.phase.b;
beta  = Model.phase.beta;
beta0 = Model.phase.beta0;   

Hk = tanh(Xphase*W + b);
S_phase = (Hk*beta + beta0)*sd + mu;  

%% ===================== PROJECTION =====================
W = Model.proj.W;
b = Model.proj.b;
beta  = Model.proj.beta;
beta0 = Model.proj.beta0;  

Hproj = tanh([zeros(size(Xlags)) Xphase]*W + b);
S_proj = (Hproj*beta + beta0)*sd + mu; 

%% ===================== APPLY ALPHA =====================
S_proj  = Model.alpha_proj  * S_proj;
S_phase = Model.alpha_phase * S_phase;

%% ===================== FULL RECONSTRUCTION =====================
S_phase_full = nan(N,1);
S_proj_full  = nan(N,1);

S_phase_full(idx + h) = S_phase;
S_proj_full(idx + h)  = S_proj;

%% ===================== PHYSICAL CONSTRAINTS =====================
isNonNegative = min(series) >= 0;
zeroRatio = mean(series < 0.02*max(series));
isSolarLike = isNonNegative && zeroRatio > 0.25;

if isSolarLike
    
    ytrue = max(0,series);
    
    % borne physique
    S_phase_full = max(0, min(S_phase_full, ytrue));
    S_proj_full  = max(0, min(S_proj_full,  ytrue));
    
    % nuit
    seuil = 0.05*max(ytrue);
    mask_night = ytrue < seuil;
    
    S_phase_full(mask_night) = 0;
    S_proj_full(mask_night)  = 0;
    
end

%% ===================== FINAL CLEAN =====================
S_phase_full(~isfinite(S_phase_full)) = nan;
S_proj_full(~isfinite(S_proj_full))   = nan;

end