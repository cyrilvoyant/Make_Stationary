function [S_proj_full, S_phase_full] = apply_elm_full(series, Model, LagH, H, VarName)
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
S_proj = Model.a_proj*S_proj + Model.b_proj;
S_phase = Model.a_phase*S_phase + Model.b_phase;
%% ===================== PHYSICAL CONSTRAINTS =====================

isSolarVariable = any(strcmp(VarName,...
    {'PV','GHI_30min','GHI_1h'}));

if isSolarVariable

    S_phase = max(0,S_phase);
    S_proj  = max(0,S_proj);

end
%% ===================== FULL RECONSTRUCTION =====================
S_phase_full = nan(N,1);
S_proj_full  = nan(N,1);

S_phase_full(idx + h) = S_phase;
S_proj_full(idx + h)  = S_proj;

%% ===================== FINAL CLEAN =====================
S_phase_full(~isfinite(S_phase_full)) = nan;
S_proj_full(~isfinite(S_proj_full))   = nan;

end
