function [obs, pred_P, pred_CLIM] = pred_Ref_PV_WT_PT_WS_T2M(X, Horizon, TRAIN_LEN, is_nonnegative)
% =========================================================
% Reference models for PV, WT, PT, WS and T2M variables
%
% VALID FOR:
%   PV, WT, PT, WS, T2M
%
% MODELS:
%   - Persistence (P)
%   - Climatology (CLIM - hourly)
%
% =========================================================

X = X(:);
N = length(X);

%% ================= TEST INDEX =================
idx_test = (TRAIN_LEN+1):(N-Horizon);
L = length(idx_test);

obs       = zeros(L,1);
pred_P    = nan(L,1);
pred_CLIM = nan(L,1);

%% ================= CLIMATOLOGY =================
% Hourly climatology

hour_train = mod((1:TRAIN_LEN)-1,24) + 1;

CLIM = nan(24,1);

for h = 1:24
    idx_h = (hour_train == h);
    CLIM(h) = mean(X(idx_h),'omitnan');
end

%% ================= LOOP =================
for k = 1:L

    p = idx_test(k);

    % ---- Observation (target)
    obs(k) = X(p + Horizon);

    % ---- Persistence
    pred_P(k) = X(p);

    % ---- Climatology (aligned with forecast time)
    h = mod(p + Horizon - 1,24) + 1;
    pred_CLIM(k) = CLIM(h);

end

%% ================= PHYSICAL CONSTRAINTS =================
if is_nonnegative
    pred_P(pred_P < 0)       = 0;
    pred_CLIM(pred_CLIM < 0) = 0;
end

%% ================= CLEAN =================
valid = isfinite(obs) & isfinite(pred_P) & isfinite(pred_CLIM);

obs       = obs(valid);
pred_P    = pred_P(valid);
pred_CLIM = pred_CLIM(valid);

end