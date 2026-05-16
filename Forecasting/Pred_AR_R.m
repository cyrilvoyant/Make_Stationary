function [Rhat,mes] = Pred_AR_R( ...
    X,y,Beta_AR,mu_R,sigma_R)

% =========================================================
% AR prediction on normalized residuals
%
% INPUTS:
%   X         : input matrix (normalized residuals)
%   y         : target vector (normalized residuals)
%   Beta_AR   : AR coefficients
%   mu_R      : residual mean (training period)
%   sigma_R   : residual std  (training period)
%
% OUTPUTS:
%   Rhat : predicted residuals (original scale)
%   mes  : observed residuals  (original scale)
%
% IMPORTANT:
%   - Same normalization as training
%   - Bias term included
%   - Output restored to physical scale
% =========================================================

%% ================= DIMENSIONS ============================
n = size(X,1);

%% ================= ADD BIAS ==============================
I = [X ones(n,1)];

%% ================= PREDICTION IN Z-SPACE =================
Rhat_z = I * Beta_AR;

mes_z = y(:);

%% ================= SAFETY ================================
if sigma_R < 1e-10
    sigma_R = 1;
end

%% ================= INVERSE NORMALIZATION =================
Rhat = Rhat_z .* sigma_R + mu_R;

mes = mes_z .* sigma_R + mu_R;

%% ================= COLUMN VECTORS ========================
Rhat = Rhat(:);
mes  = mes(:);

end