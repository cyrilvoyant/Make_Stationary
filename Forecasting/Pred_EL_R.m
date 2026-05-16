function [Rhat,mes] = Pred_EL_R( ...
    X,y,W,Beta,Nbre_Hidden,mu_R,sigma_R)

% =========================================================
% Prediction function for ELM residual forecasting
%
% INPUTS:
%   X            : input matrix
%   y            : target vector (normalized)
%   W            : hidden weights
%   Beta         : output weights
%   Nbre_Hidden  : number of hidden neurons
%   mu_R         : residual mean (train)
%   sigma_R      : residual std (train)
%
% OUTPUTS:
%   Rhat : predicted residuals (original scale)
%   mes  : observed residuals  (original scale)
%
% IMPORTANT:
%   Must use SAME activations as Train_EL
% =========================================================

%% ================= INPUT MATRIX ==========================
n = size(X);

I = [X ones(n(1),1)];

%% ================= ACTIVATIONS ===========================
A = W * I';

% ---- Numerical stability
A = max(min(A,50),-50);

%% ---- SAME ACTIVATIONS AS TRAIN_EL
H1 = 1 ./ (1 + exp(-A));     % sigmoid
H2 = exp(-(A.^2)/2);         % gaussian

%% =========================================================
% MIXED ACTIVATION
% MUST MATCH Train_EL
%% =========================================================
Tmix = 0.6;

split = floor(Tmix * Nbre_Hidden);

Hbis = zeros(Nbre_Hidden,n(1));

if split > 0
    Hbis(1:split,:) = H1(1:split,:);
end

if split < Nbre_Hidden
    Hbis(split+1:end,:) = H2(split+1:end,:);
end

%% ================= FINAL H ===============================
H = [Hbis; ones(1,n(1))];

%% ================= PREDICTION ============================
Rhat_z = H' * Beta;

mes_z = y(:);

%% ================= INVERSE Z-SCORE =======================
Rhat = Rhat_z .* sigma_R + mu_R;

mes = mes_z .* sigma_R + mu_R;

%% ================= COLUMN VECTORS ========================
Rhat = Rhat(:);
mes  = mes(:);

end