function [Beta,W,nRMSE_best] = ...
    Train_EL(X,y,Nbre_Hidden,Lambda,T,Nbre_Run)

% =========================================================
% Extreme Learning Machine training
% with mixed activations + ridge regression
%
% INPUTS:
%   X            : input matrix
%   y            : target vector
%   Nbre_Hidden  : number of hidden neurons
%   Lambda       : ridge parameter
%   T            : proportion of sigmoid neurons
%   Nbre_Run     : number of random initializations
%
% OUTPUTS:
%   Beta         : output weights
%   W            : hidden weights
%   nRMSE_best   : best normalized RMSE
%
% IMPORTANT:
%   - Mixed activations:
%       sigmoid + gaussian
%   - Ridge regression stabilization
%   - Multiple random initializations
% =========================================================

%% ================= DIMENSIONS ============================
n = size(X,1);
p = size(X,2);

%% ================= INPUT MATRIX ==========================
I = [X ones(n,1)];

%% ================= INITIALIZATION ========================
best_err   = Inf;
nRMSE_best = Inf;

Beta = [];
W    = [];

%% =========================================================
% MULTI-RUN TRAINING
%% =========================================================
for j = 1:Nbre_Run

    %% ================= RANDOM WEIGHTS ====================
    Wj = 2*rand(Nbre_Hidden,p+1) - 1;

    %% ================= HIDDEN ACTIVATIONS ================
    A = Wj * I';

    % ---- Numerical stability
    A = max(min(A,50),-50);

    %% ---- Activation functions
    H1 = 1 ./ (1 + exp(-A));   % sigmoid
    H2 = exp(-(A.^2)/2);       % gaussian

    %% =====================================================
    % MIXED ACTIVATION LAYER
    %% =====================================================
    split = floor(T * Nbre_Hidden);

    Hbis = zeros(Nbre_Hidden,n);

    if split > 0
        Hbis(1:split,:) = H1(1:split,:);
    end

    if split < Nbre_Hidden
        Hbis(split+1:end,:) = H2(split+1:end,:);
    end

    %% ================= FINAL H MATRIX ====================
    H = [Hbis; ones(1,n)];

    %% =====================================================
    % RIDGE REGRESSION (More numerically stable)
    %% =====================================================
    try

        Areg = H*H' + Lambda*eye(size(H,1));

        Beta_j = Areg \ (H*y);

    catch
        continue
    end

    %% ================= TRAIN PREDICTION ==================
    pred = H' * Beta_j;

    %% ================= CLEAN =============================
    valid = isfinite(pred) & isfinite(y);

    if sum(valid) < 10
        continue
    end

    pred = pred(valid);
    mes  = y(valid);

    %% ================= TRAIN ERROR =======================
    err = mean((pred - mes).^2);

    %% ================= NORMALIZED RMSE ===================
    denom = std(mes);

    if denom < 1e-10
        denom = 1;
    end

    nRMSE_j = sqrt(err) / denom;

    %% ================= BEST MODEL ========================
    if isfinite(err) && err < best_err

        best_err   = err;
        nRMSE_best = nRMSE_j;

        Beta = Beta_j;
        W    = Wj;
    end

end

%% =========================================================
% FALLBACK SECURITY
%% =========================================================
if isempty(Beta)

    warning('Train_EL: No valid model found -> fallback')

    Beta = zeros(Nbre_Hidden+1,1);

    W = 2*rand(Nbre_Hidden,p+1)-1;

    nRMSE_best = NaN;

end

end