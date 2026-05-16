function [X,y] = InputOutput_R(Rz, Horizon, Nin)
% =========================================================
% Construct EL/AR input-output matrices from a time series
%
% INPUTS:
%   Rz       : normalized residual signal (column vector)
%   Horizon  : forecasting horizon
%   Nin      : number of input lags
%
% OUTPUTS:
%   X : input matrix
%       [R(t), R(t-1), ..., R(t-Nin+1)]
%
%   y : target vector
%       R(t + Horizon)
%
% IMPORTANT:
%   - Temporal alignment is preserved
%   - Forecast horizon is handled internally
%   - First valid sample starts at t = Nin
%   - Last valid sample ends at t = N - Horizon
%
% EXAMPLE:
%   Nin = 3
%   Horizon = 2
%
%   X = [R(t)   R(t-1) R(t-2)]
%   y =  R(t+2)
%
% =========================================================

%% ================= FORCE COLUMN VECTOR ==================
Rz = Rz(:);

%% ================= SIGNAL LENGTH ========================
N = length(Rz);

%% ================= NUMBER OF VALID SAMPLES ==============
Nvalid = N - Horizon - Nin + 1;

if Nvalid <= 0
    error('InputOutput_R:NotEnoughData', ...
        'Time series too short for selected Horizon and Nin.');
end

%% ================= PREALLOCATE ==========================
X = zeros(Nvalid, Nin);
y = zeros(Nvalid, 1);

%% ================= BUILD INPUT/TARGET ===================
k = 1;

for t = Nin : (N - Horizon)

    % ---- Input lags
    X(k,:) = Rz(t:-1:t-Nin+1);

    % ---- Future target
    y(k) = Rz(t + Horizon);

    k = k + 1;

end

end