function [S,R] = crossed_fourier_deseason(x,u,v,K,L,lambda)
% CROSSED_FOURIER_DESEASON
% Crossed diurnal–annual Fourier deseasonalisation with ridge regularisation
%
% x : signal (Nx1)
% u : diurnal phase   in [0,1)
% v : annual phase    in [0,1)
% K : nb diurnal harmonics
% L : nb annual harmonics
% lambda : ridge parameter

x = x(:);
u = u(:);
v = v(:);

Phi = [];

for k = 1:K
    for l = 0:L
        Phi = [Phi, ...
            cos(2*pi*k*u).*cos(2*pi*l*v), ...
            cos(2*pi*k*u).*sin(2*pi*l*v), ...
            sin(2*pi*k*u).*cos(2*pi*l*v), ...
            sin(2*pi*k*u).*sin(2*pi*l*v)];
    end
end

theta = (Phi'*Phi + lambda*eye(size(Phi,2))) \ (Phi'*x);

S = Phi * theta;
R = x - S;
end