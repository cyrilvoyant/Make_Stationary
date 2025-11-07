function beta = ols_fit(X, Y)
    % y = y(:);
    % [U,S,V] = svd(X,'econ');         % X = U S V'
    % s = diag(S);
    % % Pseudo-inverse OLS: beta = V * S^{-1} * U' * y
    % beta = V * ((s.\(U' * y)));
    % % Équivalent court (QR): beta = X \ y;
lambda = 10^(-6);
    % if nargin < 3, error('lambda manquant'); end
    % if ~isscalar(lambda) || lambda < 0, error('lambda doit être un scalaire >=0'); end
    % if isvector(Y), Y = Y(:); end

    % Centrage pour ne pas pénaliser l’intercept
    muX = mean(X,1);
    muY = mean(Y,1);
    Xc  = X - muX;          % N×p
    Yc  = Y - muY;          % N×k

    [N,p] = size(Xc);

    % Choix numérique : Cholesky si N>=p, sinon SVD (p>>N)
    if N >= p
        % Résout (X'X + λI) beta = X'Y
        A = (Xc.'*Xc) + lambda*eye(p);
        % Cholesky robuste
        R = chol(A + 1e-12*eye(p));     % jitter minime
        beta = R \ (R.' \ (Xc.'*Yc));   % p×k
    else
        % Forme SVD stable : beta = V * diag(s./(s.^2+λ)) * U' * Yc
        [U,S,V] = svd(Xc,'econ');
        s = diag(S);
        D = s ./ (s.^2 + lambda);       % p'×1
        beta = V * (bsxfun(@times, D, (U.'*Yc)));  % p×k
    end

    % Intercept (non pénalisé) : beta0 = E[Y] - E[X]*beta
    beta0 = muY - muX*beta;

end