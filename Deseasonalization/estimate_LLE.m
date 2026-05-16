function lambda = estimate_LLE(x, tau, m, maxIter)
% ESTIMATE_LLE  Largest Lyapunov Exponent (Rosenstein method)

x = x(:);
x = x(isfinite(x));

N = length(x);
if N < (m+1)*tau + maxIter
    lambda = NaN;
    return
end

M = N - (m-1)*tau;
X = zeros(M,m);
for i = 1:m
    X(:,i) = x((1:M) + (i-1)*tau);
end

D = squareform(pdist(X));
D(D==0) = Inf;

theiler = tau;
for i = 1:M
    D(i, max(1,i-theiler):min(M,i+theiler)) = Inf;
end

[~, nn] = min(D,[],2);

div = zeros(maxIter,1);
count = zeros(maxIter,1);

for k = 1:maxIter
    idx = (1:M-k)';
    valid = (nn(idx)+k <= M);
    idx = idx(valid);
    if isempty(idx), break; end

    dist = vecnorm(X(idx+k,:) - X(nn(idx)+k,:), 2, 2);
    good = dist > 0 & isfinite(dist);

    div(k) = mean(log(dist(good)));
    count(k) = sum(good);
end

valid = count > 5;
t = (1:maxIter)';
t = t(valid);
div = div(valid);

if numel(t) < 5
    lambda = NaN;
    return
end

p = polyfit(t, div, 1);
lambda = p(1);
end
