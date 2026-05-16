function [S,R] = rolling_median_deseason(x,window_days)
    x = x(:); N = numel(x);
    S = nan(N,1); H = 24;
    for h = 0:H-1
        idx = find(mod((1:N)-1,H)==h);
        days = ceil(idx/H);
        for i = 1:numel(idx)
            mask = abs(days - days(i)) <= window_days;
            S(idx(i)) = median(x(idx(mask)),'omitnan');
        end
    end
    R = x - S;
end