function [S,R] = rolling_median_deseason(x,window_days,samples_per_day)

x = x(:);
N = numel(x);

S = nan(N,1);

for h = 0:samples_per_day-1

    idx = find(mod((1:N)-1,samples_per_day)==h);

    days = ceil(idx/samples_per_day);

    for i = 1:numel(idx)

        mask = (days <= days(i)) & ...
               (days >= days(i)-window_days);

        S(idx(i)) = median(x(idx(mask)),'omitnan');

    end
end

R = x - S;