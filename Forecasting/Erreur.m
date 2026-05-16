% function [nRMSE,nMAE,R2,r2,nMBE] = Erreur(mesure,prevision)
% 
%     % --------------------------------------------------
%     % Force inputs d'etre des vecteurs 
%     % --------------------------------------------------
%     mesure    = mesure(:);
%     prevision = prevision(:);
%     %CS        = CS(:);
%     % --------------------------------------------------
%     % Filtre NaN et valeurs nulles
%     % --------------------------------------------------
%     idxVal = ~isnan(mesure) & ~isnan(prevision); %& mesure>0; %CS >= 10;
%     
%     mesure    = mesure(idxVal);
%     prevision = prevision(idxVal);
%     
%     % --------------------------------------------------
%     % Erreurs normalisées (par la moyenne observée)
%     % --------------------------------------------------
%     nMAE  = mean(abs(mesure - prevision)) / max(mesure);
%     nRMSE = sqrt(mean((mesure - prevision).^2)) / max(mesure);
%     nMBE  = mean(prevision - mesure) / max(mesure);
% 
%     % --------------------------------------------------
%     % R² (coefficient de détermination) + Pearson r²
%     % --------------------------------------------------
%     if numel(mesure) > 1
%         SSE = sum((mesure - prevision).^2);
%         SST = sum((mesure - mean(mesure)).^2);
%         R2  = 1 - SSE / SST;
% 
%         C   = corrcoef(mesure, prevision);
%         r2  = C(1,2)^2;
%     else
%         R2 = NaN;
%         r2 = NaN;
%     end
% 
% end

function [nRMSE,nMAE,R2,r2,nMBE] = Erreur(obs,pred)

% =============================
% Force column vectors
% =============================
obs  = obs(:);
pred = pred(:);

% =============================
% Remove invalid values
% =============================
valid = isfinite(obs) & isfinite(pred);

obs  = obs(valid);
pred = pred(valid);

if isempty(obs)
    nRMSE = NaN; nMAE = NaN; R2 = NaN; r2 = NaN; nMBE = NaN;
    return;
end

% =============================
% Normalization factor
% =============================
mean_obs = mean(obs);

if abs(mean_obs)< 1e-8
    nRMSE = NaN; nMAE = NaN; R2 = NaN; r2 = NaN; nMBE = NaN;
    return;
end

% =============================
% Errors
% =============================
e = pred - obs;

nMAE  = mean(abs(e)) / mean_obs;
nRMSE = sqrt(mean(e.^2)) / mean_obs;
nMBE  = mean(e) / mean_obs;

% =============================
% R² (deterministic)
% =============================
SSE = sum(e.^2);
SST = sum((obs - mean_obs).^2);

if SST > 1e-10
    R2 = 1 - SSE / SST;
else
    R2 = NaN;
end

% =============================
% Correlation r²
% =============================
if std(obs) > 1e-10 && std(pred) > 1e-10
    C  = corrcoef(obs,pred);
    r2 = C(1,2)^2;
else
    r2 = NaN;
end

end
