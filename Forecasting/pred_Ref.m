function [mes,pred_AR,pred_SP,pred_P,pred_CS,pred_CLIPER,pred_ES,pred_ARTU,pred_COMB] = ...
         pred_Ref(Glo_Aj,G_clearsky_h,Horizon,TRAIN_LEN)

Glo_Aj = Glo_Aj(:);
G_clearsky_h = G_clearsky_h(:);

N = length(Glo_Aj);

%% ================= CSI ======================
CSI = Glo_Aj - G_clearsky_h;
average_CSI = mean(CSI,'omitnan');
CSI_centre = CSI - average_CSI;

CSI = fillmissing(CSI,'previous');
CSI_centre = fillmissing(CSI_centre,'previous');

%% ================= AR ======================
p_order = 6;

X = zeros(N,p_order);
y = zeros(N,1);

for j = 1:p_order
    for i = p_order : N-Horizon
        X(i,j) = CSI_centre(i-j+1);
        y(i)   = CSI_centre(i+Horizon);
    end
end

%% ================= TRAIN ====================
Xtr = X(1:TRAIN_LEN,:);
ytr = y(1:TRAIN_LEN);

w = (Xtr' * Xtr) \ (Xtr' * ytr);

%% ================= TEST =====================
idx_test = (TRAIN_LEN+1):(N-Horizon);
L = length(idx_test);

mes          = zeros(L,1);
pred_AR      = zeros(L,1);
pred_P       = zeros(L,1);
pred_CS      = zeros(L,1);
pred_SP      = zeros(L,1);
pred_ES      = zeros(L,1);
pred_ARTU    = zeros(L,1);
pred_CLIPER  = zeros(L,1);

for k = 1:L
    p = idx_test(k);

    % ===== TRUE VALUE =====
    mes(k) = Glo_Aj(p+Horizon);

    % ===== AR =====
    pred_AR(k) = (X(p,:) * w) + average_CSI + G_clearsky_h(p+Horizon);

    % ===== PERSISTENCE =====
    pred_P(k) = Glo_Aj(p);

    % ===== CLEAR SKY =====
    pred_CS(k) = G_clearsky_h(p+Horizon);

    % ===== SMART PERSISTENCE (FIXED) =====
    pred_SP(k) = Glo_Aj(p) ...
                 - G_clearsky_h(p) ...
                 + G_clearsky_h(p+Horizon);
end

%% ================= CLIPER / ES / ARTU ===================

acf = autocorr(Glo_Aj - G_clearsky_h,'NumLags',30);

rho  = acf(Horizon+1);
rho2 = acf(2*Horizon+1);

[K,alpha] = K_alpha(rho,rho2,2);
S = alpha + K;
P = alpha * K;

ave = mean(Glo_Aj - G_clearsky_h,'omitnan');

pred_CLIPER = rho*(pred_SP - pred_CS) + (1-rho)*ave + pred_CS;

for k = 1:L
    p = idx_test(k);

    if p > Horizon

        % ===== ES (FIXED) =====
        pred_ES(k) = rho*(Glo_Aj(p) - G_clearsky_h(p)) ...
                     + (1-rho)*ave + G_clearsky_h(p+Horizon);

        % ===== ARTU (FIXED) =====
        pred_ARTU(k) = S*(Glo_Aj(p) - G_clearsky_h(p)) ...
                       - P*(Glo_Aj(p-Horizon) - G_clearsky_h(p-Horizon)) ...
                       + (1+P-S)*ave + G_clearsky_h(p+Horizon);

    else
        pred_ES(k)   = pred_CS(k);
        pred_ARTU(k) = pred_CS(k);
    end
end

%% ================= COMB ====================
pred_COMB = (pred_SP + pred_CLIPER + pred_ES + pred_ARTU)/4;

%% ================= CLIPPING =================
pred_AR(pred_AR<0)=0;
pred_P(pred_P<0)=0;
pred_CS(pred_CS<0)=0;
pred_SP(pred_SP<0)=0;
pred_CLIPER(pred_CLIPER<0)=0;
pred_ES(pred_ES<0)=0;
pred_ARTU(pred_ARTU<0)=0;
pred_COMB(pred_COMB<0)=0;

end