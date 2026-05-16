function [K,alpha] = K_alpha(corrh,corr2h,R)

%K_alpha input corrh the autocorrelation corresponding to lag h (-1<..<1)
%              corr2h the autocorrelation corresponding to lag 2h (-1<..<1)
%              Q the quality of the measurements, Q=0 perfect case, Q=1 for estimated error of 1% (R=1%), Q=2 for 5% (R=5%), Q=3 for 10% (R=10%)  
%   	       Q = 2 seems be a the best solution if user doesn't know the estimated measurement error
%K_alpha ouput K is the K value of ARTU method
%              alpha is the alpha value of the ARTU method 

if R==0 %0 
load('R0.mat');
end
if R==1 %0 
load('R0.01.mat');
end
if R==2 %0 
load('R0.05.mat');
end
if R==3 %0 
load('R0.1.mat');
end


[Y]=meshgrid(-1:0.05:1);
X=Y';

K = interp2(Y,X,res_K,corr2h,corrh);
alpha = interp2(Y,X,res_a,corr2h,corrh);

end

