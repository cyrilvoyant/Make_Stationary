function [Beta] = Train_AR(X,y)

n = size(X,1);

I = [X ones(n,1)];

Beta = I \ y;

end