function [PC,sqrtEV] = PCA(Data)
[N,~] = size(Data);
Y = Data / sqrt(N-1);
[~,sqrtEV,PC] = svd(Y,'econ');
sqrtEV = diag(sqrtEV);
end