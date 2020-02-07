function [A1_norm, A1_mean, list_A, list_A0] = normalization(list_patch, test_data)

combine_matrix = [test_data];
for i=1:size(list_patch, 2)
    combine_matrix = [combine_matrix; list_patch{i}];
end


A1 = test_data;
AA1 = combine_matrix;
AA1_MeanVec = sum(AA1,1) ./ sum(AA1~=0,1);

A1_MeanMat = repmat(AA1_MeanVec, size(A1, 1), 1);
A1_new = A1 - A1_MeanMat;
A1_new(A1==0) = 0;

list_A = {};
list_A0 = {};
for i=1:size(list_patch, 2)
    list_A{i} = list_patch{i} - A1_MeanMat;
    tmp = list_A{i};
    tmp(A1==0) = 0;
    list_A0{i} = tmp;
end

A1_norm = A1_new;
A1_mean = A1_MeanMat;
end