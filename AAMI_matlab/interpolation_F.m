function [final_result] = interpolation_F(A1, A1_norm, A1_mean, interpolation_function)
list_V0 = interpolation_function{1};
list_F = interpolation_function{2};
list_V = interpolation_function{3}; 
list_alpha = interpolation_function{4};
tmp_result = zeros(size(A1));
for i=1:size(list_V, 2)
    diag_matrix = diag(ones(size(list_A0, 1))*list_alpha(i));
    tmp = A1_norm*list_V0{i}*list_F{i}*list_V{i}';

    tmp_result = tmp_result + diag_matrix * tmp;
end   
result = A1_norm;
result(A1_norm == 0) = tmp_result(A1_norm == 0);
reconstructed = result + A1_mean;
final_result = A1;
final_result(A1 == 0) = reconstructed(A1 == 0);
end