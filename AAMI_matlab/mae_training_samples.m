function [MAEs] = mae_training_samples(list_A, list_A0, missing_mask, interpolation_function)
list_V0 = interpolation_function{1};
list_F = interpolation_function{2};
list_V = interpolation_function{3}; 
list_alpha = interpolation_function{4};

MAEs = [];
for sample_i=1:size(list_A, 2)
    current_missing_sample = list_A0{sample_i};
	current_original_sample = list_A{sample_i};
    tmp_result = zeros(size(current_missing_sample));
    for i=1:size(list_V, 2)
        diag_matrix = diag(ones(size(list_A0, 1))*list_alpha(i));
        tmp = current_missing_sample*list_V0{i}*list_F{i}*list_V{i}';

        tmp_result = tmp_result + diag_matrix * tmp;
    end   
    result = current_missing_sample;
    result(current_missing_sample == 0) = tmp_result(current_missing_sample == 0);
    MAE = sum(sum(abs(result - current_original_sample) .* (1-missing_mask))) / sum(sum(1-missing_mask));
    MAEs = [MAEs, MAE];
end
end