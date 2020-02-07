close all, clear all
addpath('mocaptoolbox');
addpath('mocaptoolbox/private');
addpath('MoCapToolboxExtension'); %some extensions to the MoCap Toolbox


[test_data, missing_mask, list_patch] = data();
A1 = test_data .* missing_mask;
[A1_norm, A1_mean, list_A, list_A0] = normalization(list_patch, A1);


%weight_sample = ones(1,size(list_A0, 2)) * (1/size(list_A0, 2));
%interpolation_function = create_interpolation_F(list_A0, list_A, weight_sample);               
%[final_result] = interpolation_F(A1, A1_norm, A1_mean, interpolation_function);
%MAE = sum(sum(abs(test_data - final_result) .* (1-missing_mask))) / sum(sum(1-missing_mask))

number_loop = 100;
list_function = {}
list_beta = [];
threshold = 0.7;
power_coefficient = 3;
number_sample = size(list_patch, 2);
limit_error = 2;
list_mean_error = [];
partly_accumulate = [];


index_maxError = -1;
index_maxPosition = -1;
last_weight = 9999;

weight_sample = ones(1,size(list_A0, 2)) * (1/size(list_A0, 2));

for loop_i=1:number_loop
    loop_i
	current_interpolation_function = create_interpolation_F(list_A0, list_A, weight_sample);
    list_function{loop_i} = current_interpolation_function;
	accumulate_error_weight = 0;

    error_sample = mae_training_samples(list_A, list_A0, missing_mask, current_interpolation_function);
	list_mean_error = [list_mean_error, mean(error_sample)];
	index_maxError = 1;
    for x=2:number_sample
		if error_sample(x) > error_sample(index_maxError)
			index_maxError = x;
        end
		if (error_sample(index_maxError) > limit_error) & (index_maxPosition == -1)
			index_maxPosition = loop_i;
        end
    end
    
    if (loop_i - index_maxPosition >= 10) & (index_maxPosition ~= -1)
        weight_sample(index_maxError) = 0;
        index_maxPosition = -1;
    end
    
    for x=1:number_sample
        if error_sample(x) > threshold
            accumulate_error_weight = accumulate_error_weight + weight_sample(x);
        end
    end
    current_beta = accumulate_error_weight ^ power_coefficient;
    list_beta = [list_beta, current_beta];
    
    new_distribution = [];
    accumulate_Z = 0;
    for x=1:number_sample
        if error_sample(x) <= threshold
            new_weight_sample = weight_sample(x) * current_beta;
        else
            new_weight_sample = weight_sample(x);
        end
        new_distribution = [new_distribution, new_weight_sample];
        accumulate_Z = accumulate_Z + new_weight_sample;
    end
    
    for x=1:number_sample
        new_distribution(x) = new_distribution(x) / accumulate_Z;
    end
    [mean_error] = validate_combination_interpolate(list_A, list_A0, missing_mask, list_function, list_beta)
    list_mean_error = [list_mean_error, mean_error];
    if (loop_i > 1) & (mean_error > list_mean_error(loop_i - 1) + 0.0001)
        break
    end
end







