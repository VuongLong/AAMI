function [mean_error] = validate_combination_interpolate(list_A, list_A0, missing_mask, list_function, list_beta)
sum_beta_error_sample = zeros(1,size(list_A, 2));
sum_beta = 0;
for loop_i=1:size(list_function, 2)
    interpolation_function = list_function{loop_i};
    error_sample = mae_training_samples(list_A, list_A0, missing_mask, interpolation_function);
    function_beta = log(1/list_beta(loop_i));
    beta_error_sample = function_beta * error_sample;
    
    sum_beta = sum_beta + function_beta;
    sum_beta_error_sample = sum_beta_error_sample + beta_error_sample;
end
sum_beta_error_sample = sum_beta_error_sample / sum_beta;
mean_error = mean(sum_beta_error_sample);
end