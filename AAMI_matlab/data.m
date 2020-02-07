function [test_data, missing_mask, list_patch] = data()
test_data_file = '../fastsong/fastsong7.txt';
selected_data = [450, 550-1];
test_data = read_matrix_data(test_data_file, selected_data);

missing_mask_file = '../fastsong/fastsong7/test_data_Aniage_num_gap/12/16_test.txt';
missing_mask = read_matrix_data(missing_mask_file, selected_data);

REFERENCE_DIR = [
"../fastsong/fastsong7.txt";
"../fastsong/fastsong7.txt"];
SELECTED_DATA = [
[50, 450-1];
[550, 750-1]];
PATCH_LENGTH = 100;

list_patch = {};
count = 1;

% we check data and these joints is dupplicated
remove_similar_joints = [18, 14, 9, 5];

for i=1:size(REFERENCE_DIR)
    reference_data = read_matrix_data(REFERENCE_DIR(i), SELECTED_DATA(i,:));

    for x=1:size(remove_similar_joints, 2)
        reference_data(:,remove_similar_joints(x)*3+3) = [];
        reference_data(:,remove_similar_joints(x)*3+2) = [];
        reference_data(:,remove_similar_joints(x)*3+1) = [];
    end
    number_patch = (SELECTED_DATA(i,2)-SELECTED_DATA(i,1)+1)/PATCH_LENGTH;
    for j=1:(number_patch)
        list_patch{count} = reference_data(PATCH_LENGTH*(i-1)+1:PATCH_LENGTH*i, :);
        count = count + 1;
    end
end

for x=1:size(remove_similar_joints, 2)
    test_data(:,remove_similar_joints(x)*3+3) = [];
    test_data(:,remove_similar_joints(x)*3+2) = [];
    test_data(:,remove_similar_joints(x)*3+1) = [];
end
end