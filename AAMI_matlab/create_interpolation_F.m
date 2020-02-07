function [interplate] = create_interpolation_F(list_A0, list_A, weight_sample)

list_V = {};
list_V0 = {};
list_F = {};
max_number_EV = 0;
for i=1:size(list_A0, 2)
    [PC,sqrtEV] = PCA(list_A0{i});
    list_V0{i} = PC';
    tmp_sqrtEV = sqrtEV;
    tmp_sqrtEV(tmp_sqrtEV>=0.01)=1;
    tmp_sqrtEV(tmp_sqrtEV<0.01)=0;
    EV = sum(tmp_sqrtEV(tmp_sqrtEV==1));
    if EV > max_number_EV
        max_number_EV = EV;
    end
    
    [PC,sqrtEV] = PCA(list_A{i});
    tmp_sqrtEV = sqrtEV;
    tmp_sqrtEV(tmp_sqrtEV>=0.01)=1;
    tmp_sqrtEV(tmp_sqrtEV<0.01)=0;
    EV = sum(tmp_sqrtEV(tmp_sqrtEV==1));
    if EV > max_number_EV
        max_number_EV = EV;
    end
    list_V{i} = PC';
end

for i=1:size(list_A0, 2)
    list_V{i} = list_V{i}(:, 1:max_number_EV);
    list_V0{i} = list_V0{i}(:, 1:max_number_EV);
    list_F{i} = list_V0{i}' * list_V{i};
end


list_Q = {};
count = 0;
for iloop=1:size(list_A0, 2)
    for kloop=1:size(list_A0, 2)
        count = count + 1;
        diag_matrix = diag(ones(size(list_A0, 1))*weight_sample(iloop));
        qi = list_V{kloop}*list_F{kloop}';
        qi = qi * list_V0{iloop}';
        qi = qi * list_A0{iloop}';
        qi = qi * diag_matrix;
        qi = qi * list_A{iloop};
        list_Q{count} = qi;
    end
end

tmp_matrix = list_Q{1};
for i=2:size(list_Q, 2)
    tmp_matrix = tmp_matrix + list_Q{i};
end

right_hand = reshape(tmp_matrix,1,size(tmp_matrix, 1)*size(tmp_matrix, 2));          
                
list_P = {};
for jloop=1:size(list_A0, 2)
    list_Pijk = {};
    count_Pijk = 0;
    for iloop=1:size(list_A0, 2)
        for kloop=1:size(list_A0, 2)
            diag_matrix = diag(ones(size(list_A0, 1))*weight_sample(iloop));
            
            Pijk = list_V{kloop} * list_F{kloop}';
            Pijk = Pijk * list_V0{iloop}';
            Pijk = Pijk * list_A0{iloop}';
            Pijk = Pijk * diag_matrix;
            Pijk = Pijk * list_A0{iloop};
            Pijk = Pijk * list_V0{iloop}; 
            Pijk = Pijk * list_F{jloop};
            Pijk = Pijk * list_V{jloop}';

            count_Pijk = count_Pijk + 1;
            list_Pijk{count_Pijk} = Pijk;
        end
    end

    tmp_matrix = list_Pijk{1};
    for i=2:size(list_Pijk, 2)
        tmp_matrix = tmp_matrix + list_Pijk{i};
    end
    tmp_matrix = reshape(tmp_matrix,1,size(tmp_matrix, 1)*size(tmp_matrix, 2));  
    list_P{jloop} = tmp_matrix;
end   
left_hand = [];
for i=1:size(list_P, 2)
    left_hand = [left_hand; list_P{i}];
end

list_alpha = (left_hand*left_hand')/(right_hand*left_hand');
interplate = {list_V0, list_F, list_V, list_alpha};
end
