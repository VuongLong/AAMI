function [original] = read_c3d_data(dataf_ile, data_length)
%Load data (requires the MoCap toolbox)
original = mcread(dataf_ile);


if data_length ~= -1
original.data = original.data(data_length(1):data_length(2), :);
original.nFrames = data_length(2)-data_length(1)+1;
original.other.residualerror = original.other.residualerror(data_length(1):data_length(2), :);
end
%Remove invalid markers in hdm files
markernames = original.markerName;
ids = find(~ismember(markernames,{'*0','*1','*2'}));
original = mcgetmarker(original,ids);

%interpolate some unexpected really small gaps in some hdm files
%if strcmp(lower(original.filename(1:3)),'hdm')
    numberofnans = sum(sum(isnan(original.data)));
    fprintf('number of nans in original file: %d\n',numberofnans);
    original.data = naninterp(original.data,'pchip');
    disp('salut');
%end

end