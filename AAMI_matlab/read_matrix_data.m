function [data] = read_matrix_data(data_file, data_length)

original = mcread('format.c3d');
data = readmatrix(data_file);
original.data = data;
original.filename = data_file;
original.nFrames = size(data, 1);
original.nMarkers = size(data, 2) / 3;
original.markerName = original.markerName(1: original.nMarkers);
original.freq = 30;

if data_length ~= -1
original.data = original.data(data_length(1):data_length(2), :);
original.nFrames = data_length(2)-data_length(1)+1;
original.other.residualerror = original.other.residualerror(data_length(1):data_length(2), :);
end
data = data(data_length(1):data_length(2), :);
end

