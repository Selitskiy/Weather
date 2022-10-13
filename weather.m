%% Clear everything 
clearvars -global;
clear all; close all; clc;

% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%% Load data
dataFile = 'measurements_July2022_cleared.xlsx';

dataDir = '~/data/Weather_data';
dataFullName = strcat(dataDir,'/',dataFile);

Mt = readmatrix(dataFullName);
M = Mt(:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]);
[l_whole_ex, ~] = size(M);


% input dimesion (parms x days)
x_in = 10;
t_in = 500;
% output dimensions (parms x days)
y_out = 3;
t_out = 500;


% Leave space for last full label
l_whole = l_whole_ex - t_out;

% Break the whole dataset in training sessions,
% Set training session length (with m_in datapoints of length m_in), 
l_sess = 3*t_in + t_out;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);


ini_rate = 0.01; 
max_epoch = 1000;

regNets = cell([n_sess, 1]);


%% Train or pre-load regNets
for i = 1:n_sess

    norm_fl = 0;
    regNet = AnnNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);


    modelName = regNet.name;

    [regNet, X, Y, B, k_ob] = regNet.TrainTensors(M, l_sess, n_sess, norm_fl);

    regNet = regNet.Train(i, X, Y);

    regNets{i} = regNet;

end