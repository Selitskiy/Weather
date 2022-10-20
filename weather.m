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
t_in = 144; %500;
% output dimensions (parms x days)
y_out = 3;
t_out = 144; %500;


% Leave space for last full label
l_whole = l_whole_ex - t_out;

% Break the whole dataset in training sessions,
% Set training session length (with m_in datapoints of length m_in), 
l_sess = 3*t_in + t_out;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);


ini_rate = 0.01; 
max_epoch = 100;
norm_fl = 0;

regNets = cell([n_sess, 1]);


%% Train or pre-load regNets
for i = 1:n_sess-1

    %regNet = LinRegNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
    %regNet = AnnNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
    %regNet = ReluNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
    %regNet = KgNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);

    %regNet = SigNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
    regNet = TanhNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);


    modelName = regNet.name;

    [regNet, X, Y, Bi, Bo, k_ob] = regNet.TrainTensors(M, l_sess, n_sess, norm_fl);

    regNet = regNet.Train(i, X, Y);

    regNets{i} = regNet;

end


%% Test parameters 
% the test input period - same as training period, to cover whole data
l_test = l_sess;

% Test from particular training session
sess_off = 0;
% additional offset after training sessions (usually for the future forecast)
offset = 0;

% Left display margin
l_marg = 1;

%% For whole-through test, comment out secion above
% Number of training sessions with following full-size test sessions 
t_sess = floor((l_whole - l_sess) / l_test);

[X2, Y2, Yh2, Bti, Bto, k_tob] = regNets{1}.TestTensors(M, l_sess, l_test, t_sess, sess_off, offset, norm_fl);

%% test
[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, t_sess, sess_off, k_tob);


%% re-scale in observation bounds
if(norm_fl)
    [Y2, Yh2] = regNets{1}.ReScale(Y2, Yh2, Bto, t_sess, sess_off, k_tob);
end


%% Calculate errors
[S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Y2, Yh2); 

fprintf('%s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MaxAPErr %f+-%f\n', modelName, dataFile, norm_fl, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


[S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Y2, Yh2); 

fprintf('%s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MaxRSErr %f+-%f\n', modelName, dataFile, norm_fl, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));
