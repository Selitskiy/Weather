%% Clear everything 
clearvars -global;
clear all; close all; clc;

% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%% Load data
%dataFile = 'measurements_July2022_cleared.xlsx';
dataFile = 'measurements_July-October2022_cleared.xlsx';

dataDir = '~/data/Weather_data';
dataFullName = strcat(dataDir,'/',dataFile);

Mt = readmatrix(dataFullName);
M = Mt(:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]);
[l_whole_ex, ~] = size(M);


% input dimesion (parms x days)
%x_off = 10;
%x_in = 3;

%x_off = 0;
%x_in = 13;

x_off = 0;
x_in = 10;
t_in = 144;

% output dimensions (parms x days)
y_off = 10;
y_out = 3;
t_out = 144;
ts_out = 36; 

% Leave space for last full label
l_whole = l_whole_ex - t_out;
% Proportional to even number of whole outpouts
%l_whole = (floor(l_whole_ex/t_out)-1)*t_out; 


% Break the whole dataset in training sessions,
% Set training session length (space to slide window of size t_in datapoints, 
% plus length of last label t_out, plus size of input for test on next session), 
l_sess = 12*t_in + t_out + t_in;

% Test output period - if same as training period, will cover whole data
l_test = l_sess; %t_out; %l_sess;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);

% Check sessions fit
if (l_whole - n_sess*l_sess) < l_test
    n_sess = n_sess - 1;
end


ini_rate = 0.001; 
max_epoch = 100;
norm_fli = 1;
norm_flo = 1;

regNets = cell([n_sess, 1]);


%% Train or pre-load regNets
for i = 1:n_sess

    %regNet = LinRegNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = AnnNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = ReluNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = KgNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);

    regNet = SigNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = TanhNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = RbfNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = TransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = LstmValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = GruValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = LinRegNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = AnnNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = ReluNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = KgNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);

    %regNet = SigNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = TanhNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = RbfNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = TransNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);

    %regNet = LstmNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = GruNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    modelName = regNet.name;

    [regNet, X, Y, Bi, Bo, Sx, Sy, k_ob] = regNet.TrainTensors(M, l_sess, n_sess, norm_fli, norm_flo);

    regNet = regNet.Train(i, X, Y);

    regNets{i} = regNet;

end


%% Test parameters 
% Test from particular training session
sess_off = 0;
% additional offset after training sessions (usually for the future forecast)
offset = 0;

% Left display margin
l_marg = 1;


% Number of training sessions with following full-size test sessions 
%t_sess = floor((l_whole - l_sess) / l_test);
t_sess = n_sess;

%Just one immediate prediction of length t_out (if 0, how many t_out fit
%into l_test)
k_tob = 0;

[X2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = regNets{1}.TestTensors(M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);

%% test
[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, t_sess, sess_off, k_tob);

%% re-scale in observation bounds
%if(norm_fli)
%    [X, X2] = regNets{1}.ReScaleIn(X, X2, Bi, n_sess, t_sess, sess_off, k_ob, k_tob);
%end

if(norm_flo)
    [Y, Y2, Yhs2] = regNets{1}.ReScaleOut(Y, Y2, Yhs2, Bo, Bto, n_sess, t_sess, sess_off, k_ob, k_tob);
end

%% Calculate errors
[Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Y2, Yh2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MeanMaxAPErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


[Er, S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Y2, Yh2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MeanMaxRSErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));



%%
regNets{1}.Err_graph(M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName);

%%
%regNets{1}.TestIn_graph(M, l_whole_ex, X, Y, X2, Y2, Sx, Sy, Sx2, Sy2, l_whole, n_sess, l_sess, k_ob, k_tob, t_sess, sess_off, offset, l_marg, modelName);


% PCA for training

% No normalization
%X1n = Xt;
% Normalize each image chanell (variable or dimension) across image set (horizontal)
%[X1n,PS] = mapstd(Xt);

C = cov( X1n' );
[V, ~, ~] = eig(C);
Vn1 = flipud(inv(V));    
X1pe = Vn1 * X1n;

% Remove minor components with normalized var < thresh
Expl = var(X1pe, 0, 2)/sum(var(X1pe, 0, 2))*100;
It = Expl >= thresh;
Vn1t = Vn1(It, :);
X1pet = X1pe(It, :);
Xt = X1pet;

% PCA for validation

% No normalization
%X2n = Xv;
% Map test set into transformations generated by the training set (PS, Vn1t)
X2n = mapstd('apply',Xv,PS);
X2pet = Vn1t * X2n;
Xv = X2pet;
