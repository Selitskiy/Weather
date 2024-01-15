%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/ANNLib/');
addpath('~/Weather/');


%% Load data
dataDir = '~/data/Weather_data';



%Scenario 1
%dataFile = 'Measurements 04_2022-06_2023 Scenario1.xlsx';
%yLab = 'Soil Moisture (%)';
%Scenario 2
%dataFile = 'Measurements 04_2022-06_2023 Scenario2.xlsx';
%yLab = 'ORP Smooth_mV';
%Scenario 3
dataFile = 'Measurements 04_2022-06_2023 Scenario3.xlsx';
yLab = 'Water EC (muS/cm)';
%Scenario 4
%dataFile = 'Measurements 04_2022-06_2023 Scenario4.xlsx';
%yLab = 'PH Smooth';

dataFullName = strcat(dataDir,'/',dataFile);

%Number of days
d_mult = 7; %14; %7; %3;
d_div = 4; %2 hours %%36; %24; 12; 6; %experiment
part_mult = 1;
%part_mult = 5; %15 days

M_off = 1;
%Dilation
%M_div = 1;
%M_div = 14/d_mult; %2 weeks;
M_div = d_div; %experiment

Mti = readmatrix(dataFullName);

%Convert to 30 minutes interval
[n,~] = size(Mti);
t_sw = 27860;

Mt1 = Mti(1:3:t_sw-1,:);
Mt2 = Mti(t_sw:2:end,:);
Mt = vertcat(Mt1, Mt2);
m_day = 24 * 2;


% Scenario 1
%M = Mt(floor(M_off:M_div:end), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]);
% Scenario 2-4
M = Mt(floor(M_off:M_div:end), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]);

[l_whole_ex, ~] = size(M);


% input dimesion (parms x days)
% Scenario 1
x_off = 0;
x_in = 13;


%t_in = 144*d_mult;
t_in = floor(m_day/d_div*d_mult); %experiment

% output dimensions (parms x days)
% Scenario 2-4
y_off = 12;
y_out = 1;

% Scenario 1
%y_off = 10;
%y_out = 3;

%t_out = 144*d_mult;
t_out = 1; %floor(144/d_div*d_mult); %experiment
t_out_ae = t_in; %autoencoder span


%ts_out = 36*d_mult; 
ts_out = floor(36/d_div*d_mult); %experiment

l_whole = l_whole_ex;
% Leave space for last full label
%l_whole = l_whole_ex - t_out;
% Proportional to even number of whole outpouts
%l_whole = (floor(l_whole_ex/t_out)-1)*t_out; 


% Break the whole dataset in training sessions,
% Set training session length (space to slide window of size t_in datapoints, 
% plus length of last label t_out, plus size of input for test on next session), 
%l_sess = floor(12/d_mult*part_mult)*t_in + t_out + t_in; 
%l_sess = ceil(14/d_mult*part_mult)*t_in + 2*t_in + t_out_ae; %extra t_in for AE range

% 4 training days (3*t_in+1 batch)
% plus label (which is t_in for AE), plus space for test (both input and output peeking 1 position to other session)
l_sess = 4*t_in + t_out_ae + t_in + t_out_ae-1; 


% Test output period - if same as training period, will cover whole data
l_test = l_sess; %t_out; %l_sess;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);

% Check sessions fit
if (l_whole - n_sess*l_sess) < l_test
    n_sess = n_sess - 1;
end



%Clean up outliers
for i = 1:l_whole_ex
    for j = 1:y_out
        if M(i, y_off+j) == 0
           Mav = (M(i-5, y_off+j) + M(i+5, y_off+j))/2;
           fprintf('Data clean up i=%d j=%d M=%f -> M=%f\n', i, j, M(i, y_off+j), Mav);
           M(i, y_off+j) = Mav;
        end
    end
end



ini_rate = 0.001; 
max_epoch = 500; %mlp 2000;%200;
%max_epoch = 40; %200; %rnn seq %20; %rnn vect

norm_fli = 1;
norm_flo = 1;

%injevted metadata (timestams and log timestamps)
k_inj = 2;

drop_out = 0;

regNets = cell([n_sess, 1]);
identNets = cell([n_sess, 1]);

%Models direrctory
dataModDir = '~/data/Weather_data';

useRetrain = 0;

%% Train or pre-load regNets
for i = 1:n_sess

    %%regNet = LinRegNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = AnnNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = ReluNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = KgNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = SigNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = TanhNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = RbfNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = VTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = TransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = BtransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = DpTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = DpBatchTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = Dp2BatchTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = Dp2BTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = VaswaniTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = ReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    %%regNet = TTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = Lr2ReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = Lr3ReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = Lr4ReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    %regNet = TLr3ReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = BTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = TBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    %%regNet = TBLrBAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = TLrTLrAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = LrBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = GlBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = LLrBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = TnLlrTTnLlrBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    %regNet = TnBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = TTnBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = TnTTnBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    

    %regNet = resReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = res2LrReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = res3LrReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = resTTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    regNet = resT3LrReLUAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = resBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = resTBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %regNet = resTnTTnBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);

    %%regNet = resBBTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);
    %%regNet = res3BTransAENet2D(x_off, x_in, t_in, y_off, y_out, t_out, t_out_ae, ini_rate, max_epoch, k_inj, 3/x_in);


    %regNet = SeqCnnMlpNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = SeqCnnNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = SeqCnnSpecNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = SeqCnnSpecTNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnMlpNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnGruNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnGruSpecNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnSpecNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = CnnSpec2Net2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    
    %regNet = LstmValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = GruValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = GruVal3Net2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = LinRegNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = AnnNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = ReluNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = KgNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);

    %regNet = SigNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = TanhNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = RbfNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
    %regNet = TransNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);

    %regNet = LstmNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch); %todo
    %regNet = GruNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);


    %regNet = ReluNetPca2D(x_off, x_in, t_in, y_off, y_out, t_out, 5, 0, ini_rate, max_epoch);

    %regNet = LstmValNetPca2D(x_off, x_in, t_in, y_off, y_out, t_out, 5, 0, ini_rate, max_epoch);


    [regNet, X, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = regNet.TrainTensors(M, l_sess, n_sess, norm_fli, norm_flo);

    
    modelName = regNet.name;

    dataModFile = strcat(dataModDir,'/', 'wearther.', modelName, '.', string(i), '.', string(n_sess),...
        '.', string(M_off), '.', string(M_div), '.', dataFile,...
        '.', string(x_off), '.', string(x_in), '.', string(t_in),...
        '.', string(y_off), '.', string(y_out), '.', string(t_out),'.', string(t_out_ae),...
        '.', string(norm_fli), '.', string(norm_flo), '.', string(ini_rate), '.', string(max_epoch), '.mat');

    if ~isfile(dataModFile)

        if (useRetrain) && (i>1)
            regNet.trainedNet = regNets{i-1}.trainedNet;
            regNet.lGraph = regNets{i-1}.trainedNet.layerGraph;   
        end

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
    
        regNet = regNet.Train(i, X, Y);

        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]);    

        fprintf('Saving %s %d\n', dataModFile, i);
        save(dataModFile, 'regNet');

    else

        fprintf('Loading %s %d\n', dataModFile, i);
        load(dataModFile, 'regNet');

    end


    regNets{i} = regNet;

end



%% Attention Input Identity net
% Train or pre-load Identity nets

max_epoch = 20;
useIdentNets = 0;

for i = 1:n_sess

    identNet = ReluNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);
    identNet.mb_size = 32;
    identNet = identNet.Create();

    dataIdentFile = strcat(dataModDir,'/', 'wearther.ident.', string(i), '.', string(n_sess),...
        '.', string(M_off), '.', string(M_div), '.', dataFile,...
        '.', string(x_off), '.', string(x_in), '.', string(t_in),...
        '.', string(y_off), '.', string(y_out), '.', string(t_out),'.', string(t_out_ae),...
        '.', string(norm_fli), '.', string(norm_flo), '.', string(ini_rate), '.', string(max_epoch), '.mat');

    if useIdentNets ~= 0

        if isfile(dataIdentFile)
            fprintf('Loading Ident net %d from %s\n', i, dataIdentFile);
            load(dataIdentFile, 'identNet');
        else

            fprintf('Training Ident net %d\n', i);

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

            tNet = trainNetwork(XI(:, 1:k_ob*i)', C(1:k_ob*i)', identNet.lGraph, identNet.options);

            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);  

            identNet.trainedNet = tNet;
            identNet.lGraph = tNet.layerGraph; 

            save(dataIdentFile, 'identNet');
        end

        identNets{i} = identNet;

    end
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
[X2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = regNets{1}.TestTensors(M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);

%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));
    
[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, XI2, identNets, t_sess, sess_off, k_tob);

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% re-scale in observation bounds
%if(norm_fli)
%    [X, X2] = regNets{1}.ReScaleIn(X, X2, Bi, n_sess, t_sess, sess_off, k_ob, k_tob);
%end

if(norm_flo)
    [Y, Y2, Yhs2, Ya2, Yha2, Sa] = regNets{1}.ReScaleOut(Y, Y2, Yhs2, Bo, Bto, n_sess, t_sess, sess_off, k_ob, k_tob, Yh2);
end

%% Calculate errors
[Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Ya2, Yha2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MeanMaxAPErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


[Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Ya2, Yha2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MeanMaxRSErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));


[Ec, S2C, S2MeanC, S2StdC, S2sC, ma_errC, sess_ma_idxC, ob_ma_idxC, mi_errC, sess_mi_idxC, ob_mi_idxC] = regNets{1}.Calc_cont_rmse(Ya2, Yha2, k_tob); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, Cont RMSErr: %f+-%f MeanMaxRSErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2C, S2StdC, mean(ma_errC), std(ma_errC));

%%
regNets{1}.Err_graph(M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName, yLab, Ya2, Sa);

%%
%regNets{1}.TestIn_graph(M, l_whole_ex, X, Y, X2, Y2, Sx, Sy, Sx2, Sy2, l_whole, n_sess, l_sess, k_ob, k_tob, t_sess, sess_off, offset, l_marg, modelName);

%% Write results

%% Write per-image scores to a file
outFile = strcat(dataModDir,'/', 'wearther_out.', modelName, '.', string(n_sess),...
        '.', string(M_off), '.', string(M_div), '.', dataFile,...
        '.', string(x_off), '.', string(x_in), '.', string(t_in),...
        '.', string(y_off), '.', string(y_out), '.', string(t_out), '.', string(t_out_ae),...
        '.', string(norm_fli), '.', string(norm_flo), '.', string(ini_rate), '.', string(max_epoch), '.txt');

fd = fopen( outFile,'w' );

fprintf(fd, "MAPErr: %f+-%f  RMSErr: %f+-%f CRMSErr: %f+-%f\n", S2Mean, S2Std, S2MeanQ, S2StdQ, S2MeanC, S2StdC);

fclose(fd);


%Dout.M = M;
%Dout.Y2 = Y2;
%Dout.Sy2 = Sy2;
%outFile = strcat(dataModDir,'/', 'wearther_out.', modelName, '.', string(n_sess),...
%        '.', string(M_off), '.', string(M_div), '.', dataFile,...
%        '.', string(x_off), '.', string(x_in), '.', string(t_in),...
%        '.', string(y_off), '.', string(y_out), '.', string(t_out),...
%        '.', string(norm_fli), '.', string(norm_flo), '.', string(ini_rate), '.', string(max_epoch), '.mat');

%fprintf('Saving %s\n', outFile);
%save(outFile, 'Dout');