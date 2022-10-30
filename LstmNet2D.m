classdef LstmNet2D < RNNBaseNet2D & RNNInputNet2D

    properties

    end

    methods
        function net = LstmNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch)

            net = net@RNNBaseNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
            net = net@RNNInputNet2D();

            net.name = "lstm2d";

        end


        function [net, X, Y, Bi, Bo, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, k_ob] = TrainTensors@RNNInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            sLayers = [
                sequenceInputLayer(net.x_in+net.y_out)
                lstmLayer(net.k_hid1)%, 'OutputMode','last')
                lstmLayer(net.k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(net.x_in+net.y_out)
                regressionLayer
            ];

            net.lGraph = layerGraph(sLayers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@RNNInputNet2D(net, i, X, Y);
 
        end


    end

end