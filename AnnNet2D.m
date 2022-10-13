classdef AnnNet2D < BaseNet2D & MLPInputNet2D

    properties

    end

    methods
        function net = AnnNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch)

            net = net@BaseNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);
            net = net@MLPInputNet2D();

            net.name = "ann2d";



        end


        function [net, X, Y, B, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fl)

            [net, X, Y, B, k_ob] = TrainTensors@MLPInputNet2D(net, M, net.x_in, net.t_in, net.y_out, net.t_out, l_sess, n_sess, norm_fl);

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1)
                fullyConnectedLayer(net.k_hid2)
                fullyConnectedLayer(net.n_out)
                regressionLayer
            ];

            net.lGraph = layerGraph(layers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'Plots', 'training-progress',...
                'MaxEpochs',net.max_epoch);
        end



        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@MLPInputNet2D(net, i, X, Y);
        end

        
    end
end