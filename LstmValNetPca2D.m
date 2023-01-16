classdef LstmValNetPca2D < LstmValLayers2D & RNNValBaseNet2D & RNNValInputNetPca2D

    properties

    end

    methods
        function net = LstmValNetPca2D(x_off, x_in, t_in, y_off, y_out, t_out, x_pca, m_pca, ini_rate, max_epoch)

            net = net@LstmValLayers2D();
            net = net@RNNValBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@RNNValInputNetPca2D(x_pca, m_pca);

            net.name = "lstmpca2dval";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@RNNValInputNetPca2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@RNNValInputNetPca2D(net, i, X, Y);
 
        end


    end

end