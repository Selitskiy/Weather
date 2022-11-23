classdef GruNet2D < GruLayers2D & RNNBaseNet2D & RNNInputNet2D

    properties

    end

    methods
        function net = GruNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@GruLayers2D();
            net = net@RNNBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@RNNInputNet2D();

            net.name = "gru2dseq";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@RNNInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@RNNInputNet2D(net, i, X, Y);
 
        end


    end

end