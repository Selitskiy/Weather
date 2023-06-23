classdef CnnGruSpecNet2D < CnnGruSpecLayers2D & RNNValBaseNet2D & CNNRNNInputNet2D

    properties
    end

    methods

        function net = CnnGruSpecNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@CnnGruSpecLayers2D();
            net = net@RNNValBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@CNNRNNInputNet2D();

            net.name = "cnngruspec2d";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@CNNRNNInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@CNNRNNInputNet2D(net, i, X, Y);
 
        end



    end

end