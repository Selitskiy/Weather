classdef CnnMlpNet2D < CnnMlpLayers2D & BaseNet2D & CNNInputNet2D

    properties
    end

    methods

        function net = CnnMlpNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@CnnMlpLayers2D();
            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@CNNInputNet2D();

            net.name = "cnnmlp2d";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@CNNInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@CNNInputNet2D(net, i, X, Y);
 
        end



    end

end