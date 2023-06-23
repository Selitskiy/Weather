classdef CnnSpec2Net2D < CnnSpec2Layers2D & BaseNet2D & CNNInputNet2D

    properties
    end

    methods

        function net = CnnSpec2Net2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@CnnSpec2Layers2D();
            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@CNNInputNet2D();

            net.name = "cnn2dspec2";

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