classdef SigNet2D < SigLayers2D & BaseNet2D & MLPInputNet2D

    properties

    end

    methods
        function net = SigNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@SigLayers2D();
            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@MLPInputNet2D();

            net.name = "sig2d";

        end


        function [net, X, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = TrainTensors@MLPInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end



        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@MLPInputNet2D(net, i, X, Y);
        end

        
    end
end