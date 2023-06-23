classdef ReluNetCl < ReluLayersCl & BaseNetCl %& MLPInputNet2D

    properties

    end

    methods
        function net = ReluNetCl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch)

            net = net@ReluLayersCl();
            net = net@BaseNetCl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);
            %net = net@MLPInputNet2D();

            net.name = "relucl";

        end


        %function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

        %    [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@MLPInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

        %    net = Create(net);

        %end



        %function net = Train(net, i, X, Y)
        %    fprintf('Training %s Class net %d\n', net.name, i); 

        %    net = Train@MLPInputNet2D(net, i, X, Y);
        %end

        
    end
end