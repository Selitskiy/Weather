classdef TransNetSeq2D < TransLayers2D & BaseNetSeq2D & MLPInputNetSeq2D

    properties

    end

    methods
        function net = TransNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch)

            net = net@TransLayers2D();
            net = net@BaseNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
            net = net@MLPInputNetSeq2D();

            net.name = "trans2dseq";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@MLPInputNetSeq2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end



        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@MLPInputNetSeq2D(net, i, X, Y);
        end

        
    end
end