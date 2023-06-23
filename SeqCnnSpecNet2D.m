classdef SeqCnnSpecNet2D < SeqCnnSpecLayers2D & BaseNet2D & SeqCNNRNNInputNet2D

    properties
    end

    methods

        function net = SeqCnnSpecNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@SeqCnnSpecLayers2D();
            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@SeqCNNRNNInputNet2D();

            net.name = "seqcnn2dspec";

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

            [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@SeqCNNRNNInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

            net = Create(net);

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@SeqCNNRNNInputNet2D(net, i, X, Y);
 
        end



    end

end