classdef SeqCNNRNNInputNet2D < MLPInputNet2D

    properties

    end

    methods
        function net = SeqCNNRNNInputNet2D()

            net = net@MLPInputNet2D();

        end


        function [net, Xc, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);
            net.mb_size = 2^floor(log2(k_ob)-4);
        end

        function [Xc2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
        end


        function net = Train(net, i, X, Y)

            %[m,n,k,l] = size(X);
            %Xi = permute(X(:, :, :, :, i), [1 3 2 4]);

            tNet = trainNetwork(X(:, :, :, :, i), Y(:, :, i), net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            
        end

        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                predictedScores = predict(regNets{i}.trainedNet, X2(:, :, :, :, i));
                Y2(:, :, i) = predictedScores;
            end
        end
        
    end
end