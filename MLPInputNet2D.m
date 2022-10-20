classdef MLPInputNet2D

    properties
        mb_size

    end

    methods

        function net = MLPInputNet2D()

            net.mb_size = [];

        end


        function [net, X, Y, Bi, Bo, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fl)
            [X, Xc, Xr, Ys, Y, Bi, Bo, XI, C, k_ob] = generic_train_tensors2D(M, net.x_in, net.t_in, net.y_out, net.t_out, l_sess, n_sess, norm_fl);
            net.mb_size = 2^floor(log2(k_ob)-3);
        end


        function [X2, Y2, Yh2, Bti, Bto, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bti, Bto, k_tob] = generic_test_tensors2D(M, net.x_in, net.t_in, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, 0);
        end


        function net = Train(net, i, X, Y)
            tNet = trainNetwork(X(:, :, i)', Y(:, :, i)', net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            
        end


        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                predictedScores = predict(regNets{i}.trainedNet, X2(:, :, i)');
                Y2(:, :, i) = predictedScores';
            end
        end
        
    end

end