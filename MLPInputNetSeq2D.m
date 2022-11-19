classdef MLPInputNetSeq2D < MLPInputNet2D

    properties
    end

    methods

        function net = MLPInputNetSeq2D()

            net = net@MLPInputNet2D();
        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, n_xy, k_ob] = generic_train_ar_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.ts_out, l_sess, n_sess, norm_fli, norm_flo);
            net.mb_size = 2^floor(log2(k_ob)-4);

            net.n_xy = n_xy;
            net.n_out = net.n_xy * net.ts_out;
            net.n_out2 = net.n_xy * net.t_out;
        end

        function [X2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_ar_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, net.n_xy, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
        end


        %function net = Train(net, i, X, Y)
        %    tNet = trainNetwork(X(:, :, i)', Y(:, :, i)', net.lGraph, net.options);
        %    net.trainedNet = tNet;
        %    net.lGraph = tNet.layerGraph;            
        %end


        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off

                %predictedScores = predict(regNets{i}.trainedNet, X2(:, :, i)');
                %Y2(:, :, i) = predictedScores';

                %W1 = regNets{i}.W1;

                X2i = X2(:, :, i);
                %W1i = W1(:, : ,i);

                for j = 1:net.ts_out:net.t_out
                    %Y2t = W1i * Xr2i;
                    predictedScores = predict(regNets{i}.trainedNet, X2i');
                    Y2t = predictedScores';

                    Y2((j-1)/net.ts_out*net.n_out+1:((j-1)/net.ts_out+1)*net.n_out, :, i) = Y2t;
                    X2i(1:end-net.n_xy*net.ts_out, :) = X2i(1+net.n_xy*net.ts_out:end, :);
                    for k = 1:net.ts_out
                        X2i(end+1-net.x_in*(net.ts_out-k+1):end-net.x_in*(net.ts_out-k),:) = Y2t(1 + (k-1)*net.x_in:k*net.x_in, :);
                    end
                end
            end
        end
        
    end

end