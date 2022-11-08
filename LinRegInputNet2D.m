classdef LinRegInputNet2D

    properties
        W1 = [];

    end

    methods
        function net = LinRegInputNet2D()
        end


        function [net, Xr, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Ys, Y, Bi, Bo, XI, C, Sx, Sys, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);

            net.W1 = zeros([net.n_out, net.m_in+1, n_sess]);
        end


        function [Xr2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
        end


        function net = Train(net, i, X, Y)

                Xi = X(:, :, i);
                Yi = Y(:, :, i);
                XiT = Xi.';
                net.W1(:,:,i) = Yi * XiT / (Xi * XiT);

        end

        function [Xr2, Y2] = Predict(net, Xr2, Y2, regNets, t_sess, sess_off, k_tob)

            for i = 1:t_sess
                W1 = regNets{i}.W1;

                Xr2i = Xr2(:, :, i);
                W1i = W1(:, : ,i);
                Y2(:, :, i) = W1i * Xr2i;
            end

        end
        
    end
end