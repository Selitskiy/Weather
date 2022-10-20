classdef LinRegInputNet2D

    properties
        W1

    end

    methods
        function net = LinRegInputNet2D()
            W1 = [];

        end


        function [net, Xr, Y, B, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fl)
            [X, Xc, Xr, Ys, Y, B, XI, C, k_ob] = generic_train_tensors2D(M, net.x_in, net.t_in, net.y_out, net.t_out, l_sess, n_sess, norm_fl);

            net.W1 = zeros([net.n_out, net.m_in+1, n_sess]);
        end


        function [Xr2, Y2, Yh2, Bt, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fl)
            [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bt, k_tob] = generic_test_tensors2D(M, net.x_in, net.t_in, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, 0);
        end


        function net = Train(net, i, X, Y)

            %W1 = zeros([net.n_out, net.m_in+1, net.n_sess]);
            %parfor i = 1:net.n_sess
                Xi = X(:, :, i);
                Yi = Y(:, :, i);
                XiT = Xi.';
                net.W1(:,:,i) = Yi * XiT / (Xi * XiT);
            %end

            %net.W1 = W1;
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