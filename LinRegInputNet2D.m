classdef LinRegInputNet2D

    properties
        W1 = [];

    end

    methods
        function net = LinRegInputNet2D()
        end


        function [net, Xr, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo, net.x_in);

            net.W1 = zeros([net.n_out, net.m_in+1, n_sess]);
        end


        function [Xr2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob, net.x_in, []);

        end


        function net = Train(net, i, X, Y)

                Xi = X(:, :, i);
                Yi = Y(:, :, i);
                XiT = Xi.';
                net.W1(:,:,i) = Yi * XiT / (Xi * XiT);

        end

        function [Xr2, Y2] = Predict(net, Xr2, Y2, regNets, XI2, identNets, t_sess, sess_off, k_tob)

            for i = 1:t_sess-sess_off

                % GPU on
                gpuDevice(1);
                reset(gpuDevice(1));

                for j = 1:k_tob
                    if size(identNets{1}) == 0
                        fprintf('IdentityClass Session:%d, Observation:%d\n', i, j);
                        regNet = regNets{i};
                    else
                        i_idx = (i-1)*k_tob + j;
                        predictClass = classify(identNets{i+sess_off}.trainedNet, XI2(:, i_idx)'); %X2(:, j, i)');
                        prClNum = double(predictClass);
                        fprintf('IdentityClass Session:%d, Observation:%d, IdentClNum:%d\n', i, j, prClNum);

                        regNet = regNets{prClNum};
                    end

                    W1 = regNet.W1;

                    Xr2i = Xr2(:, j, i);
                    W1i = W1(:, :, i);
                    Y2(:, j, i) = W1i * Xr2i;
                end

                % GPU off
                delete(gcp('nocreate'));
                gpuDevice([]);

            end

        end
        
    end
end