classdef RNNValInputNet2D < MLPInputNet2D

    properties
        k_lob = 0;
        k_tob = 0;

        n_xy = 0;

        %reset_fl = 0;
    end

    methods
        function net = RNNValInputNet2D()

            net = net@MLPInputNet2D();

        end


        function [net, Xs, Ys, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            %[X, Y, Bi, Bo, Sx, Sy, n_xy, k_ob] = generic_train_seq_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);
            %Yc = struct;
            %Yc.Ys = Ys;
            %Yc.Y = Y;
            %net.n_xy = n_xy;
            %net.k_lob = k_lob;
            %net.mb_size = 2^floor(log2(k_lob)-4);
            net.mb_size = 2^floor(log2(k_ob)-4);
        end


        function [X2s, Y2s, Ysh2, Yshs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, X2s, Y2s, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);

            %Y2c = struct;
            %Y2c.Y2s = Y2s;
            %Y2c.Y2 = Y2;
        end


        function net = Train(net, i, X, Y)

            %Xi = permute(X(:,:,:,i), [3 1 2]);
            %Yi = permute(Y(:,:,:,i), [3 1 2]);

            [k,n,m] = size(X(:,:,:,i));
            CX = cell(m,1);
            for j=1:m
                CX{j}=X(:,:,j,i);
            end

            [k,n,m] = size(Y(:,:,:,i));
            CY = cell(m,1);
            for j=1:m
                CY{j}=Y(:,:,j,i);
            end

            %CX = num2cell(Xi, [2, 3]);
            %CY = num2cell(Yi, [2, 3]);

            tNet = trainNetwork(CX, CY, net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph; 

        end

        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)

            %Y2s = Y2c.Y2s;
            %Y2 = Y2c.Y2;

            for i = 1:t_sess-sess_off

                % Now feeding test data
                for j = 1:k_tob
                    lstmNet = regNets{i}.trainedNet;
                    Y2(:, :, j, i) = predict(lstmNet, X2(:, :, j, i));
                end

            end


        end
        
    end
end