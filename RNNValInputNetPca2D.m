classdef RNNValInputNetPca2D < MLPInputNetPca2D

    properties
        k_lob = 0;
        k_tob = 0;

        n_xy = 0;

    end

    methods
        function net = RNNValInputNetPca2D(x_pca, m_pca)

            net = net@MLPInputNetPca2D(x_pca, m_pca);

        end


        function [net, Xs, Ys, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob, net.Vit, Xp, Xcp, Xrp, Xsp] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo, net.x_pca);
            net.mb_size = 2^floor(log2(k_ob)-4);


            net.m_in = net.x_pca * net.t_in;

            mult = 1;
            net.k_hid1 = floor(mult * (net.m_in + 1));
            net.k_hid2 = floor(mult * (2*net.m_in + 1));
        end


        function [X2s, Y2s, Ysh2, Yshs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, X2s, Y2s, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob, Xp2, Xcp2, Xrp2, Xsp2] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob, net.x_pca, net.Vit);
        end


        function net = Train(net, i, X, Y)

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


            tNet = trainNetwork(CX, CY, net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph; 

        end

        function [X2, Y2] = Predict(net, X2, Y2, regNets, t_sess, sess_off, k_tob)

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