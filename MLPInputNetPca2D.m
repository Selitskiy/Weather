classdef MLPInputNetPca2D < MLPInputNet2D

    properties
        Vit = [];
        Vi = [];
        V = [];
        x_pca = 0;
        m_pca = 0;
    end

    methods

        function net = MLPInputNetPca2D(x_pca, m_pca)
            net = net@MLPInputNet2D();

            net.x_pca = x_pca;
            net.m_pca = m_pca;
        end


        function [net, Xp, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob, net.Vit, Xp, Xcp, Xrp, Xsp] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo, net.x_pca);
            net.mb_size = 2^floor(log2(k_ob)-4);

            %[net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@MLPInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);


            %if net.m_pca == 0
            %   net.m_pca = net.m_in; 
            %end

            %Xp = zeros([net.m_pca, k_ob, n_sess]);
            %net.Vit = zeros([net.m_pca, net.m_in, n_sess]);
            %net.Vi = zeros([net.m_in, net.m_in, n_sess]);
            %net.V = zeros([net.m_in, net.m_in, n_sess]);

            %for i = 1:n_sess
            %    [Xp(:,:,i), net.Vit(:,:,i), net.Vi(:,:,i), net.V(:,:,i), It] = pca_create(X(:,:,i), net.m_pca, 0);
            %end

            net.m_in = net.x_pca * net.t_in;



            mult = 1;
            net.k_hid1 = floor(mult * (net.m_in + 1));
            net.k_hid2 = floor(mult * (2*net.m_in + 1));
        end


        function [Xp2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob, Xp2, Xcp2, Xrp2, Xsp2] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob, net.x_pca, net.Vit);
            
            %[X2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors@MLPInputNet2D(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
                        
            %Xt2 = zeros([net.m_in, k_tob, t_sess-sess_off]);
            %for i = 1:t_sess-sess_off
            %    [Xt2(:,:,i)] = pca_map(X2(:,:,i), net.Vit(:,:,i));
            %end
        end

        
    end

end