classdef LinRegInputNetSeq2D < LinRegInputNet2D

    properties
        %ts_out = 0;
        %n_xy = 0;
        %n_out2 = 0;
    end

    methods
        function net = LinRegInputNetSeq2D()%ts_out)
            net = net@LinRegInputNet2D();

            %net.ts_out = ts_out;
        end


        function [net, Xr, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, n_xy, k_ob] = generic_train_ar_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.ts_out, l_sess, n_sess, norm_fli, norm_flo);

            net.n_xy = n_xy;
            net.n_out = net.n_xy * net.ts_out;
            net.n_out2 = net.n_xy * net.t_out;
            net.W1 = zeros([net.n_xy*net.ts_out, net.m_in+1, n_sess]);
        end

        function [Xr2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_ar_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, net.n_xy, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);
        end



        function [Xr2, Y2] = Predict(net, Xr2, Y2, regNets, t_sess, sess_off, k_tob)

            for i = 1:t_sess-sess_off
                W1 = regNets{i}.W1;

                Xr2i = Xr2(:, :, i);
                W1i = W1(:, : ,i);

                for j = 1:net.ts_out:net.t_out
                    Y2t = W1i * Xr2i;
                    Y2((j-1)/net.ts_out*net.n_out+1:((j-1)/net.ts_out+1)*net.n_out, :, i) = Y2t;
                    Xr2i(1:end-1-net.n_xy*net.ts_out, :) = Xr2i(1+net.n_xy*net.ts_out:end-1, :);
                    for k = 1:net.ts_out
                        Xr2i(end-net.x_in*(net.ts_out-k+1):end-net.x_in*(net.ts_out-k)-1,:) = Y2t(1 + (k-1)*net.x_in:k*net.x_in, :);
                    end
                end
            end

        end
        
    end
end