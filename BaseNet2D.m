classdef BaseNet2D

    properties
        name = [];

        x_in
        t_in
        m_in

        y_out
        t_out
        n_out

        k_hid1
        k_hid2
        ini_rate 
        max_epoch
    
        lGraph = [];
        options = [];
        trainedNet = [];
    end

    methods
        function net = BaseNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch)

            net.x_in = x_in;
            net.t_in = t_in;
            net.m_in = x_in * t_in;

            net.y_out = y_out;
            net.t_out = t_out;
            net.n_out = y_out * t_out;

            mult = 1;
            net.k_hid1 = floor(mult * (net.m_in + 1));
            net.k_hid2 = floor(mult * (2*net.m_in + 1));
            net.ini_rate = ini_rate;
            net.max_epoch = max_epoch;

        end

        function [Y2, Yh2] = ReScale(net, Y2, Yh2, Bt, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off
                for j = 1:k_tob
                    Y2(:, j, i) = generic_minmax_rescale2D(Y2(:, j, i), Bt(1,:,j,i), Bt(2,:,j,i));
                    %Yh2(:, j, i) = generic_minmax_rescale2D(Yh2(:, j, i), Bt(1,:,j,i), Bt(2,:,j,i));
                end
            end
        end

        function [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2)
            [S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = generic_calc_mape2D(Y2, Yh2, net.n_out); 
        end

        function [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2) 
            [S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_calc_rmse2D(Y2, Yh2, net.n_out);
        end

        function Err_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg)
            w_series_generic_err_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);
        end

    end
end