classdef RNNValBaseNet2D < RNNBaseNet2D

    properties
    end

    methods

        function net = RNNValBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@RNNBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

        end


        %function [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2)
        %    [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = generic_seq_calc_mape2D(Y2, Yh2, net.y_off, net.y_out, net.t_out); 
        %end

        %function [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2) 
        %    [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_seq_calc_rmse2D(Y2, Yh2, net.y_off, net.y_out, net.t_out);
        %end

        %function [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_cont_rmse(net, Y2, Yh2) 
        %    [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_calc_cont_rmse_rnnv2D(Y2, Yh2, net.n_out, net.y_out);
        %end

        function Err_graph(net, M, E, E2, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName, yLab)
            generic_val_err_graph2D(M, E, E2, l_whole_ex, Y2, Sy2, l_whole, l_sess, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, k_tob, t_sess, sess_off, offset, l_marg, modelName, yLab);
        end

        %function TestIn_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg)
        %    generic_test_in_graph2D(M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg);
        %end

    end
end