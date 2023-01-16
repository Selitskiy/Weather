classdef RNNBaseNet2D < BaseNet2D

    properties

    end

    methods
        function net = RNNBaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

        end

        function [Y, Y2, Yhs2] = ReScaleOut(net, Y, Y2, Yhs2, Bo, Bto, l_sess, t_sess, sess_off, offset, k_tob)

            for i = 1:t_sess-sess_off
                %MeanSess = Bo(3,:,i);
                %StdSess = Bo(4,:,i);

                for j = 1:k_tob

                    %idx = (i+sess_off)*l_sess + (j-1)*net.t_out + 1 + offset - net.t_in;

                    MeanSess = Bto(3,:,j,i);
                    StdSess = Bto(4,:,j,i);

                    %Myw = reshape( Y2(:, j, i), [net.y_out, net.t_out])';
                    Myw = Y2(:, :, j, i)';

                    %Myw = generic_mean_minmax_rescale2D(Myw, MeanSess, MinSess, MaxSess);
                    Myw = generic_mean_std_rescale2D(Myw, MeanSess, StdSess);

                    %My = reshape( Myw', [net.n_out,1] );
                    %Y2(:, j, i) = My(:);
                    Y2(:, :, j, i) = Myw';


                    Myw = Yhs2(:, :, j, i)';

                    %Myw = generic_mean_minmax_rescale2D(Myw, MeanSess, MinSess, MaxSess);
                    Myw = generic_mean_std_rescale2D(Myw, MeanSess, StdSess);

                    Yhs2(:, :, j, i) = Myw';

                end
            end
        end

        function [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2)
            [Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = generic_seq_calc_mape2D(Y2, Yh2, net.y_off, net.y_out, net.t_out); 
        end

        function [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2) 
            [Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_seq_calc_rmse2D(Y2, Yh2, net.y_off, net.y_out, net.t_out);
        end

        function [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_cont_rmse(net, Y2, Yh2) 
            [Ec, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_calc_cont_rmse_rnnv2D(Y2, Yh2, net.n_out, net.y_out);
        end

        function Err_graph(net, M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName)
            generic_seq_err_graph2D(M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, net.n_xy, k_tob, t_sess, sess_off, offset, l_marg, modelName);
        end

        %function TestIn_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg)
        %    generic_test_in_graph2D(M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg);
        %end

    end
end