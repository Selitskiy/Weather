classdef RNNBaseNet2D < BaseNet2D

    properties

    end

    methods
        function net = RNNBaseNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch)

            net = net@BaseNet2D(x_in, t_in, y_out, t_out, ini_rate, max_epoch);

        end

        function [Y2, Yhs2] = ReScale(net, Y2, Yhs2, Bo, l_sess, t_sess, sess_off, offset, k_tob)
            for i = 1:t_sess-sess_off
                MinSess = min(Bo(1,:,:,i), [], 3);
                MaxSess = max(Bo(2,:,:,i), [], 3);
                MeanSess = mean( Bo(3,:,:,i), 3);
                StdSess = mean( Bo(4,:,:,i), 3);

                for j = 1:k_tob

                    idx = (i+sess_off)*l_sess + (j-1)*net.t_out + 1 + offset - net.t_in;

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

        function [S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = Calc_mape(net, Y2, Yh2)
            [S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = generic_seq_calc_mape2D(Y2, Yh2, net.x_in, net.y_out, net.t_out); 
        end

        function [S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = Calc_rmse(net, Y2, Yh2) 
            [S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = generic_seq_calc_rmse2D(Y2, Yh2, net.x_in, net.y_out, net.t_out);
        end

        function Err_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg, modelName)
            generic_seq_err_graph2D(M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg, modelName);
        end

        %function TestIn_graph(net, M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg)
        %    generic_test_in_graph2D(M, l_whole_ex, Y2, l_whole, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, l_marg);
        %end

    end
end