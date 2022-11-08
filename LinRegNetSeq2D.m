classdef LinRegNetSeq2D < BaseNetSeq2D & LinRegInputNetSeq2D

    properties

    end

    methods
        function net = LinRegNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch)

            net = net@BaseNetSeq2D(x_off, x_in, t_in, y_off, y_out, t_out, ts_out, ini_rate, max_epoch);
            net = net@LinRegInputNetSeq2D();%ts_out);

            net.name = "reg2dseq";

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@LinRegInputNetSeq2D(net, i, X, Y);
        end


        %function [Y, Y2, Yhs2] = ReScaleOut(net, Y, Y2, Yhs2, Bo, n_sess, t_sess, sess_off, k_ob, k_tob)


        %    for i = 1:t_sess-sess_off
                %MinSess = min(Bo(1,:,:,i), [], 3);
                %MaxSess = max(Bo(2,:,:,i), [], 3);
        %        MeanSess = Bo(3,:,i);
        %        StdSess = Bo(4,:,i);

        %        for j = 1:k_tob

                    %idx = (i+sess_off)*l_sess + (j-1)*net.t_out + 1 + offset - net.t_in;

        %            Myw = reshape( Y2(:, j, i), [net.n_xy, net.t_out])';

                    %Myw = generic_mean_minmax_rescale2D(Myw, MeanSess, MinSess, MaxSess);
        %            Myw = generic_mean_std_rescale2D(Myw, MeanSess, StdSess);

        %            My = reshape( Myw', [net.n_out2,1] );
        %            Y2(:, j, i) = My(:);


        %            Myw = reshape( Yhs2(:, j, i), [net.n_xy, net.t_out])';

                    %Myw = generic_mean_minmax_rescale2D(Myw, MeanSess, MinSess, MaxSess);
        %            Myw = generic_mean_std_rescale2D(Myw, MeanSess, StdSess);

        %            My = reshape( Myw', [net.n_out2,1] );
        %            Yhs2(:, j, i) = My(:);

        %        end
        %    end
        %end

        %function Err_graph(net, M, l_whole_ex, Y2, Sy2, E, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName)
        %    generic_ar_err_graph2D(M, l_whole_ex, Y2, Sy2, E, l_whole, l_sess, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, net.n_xy, k_tob, t_sess, sess_off, offset, l_marg, modelName);
        %end

    end
end