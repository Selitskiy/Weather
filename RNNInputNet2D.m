classdef RNNInputNet2D < MLPInputNet2D

    properties
        k_lob = 0;
        k_tob = 0;

        n_xy = 0;

        reset_fl = 0;
    end

    methods
        function net = RNNInputNet2D()

            net = net@MLPInputNet2D();

        end


        function [net, Xl, Yl, Bli, Blo, Sx, Sy, k_lob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [Xl, Yl, Bli, Blo, Sx, Sy, n_xy, k_lob] = generic_train_seq_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo);
            net.n_xy = n_xy;
            net.k_lob = k_lob;
            net.mb_size = 2^floor(log2(k_lob)-4);
        end


        function [X2, Y2c, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_seq_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, net.n_xy, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, net.k_lob, k_tob);

            Y2c = struct;
            Y2c.Y2s = Y2s;
            Y2c.Y2 = Y2;
        end


        function net = Train(net, i, Xl, Yl)

            tNet = trainNetwork(Xl(:, :, i), Yl(:, :, i), net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            

        end

        function [X2, Y2] = Predict(net, X2, Y2c, regNets, t_sess, sess_off, k_tob)

            Y2s = Y2c.Y2s;
            Y2 = Y2c.Y2;

            for i = 1:t_sess-sess_off

                if( net.reset_fl )
                    lstmNet = resetState(regNets{i}.trainedNet);
                    % Trick - go through all input sequence to get the first next
                    % prediction outside the seqence, discarding intermediate predictions

                    %Either
                    %for k = 1:net.k_lob %+1
                    %    [lstmNet, Y2(:, 1, 1, i)] = predictAndUpdateState(lstmNet, X2(:, k, 1, i));
                    %end
                    %X2Last = X2(:, net.k_lob+1, 1, i);
                    %or
                    [lstmNet, Y2s(:, :, 1, i)] = predictAndUpdateState(lstmNet, X2(:, 1:end-1, 1, i));
                    %Y2(:, 1, 1, i) = Y2s(:, end, 1, i);
                    X2Last = X2(:, end, 1, i);
                else
                    %Without reset
                    lstmNet = regNets{i}.trainedNet;
                    %[lstmNet, Y2s(:, end, 1, i)] = predictAndUpdateState(lstmNet, X2(:, end-1, 1, i));
                    %Y2(:, 1, 1, i) = Y2s(:, end, 1, i);
                    X2Last = X2(:, end, 1, i);
                end
        
                % Now feeding test data
                for j = 1:k_tob

                    %lstmNet = resetState(regNets{i}.trainedNet);
                    % Trick - go through all input sequence to get the first next
                    % prediction outside the seqence, discarding intermediate predictions

                    %Either
                    %for k = 1:net.k_lob+1
                    %    [lstmNet, Y2(:, 1, j, i)] = predictAndUpdateState(lstmNet, X2(:, k, j, i));
                    %end
                    %or
                    %[lstmNet, Y2s(:, :, j, i)] = predictAndUpdateState(lstmNet, X2(:, :, j, i));
                    %Y2(:, 1, j, i) = Y2s(:, end, j, i);

                    %%Without reset
                    %%lstmNet = regNets{i}.trainedNet;
                    %%[lstmNet, Y2s(:, end, j, i)] = predictAndUpdateState(lstmNet, X2(:, end, j, i));
                    %%Y2(:, 1, j, i) = Y2s(:, end, j, i);


                    [lstmNet, Y2(:, 1, j, i)] = predictAndUpdateState(lstmNet, X2Last);

                    % Continue predicting further output points based on previous
                    % predicvtion
                    for l = 2:net.t_out
                        [lstmNet, Y2(:, l, j, i)] = predictAndUpdateState(lstmNet, Y2(1:net.x_in, l-1, j, i));
                    end

                    X2Last = Y2(1:net.x_in, net.t_out, j, i);
                
                end

            end

        end
        
    end
end