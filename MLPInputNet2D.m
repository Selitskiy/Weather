classdef MLPInputNet2D

    properties
        mb_size = [];
    end

    methods

        function net = MLPInputNet2D()

        end


        function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)
            [X, Xc, Xr, Xs, Ys, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = generic_train_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, n_sess, norm_fli, norm_flo, net.x_in);
            net.mb_size = 2^floor(log2(k_ob)-4);
        end


        function [X2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = TestTensors(net, M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
            [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = generic_test_tensors2D(M, net.x_off, net.x_in, net.t_in, net.y_off, net.y_out, net.t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob, net.x_in, []);
        end


        function net = Train(net, i, X, Y)
            tNet = trainNetwork(X(:, :, i)', Y(:, :, i)', net.lGraph, net.options);
            net.trainedNet = tNet;
            net.lGraph = tNet.layerGraph;            
        end


        function [X2, Y2] = Predict(net, X2, Y2, regNets, identNets, t_sess, sess_off, k_tob)
            for i = 1:t_sess-sess_off

                % GPU on
gpuDevice(1);
reset(gpuDevice(1));

                %for j = 1:k_tob
                    if exist('identNets') == 0
                        regNet = regNets{i}.trainedNet;
                    else
                        predictClass = classify(identNets{i+sess_off}, X2(:, :, i)');
                        prClNum = double(predictClass);
                        fprintf('IdentityClass Session:%d, Observation:%d, IdentClNum:%d\n', i, j, prClNum);

                        regNet = resetState(regNets{prClNum});
                    end

                    predictedScores = predict(regNet, X2(:, :, i)');
                    Y2(:, :, i) = predictedScores';
                %end

                % GPU off
delete(gcp('nocreate'));
gpuDevice([]);

            end
        end
        
    end

end