classdef LstmValLayers2D

    properties
    end

    methods

        function net = LstmValLayers2D()
        end


        function net = Create(net)

            sLayers = [
                sequenceInputLayer(net.x_in)
                lstmLayer(net.k_hid1)%, 'OutputMode','last')
                lstmLayer(net.k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(net.y_out)
                regressionLayer
            ];

            net.lGraph = layerGraph(sLayers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end