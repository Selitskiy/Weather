classdef GruVal3Layers2D

    properties
    end

    methods

        function net = GruVal3Layers2D()
        end


        function net = Create(net)

            s1Layers = [
                sequenceInputLayer(net.x_in, 'Name', 'Input')
                gruLayer(net.k_hid1)%, 'OutputMode','last')
                gruLayer(net.k_hid2)%, 'OutputMode','last')
                fullyConnectedLayer(1,'Name', 'Full1')
            ];
            net.lGraph = layerGraph(s1Layers);


            s2Layers = [
                gruLayer(net.k_hid1,'Name', 'Gru21')
                gruLayer(net.k_hid2,'Name', 'Gru22')
                fullyConnectedLayer(1,'Name', 'Full2')
            ];
            net.lGraph = addLayers(net.lGraph, s2Layers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Gru21');


            s3Layers = [
                gruLayer(net.k_hid1,'Name', 'Gru31')
                gruLayer(net.k_hid2,'Name', 'Gru32')
                fullyConnectedLayer(1,'Name', 'Full3')
            ];
            net.lGraph = addLayers(net.lGraph, s3Layers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Gru31');


            rLayers = [
                concatenationLayer(1, 3, 'Name', 'Concat')
                fullyConnectedLayer(net.y_out)
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, rLayers);
            net.lGraph = connectLayers(net.lGraph, 'Full1', 'Concat/in1');
            net.lGraph = connectLayers(net.lGraph, 'Full2', 'Concat/in2');
            net.lGraph = connectLayers(net.lGraph, 'Full3', 'Concat/in3');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end