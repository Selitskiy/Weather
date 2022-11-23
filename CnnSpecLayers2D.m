classdef CnnSpecLayers2D

    properties

    end

    methods
        function net = CnnSpecLayers2D()            
        end


        function net = Create(net)

            %c_in = 1;
            f_h = 5; 
            f_n = 16; 
            f_s = 1;
            p_s = 1;
    
            alayers = [
                imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1a')
                %flattenLayer('Name','Flata')
            ];

            net.lGraph = layerGraph(alayers);


            f_h = 11;

            blayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1b')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, blayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1b');


            f_h = 11;
            f_n = 1; 
            cl_n = 2;

            s2Layers = [
                %depthConcatenationLayer(cl_n, 'Name', 'Concat')
                concatenationLayer(1, cl_n, 'Name', 'Concat')
                convolution2dLayer([cl_n f_h], f_n, 'Stride',[cl_n f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv2')
                %sequenceUnfoldingLayer('Name','Unfold')
                flattenLayer('Name','Flat2')
                %concatenationLayer(1, 2, 'Name', 'Concat')
                gruLayer(net.k_hid1,'Name','Gru1')
                %fullyConnectedLayer(net.k_hid1,'Name','Full1')
                %reluLayer('Name','Relu1')
                gruLayer(net.k_hid2,'Name','Gru2')
                %fullyConnectedLayer(net.k_hid2,'Name','Full2')
                %reluLayer('Name','Relu2')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, s2Layers);

            net.lGraph = connectLayers(net.lGraph, 'Conv1a', 'Concat/in1');
            net.lGraph = connectLayers(net.lGraph, 'Conv1b', 'Concat/in2');
            %net.lGraph = connectLayers(net.lGraph, 'Flata', 'Concat/in1');
            %net.lGraph = connectLayers(net.lGraph, 'Flatb', 'Concat/in2');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end