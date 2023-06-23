classdef CnnSpec2Layers2D

    properties

    end

    methods
        function net = CnnSpec2Layers2D()            
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
                flattenLayer('Name','Flata')
                fullyConnectedLayer(net.n_out,'Name','Fulla')
                tanhLayer('Name','Tanha')
                gruLayer(net.k_hid1,'Name','Gru1a')
            ];

            net.lGraph = layerGraph(alayers);


            f_h = 13;

            blayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1b')
                flattenLayer('Name','Flatb')
                fullyConnectedLayer(net.n_out,'Name','Fullb')
                tanhLayer('Name','Tanhb')
                gruLayer(net.k_hid1,'Name','Gru1b')
            ];

            net.lGraph = addLayers(net.lGraph, blayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1b');


            f_h = 31;

            clayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1c')
                flattenLayer('Name','Flatc')
                fullyConnectedLayer(net.n_out,'Name','Fullc')
                tanhLayer('Name','Tanhc')
                gruLayer(net.k_hid1,'Name','Gru1c')
            ];

            net.lGraph = addLayers(net.lGraph, clayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1c');
            
            
            f_h = 63;

            dlayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1d')
                flattenLayer('Name','Flatd')
                fullyConnectedLayer(net.n_out,'Name','Fulld')
                tanhLayer('Name','Tanhd')
                gruLayer(net.k_hid1,'Name','Gru1d')
            ];

            net.lGraph = addLayers(net.lGraph, dlayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1d');


            f_h = 129;

            elayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1e')
                flattenLayer('Name','Flate')
                fullyConnectedLayer(net.n_out,'Name','Fulle')
                tanhLayer('Name','Tanhe')
                gruLayer(net.k_hid1,'Name','Gru1e')
            ];

            net.lGraph = addLayers(net.lGraph, elayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1e');



            f_h = 3;
            f_n = 1; 
            cl_n = 5;

            s2Layers = [
                %additionLayer(cl_n,'Name','Add')
                %%depthConcatenationLayer(cl_n, 'Name', 'Concat')
                concatenationLayer(1, cl_n, 'Name', 'Concat')
                fullyConnectedLayer(net.n_out,'Name','FullN') %t_in
                tanhLayer('Name','TanhN')
                %convolution2dLayer([cl_n f_h], f_n, 'Stride',[cl_n f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv2')
                %%sequenceUnfoldingLayer('Name','Unfold')
                %flattenLayer('Name','Flat2')
                %%concatenationLayer(1, 2, 'Name', 'Concat')
                %gruLayer(net.k_hid1,'Name','Gru1')
                %%fullyConnectedLayer(net.k_hid1,'Name','Full1')
                %%reluLayer('Name','Relu1')
                gruLayer(net.k_hid2,'Name','Gru2')
                %%fullyConnectedLayer(net.k_hid2,'Name','Full2')
                %%reluLayer('Name','Relu2')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, s2Layers);

            net.lGraph = connectLayers(net.lGraph, 'Gru1a', 'Concat/in1');
            net.lGraph = connectLayers(net.lGraph, 'Gru1b', 'Concat/in2');
            net.lGraph = connectLayers(net.lGraph, 'Gru1c', 'Concat/in3');
            net.lGraph = connectLayers(net.lGraph, 'Gru1d', 'Concat/in4');
            net.lGraph = connectLayers(net.lGraph, 'Gru1e', 'Concat/in5');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end