classdef SeqCnnSpecTLayers2D

    properties

    end

    methods
        function net = SeqCnnSpecTLayers2D()            
        end


        function net = Create(net)


            f_h = 1; 
            f_n = 4; 
            f_s = 1;
            p_s = 1;
    
            layers = [
                %imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                sequenceInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1')
                %flattenLayer('Name','Flat')
                %fullyConnectedLayer(net.t_in,'Name','Full') %t_in
                %tanhLayer('Name','Tanh')
            ];

            net.lGraph = layerGraph(layers);


            %c_in = 1;
            f_h = 1;
            f_w = 1;
            f_n = 4; 
            f_s = 1;
            p_s = 1;
    
            alayers = [
                %imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                %sequenceInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([f_w f_h], f_n, 'Stride',[f_w f_s], 'DilationFactor',[1 p_s], 'Name','Conv1a')%, 'Padding','same', 'PaddingValue','replicate')
                %flattenLayer('Name','Flata')
            ];

            net.lGraph = addLayers(net.lGraph, alayers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1', 'Conv1a');

            f_h = 5;

            blayers = [
                convolution2dLayer([f_w f_h], f_n, 'Stride',[f_w f_s], 'DilationFactor',[1 p_s], 'Name','Conv1b')%,'Padding','same', 'PaddingValue','replicate')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, blayers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1', 'Conv1b');


            f_h = 13;

            clayers = [
                convolution2dLayer([f_w f_h], f_n, 'Stride',[f_w f_s], 'DilationFactor',[1 p_s], 'Name','Conv1c')%, 'Padding','same', 'PaddingValue','replicate')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, clayers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1', 'Conv1c');
            
            
            f_h = 31;

            dlayers = [
                convolution2dLayer([f_w f_h], f_n, 'Stride',[f_w f_s], 'DilationFactor',[1 p_s], 'Name','Conv1d')%, 'Padding','same', 'PaddingValue','replicate')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, dlayers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1', 'Conv1d');


            f_h = 63;

            elayers = [
                convolution2dLayer([f_w f_h], f_n, 'Stride',[f_w f_s], 'DilationFactor',[1 p_s], 'Name','Conv1e')%,, 'Padding','same', 'PaddingValue','replicate')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, elayers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1', 'Conv1e');



            f_h = 3;
            cl_n = 5;
            f_n2 = 16;

            s2Layers = [
                concatenationLayer(2, cl_n, 'Name', 'Concat')
                %convolution2dLayer([cl_n*f_n f_h], f_n2, 'Stride',[cl_n*f_n f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv2')
                flattenLayer('Name','FlatN')
                fullyConnectedLayer(net.n_out,'Name','FullN') %t_in
                %tanhLayer('Name','TanhN')
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
            net.lGraph = connectLayers(net.lGraph, 'Conv1c', 'Concat/in3');
            net.lGraph = connectLayers(net.lGraph, 'Conv1d', 'Concat/in4');
            net.lGraph = connectLayers(net.lGraph, 'Conv1e', 'Concat/in5');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end
