classdef SeqCnnLayers2D

    properties

    end

    methods
        function net = SeqCnnLayers2D()            
        end


        function net = Create(net)

            %c_in = 1;
            f_h = 1; 
            f_n = 4; 
            f_s = 1;
            p_s = 1;
    
            alayers = [
                %imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                sequenceInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1a')
                %flattenLayer('Name','Flata')
            ];

            net.lGraph = layerGraph(alayers);


            f_h = 5;
            cl_n = 5;
            f_n2 = 16;

            s2Layers = [
                %concatenationLayer(1, cl_n, 'Name', 'Concat')
                %convolution2dLayer([cl_n*f_n f_h], f_n2, 'Stride',[cl_n*f_n f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv2')
                flattenLayer('Name','FlatN')
                fullyConnectedLayer(net.n_out,'Name','FullN') %t_in
                %tanhLayer('Name','TanhN')
                sigmoidLayer('Name','SigN')
                gruLayer(net.k_hid1,'Name','Gru1')
                gruLayer(net.k_hid2,'Name','Gru2')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, s2Layers);
            net.lGraph = connectLayers(net.lGraph, 'Conv1a', 'FlatN');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end
