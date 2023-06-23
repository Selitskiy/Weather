classdef CnnGruSpecLayers2D

    properties

    end

    methods
        function net = CnnGruSpecLayers2D()            
        end


        function net = Create(net)


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


            f_h = 13;

            blayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1b')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, blayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1b');


            f_h = 31;

            clayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1c')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, clayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1c');
            
            
            f_h = 63;

            dlayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1d')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, dlayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1d');


            f_h = 129;

            elayers = [
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1e')
                %flattenLayer('Name','Flatb')
            ];

            net.lGraph = addLayers(net.lGraph, elayers);
            net.lGraph = connectLayers(net.lGraph, 'Input', 'Conv1e');



            f_h = 3;
            f_n = 16; 
            cl_n = 5;

            s2Layers = [
                %depthConcatenationLayer(cl_n, 'Name', 'Concat')
                concatenationLayer(1, cl_n, 'Name', 'ConcatC')
                convolution2dLayer([cl_n f_h], f_n, 'Stride',[cl_n f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv2')
                %sequenceUnfoldingLayer('Name','Unfold')
                flattenLayer('Name','FlatN')
            ];
            net.lGraph = addLayers(net.lGraph, s2Layers);

            net.lGraph = connectLayers(net.lGraph, 'Conv1a', 'ConcatC/in1');
            net.lGraph = connectLayers(net.lGraph, 'Conv1b', 'ConcatC/in2');
            net.lGraph = connectLayers(net.lGraph, 'Conv1c', 'ConcatC/in3');
            net.lGraph = connectLayers(net.lGraph, 'Conv1d', 'ConcatC/in4');
            net.lGraph = connectLayers(net.lGraph, 'Conv1e', 'ConcatC/in5');


            g1layers = [
                fullyConnectedLayer(net.t_in,'Name','FullN1')
                %tanhLayer('Name','TanhN1')
                gruLayer(net.k_hid1,'Name','Gru11')
                %fullyConnectedLayer(net.k_hid1,'Name','Full1')
                %reluLayer('Name','Relu1')
                gruLayer(net.k_hid2,'Name','Gru12')
                %fullyConnectedLayer(net.k_hid2,'Name','Full2')
                %reluLayer('Name','Relu2')
            ];
            net.lGraph = addLayers(net.lGraph, g1layers);
            net.lGraph = connectLayers(net.lGraph, 'FlatN', 'FullN1');


             g2layers = [
                fullyConnectedLayer(net.t_in,'Name','FullN2')
                %tanhLayer('Name','TanhN1')
                gruLayer(net.k_hid1,'Name','Gru21')
                %fullyConnectedLayer(net.k_hid1,'Name','Full1')
                %reluLayer('Name','Relu1')
                gruLayer(net.k_hid2,'Name','Gru22')
                %fullyConnectedLayer(net.k_hid2,'Name','Full2')
                %reluLayer('Name','Relu2')
            ];
            net.lGraph = addLayers(net.lGraph, g2layers);
            net.lGraph = connectLayers(net.lGraph, 'FlatN', 'FullN2');           

             g3layers = [
                fullyConnectedLayer(net.t_in,'Name','FullN3')
                %tanhLayer('Name','TanhN1')
                gruLayer(net.k_hid1,'Name','Gru31')
                %fullyConnectedLayer(net.k_hid1,'Name','Full1')
                %reluLayer('Name','Relu1')
                gruLayer(net.k_hid2,'Name','Gru32')
                %fullyConnectedLayer(net.k_hid2,'Name','Full2')
                %reluLayer('Name','Relu2')
            ];
            net.lGraph = addLayers(net.lGraph, g3layers);
            net.lGraph = connectLayers(net.lGraph, 'FlatN', 'FullN3');  


            rlayers = [
                concatenationLayer(1, net.y_out, 'Name', 'ConcatG')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, rlayers);

            net.lGraph = connectLayers(net.lGraph, 'Gru12', 'ConcatG/in1');
            net.lGraph = connectLayers(net.lGraph, 'Gru22', 'ConcatG/in2');
            net.lGraph = connectLayers(net.lGraph, 'Gru32', 'ConcatG/in3');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end
