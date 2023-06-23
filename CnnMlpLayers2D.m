classdef CnnMlpLayers2D

    properties

    end

    methods
        function net = CnnMlpLayers2D()            
        end


        function net = Create(net)

            %c_in = 1;
            f_h = 3; 
            f_n = 1;%16; 
            f_s = 1;
            p_s = 1;
    
            layers = [
                imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1')
                %maxPooling2dLayer([1 p_s], 'Stride', [1 p_s],'Name','Pool1')
                flattenLayer('Name','FlatN')
                fullyConnectedLayer(floor(net.t_in/p_s),'Name','FullN')
                tanhLayer('Name','TanhN')
                %gruLayer(net.k_hid1,'Name','Gru1')
                fullyConnectedLayer(net.k_hid1,'Name','Full1')
                reluLayer('Name','Relu1')
                %gruLayer(net.k_hid2,'Name','Gru2')
                fullyConnectedLayer(net.k_hid2,'Name','Full2')
                reluLayer('Name','Relu2')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];

            net.lGraph = layerGraph(layers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end
