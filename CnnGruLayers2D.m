classdef CnnGruLayers2D

    properties

    end

    methods
        function net = CnnGruLayers2D()            
        end


        function net = Create(net)

            %c_in = 1;
            f_h = 5; 
            f_n = 1;%16; 
            f_s = 1;
            p_s = 1;
    
            clayers = [
                imageInputLayer([net.x_in net.t_in 1],'Normalization','none','Name','Input')
                convolution2dLayer([net.x_in f_h], f_n, 'Stride',[net.x_in f_s], 'DilationFactor',[1 p_s], 'Padding','same', 'PaddingValue','replicate', 'Name','Conv1')
                %maxPooling2dLayer([1 p_s], 'Stride', [1 p_s],'Name','Pool1')
                flattenLayer('Name','FlatN')
            ];
            net.lGraph = layerGraph(clayers);


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
                concatenationLayer(1, net.y_out, 'Name', 'Concat')
                fullyConnectedLayer(net.n_out,'Name','FullC')
                regressionLayer
            ];
            net.lGraph = addLayers(net.lGraph, rlayers);

            net.lGraph = connectLayers(net.lGraph, 'Gru12', 'Concat/in1');
            net.lGraph = connectLayers(net.lGraph, 'Gru22', 'Concat/in2');
            net.lGraph = connectLayers(net.lGraph, 'Gru32', 'Concat/in3');


            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);
        end


    end

end
