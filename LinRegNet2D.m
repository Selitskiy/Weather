classdef LinRegNet2D < BaseNet2D & LinRegInputNet2D

    properties

    end

    methods
        function net = LinRegNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch)

            net = net@BaseNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
            net = net@LinRegInputNet2D();

            net.name = "reg2d";

        end


        function net = Train(net, i, X, Y)
            fprintf('Training %s Reg net %d\n', net.name, i); 

            net = Train@LinRegInputNet2D(net, i, X, Y);
        end

        
    end
end