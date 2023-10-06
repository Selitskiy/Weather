classdef BaseNetCl < BaseNet2D

    properties
        mb_size = [];
    end

    methods
        function net = BaseNetCl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch)

            net = net@BaseNet2D(x_off, x_in, t_in, 0,0,0, ini_rate, max_epoch);

            net.n_out = n_sess;

            mult = 1;
            net.k_hid1 = floor(mult * net.m_in);
            net.k_hid2 = floor(mult * (2*net.m_in + 1));
            %net.ini_rate = ini_rate;
            %net.max_epoch = max_epoch;

        end



    end
end