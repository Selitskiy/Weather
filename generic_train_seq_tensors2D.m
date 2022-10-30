function [X, Y, Bi, Bo, k_ob] = generic_train_seq_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, n_sess, norm_fli, norm_flo)


    % Number of observations in a session (training label(sequence) does
    % not touch test period    
    k_ob = l_sess - 1;


    n_xy = x_in + y_out; 

    X = zeros([n_xy, k_ob, n_sess]);
    Y = zeros([n_xy, k_ob, n_sess]);
    Bi = zeros([4, n_xy, n_sess]);
    Bo = zeros([4, n_xy, n_sess]);

    % Re-format input into session tensor
    for i = 1:n_sess
        % scale bounds over session scale
        idx = (i-1)*l_sess + 1;

            
        Mxw = M(idx:idx+k_ob-1, 1:n_xy);
        % scale bounds over observation span
        [Bi(1,:,i), Bi(2,:,i)] = bounds(Mxw,1);
        Bi(3,:,i) = mean(Mxw,1);
        Bi(4,:,i) = std(Mxw,1);

        %Mx = reshape( Mxw', [m_in,1] );
        X(:, :, i) = Mxw';

        Myw = M(idx+1:idx+k_ob, 1:n_xy);
        % scale bounds over observation span
        [Bo(1,:,i), Bo(2,:,i)] = bounds(Myw,1);
        Bo(3,:,i) = mean(Myw,1);
        Bo(4,:,i) = std(Myw,1);

        %Mx = reshape( Mxw', [m_in,1] );
        Y(:, :, i) = Myw';
    end


    if(norm_fli)
        for i = 1:n_sess
            % scale bounds over session scale
            idx = (i-1)*l_sess + 1;

            
            Mxw = M(idx:idx+k_ob-1, 1:n_xy);
            % bounds over session
            MinSessi = Bi(1,:,i); 
            MaxSessi = Bi(2,:,i);
            MeanSessi = Bi(3,:,i);
            StdSessi = Bi(4,:,i);
    
            Mxw = generic_mean_std_scale2D(Mxw, MeanSessi, StdSessi);
            %Mxw = generic_mean_minmax_scale2D(Mxw, MeanSessi, MinSessi, MaxSessi);
            %Mxw = generic_minmax_scale2D(Mxw, MinSessi, MaxSessi);

            X(:, :, i) = Mxw';
        end
    end

    if(norm_flo)
        for i = 1:n_sess
            % scale bounds over session scale
            idx = (i-1)*l_sess + 1;

            Myw = M(idx+1:idx+k_ob, 1:n_xy);
            % bounds over session
            MinSesso = Bo(1,:,i); 
            MaxSesso = Bo(2,:,i);
            MeanSesso = Bo(3,:,i);
            StdSesso = Bo(4,:,i);
    
            Myw = generic_mean_std_scale2D(Myw, MeanSesso, StdSesso);
            %Myw = generic_mean_minmax_scale2D(Myw, MeanSesso, MinSesso, MaxSesso);
            %Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

            Y(:, :, i) = Myw';
        end
    end

end