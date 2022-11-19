function [X, Y, Bi, Bo, Sx, Sy, n_xy, k_ob] = generic_train_seq_tensors2D(M, x_off, x_in, t_in, y_off, y_out, t_out, l_sess, n_sess, norm_fli, norm_flo)


    % Number of observations in a session (training label(sequence) does
    % not touch test period    
    k_ob = l_sess - 1;

    %n_xy = x_in + y_out; 
    %remove possible overlap of x_in and y_out (f.e. for autoregression)
    x_over = (x_off+x_in)-y_off;
    if(x_over < 0)
        x_over = 0;
    end
    n_xy = x_in-x_over + y_out;


    X = zeros([x_in, k_ob, n_sess]);
    Y = zeros([n_xy, k_ob, n_sess]);
    Bi = zeros([4, x_in, n_sess]);
    Bo = zeros([4, n_xy, n_sess]);

    %Segment boundaries
    Sx = zeros([2, n_sess]);
    Sy = zeros([2, n_sess]);


    % Re-format input into session tensor
    for i = 1:n_sess
        % scale bounds over session scale
        idx = (i-1)*l_sess + 1;

        st_idx = idx;
        end_idx = idx+k_ob-1;
        Sx(1,i) = st_idx;
        Sx(2,i) = end_idx;

        Mxw = M(st_idx:end_idx, x_off+1:x_off+x_in);
        % scale bounds over observation span
        [Bi(1,:,i), Bi(2,:,i)] = bounds(Mxw,1);
        Bi(3,:,i) = mean(Mxw,1);
        Bi(4,:,i) = std(Mxw,0,1);

        Myw = M(st_idx:end_idx, x_off+1:x_off+n_xy);
        % scale bounds over observation span
        [Bo(1,:,i), Bo(2,:,i)] = bounds(Myw,1);
        Bo(3,:,i) = mean(Myw,1);
        Bo(4,:,i) = std(Myw,0,1);

        %Mx = reshape( Mxw', [m_in,1] );
        X(:, :, i) = Mxw';


        st_idx = idx+1;
        end_idx = idx+k_ob;
        Sy(1,i) = st_idx;
        Sy(2,i) = end_idx;

        Myw = M(st_idx:end_idx, x_off+1:x_off+n_xy);
        % scale bounds over observation span
        %[Bo(1,:,i), Bo(2,:,i)] = bounds(Myw,1);
        %Bo(3,:,i) = mean(Myw,1);
        %Bo(4,:,i) = std(Myw,0,1);

        %Mx = reshape( Mxw', [m_in,1] );
        Y(:, :, i) = Myw';
    end


    if(norm_fli)
        for i = 1:n_sess
            % scale bounds over session scale
            idx = (i-1)*l_sess + 1;

            st_idx = idx;
            end_idx = idx+k_ob-1;
            
            Mxw = M(st_idx:end_idx, x_off+1:x_off+x_in);
            % bounds over session
            %MinSessi = Bi(1,:,i); 
            %MaxSessi = Bi(2,:,i);
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

            st_idx = idx+1;
            end_idx = idx+k_ob;

            Myw = M(st_idx:end_idx, x_off+1:x_off+n_xy);
            % bounds over session
            %MinSesso = Bo(1,:,i); 
            %MaxSesso = Bo(2,:,i);
            MeanSesso = Bo(3,:,i);
            StdSesso = Bo(4,:,i);
    
            Myw = generic_mean_std_scale2D(Myw, MeanSesso, StdSesso);
            %Myw = generic_mean_minmax_scale2D(Myw, MeanSesso, MinSesso, MaxSesso);
            %Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

            Y(:, :, i) = Myw';
        end
    end

end