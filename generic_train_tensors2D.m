function [X, Xc, Xr, Ys, Y, Bi, Bo, XI, C, Sx, Sys, Sy, k_ob] = generic_train_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, n_sess, norm_fli, norm_flo)

    % Number of observations in a session
    k_ob = l_sess - t_in + 1 - t_in - t_out;

    m_in = x_in * t_in;
    n_out = y_out * t_out;
    n_in = y_out * t_in;

    % Re-format input into session tensor
    % ('ones' (not 'zeros') for X are for bias 'trick'
    X = zeros([m_in, k_ob, n_sess]);
    Xc = zeros([m_in, 1, 1, k_ob, n_sess]);
    Xr = ones([m_in+1, k_ob, n_sess]);
    Ys = zeros([n_in, k_ob, n_sess]);
    Y = zeros([n_out, k_ob, n_sess]);
    Bi = zeros([4, x_in, k_ob, n_sess]);
    Bo = zeros([4, y_out, k_ob, n_sess]);
    Bos = zeros([4, y_out, k_ob, n_sess]);

    %Segment boundaries
    Sx = zeros([2, k_ob, n_sess]);
    Sys = zeros([2, k_ob, n_sess]);
    Sy = zeros([2, k_ob, n_sess]);


    k_iob = k_ob * n_sess;
    XI = zeros([m_in, k_iob]);
    I = zeros([k_iob, 1]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;

            st_idx = idx;
            end_idx = idx+t_in-1;
            Sx(1,j,i) = st_idx;
            Sx(2,j,i) = end_idx;

            Mxw = M(st_idx:end_idx, 1:x_in);
            % scale bounds over observation span
            [Bi(1,:,j,i), Bi(2,:,j,i)] = bounds(Mxw,1);
            Bi(3,:,j,i) = mean(Mxw,1);
            Bi(4,:,j,i) = std(Mxw,1);

            Mx = reshape( Mxw', [m_in,1] );
            X(1:m_in, j, i) = Mx(:);
            Xc(1:m_in, 1, 1, j, i) = Mx(:);
            Xr(2:m_in+1, j, i) = Mx(:);


            st_idx = idx+1;
            end_idx = idx+t_in;
            Sys(1,j,i) = st_idx;
            Sys(2,j,i) = end_idx;

            Myw = M(st_idx:end_idx, x_in+1:x_in+y_out);
            [Bos(1,:,j,i), Bos(2,:,j,i)] = bounds(Myw,1);
            Bos(3,:,j,i) = mean(Myw,1);
            Bos(4,:,j,i) = std(Myw,1);

            My = reshape( Myw', [n_in,1] );
            Ys(:, j, i) = My(:);


            st_idx = idx+t_in;
            end_idx = idx+t_in+t_out-1;
            Sy(1,j,i) = st_idx;
            Sy(2,j,i) = end_idx;

            Myw = M(st_idx:end_idx, x_in+1:x_in+y_out);
            [Bo(1,:,j,i), Bo(2,:,j,i)] = bounds(Myw,1);
            Bo(3,:,j,i) = mean(Myw,1);
            Bo(4,:,j,i) = std(Myw,1);

            My = reshape( Myw', [n_out,1] );
            Y(:, j, i) = My(:);


            i_idx = (i-1)*k_ob + j;
            XI(1:m_in, i_idx) = Mx(:);
            I(i_idx) = i;
        end

        if(norm_fli)
            for j = 1:k_ob
                % extract and scale observation sequence
                idx = (i-1)*l_sess + j;
            
                Mxw = M(idx:idx+t_in-1, 1:x_in);
                % bounds over session
                %MinSessi = min( Bi(1,:,:,i), [], 3); 
                %MaxSessi = max( Bi(2,:,:,i), [], 3);
                MeanSessi = mean( Bi(3,:,:,i), 3);
                StdSessi = mean( Bi(4,:,:,i), 3);

                %Mxw = generic_mean_minmax_scale2D(Mxw, MeanSessi, MinSessi, MaxSessi);
                Mxw = generic_mean_std_scale2D(Mxw, MeanSessi, StdSessi);

                Mx = reshape( Mxw', [m_in,1] );
                X(1:m_in, j, i) = Mx(:);
                Xc(1:m_in, 1, 1, j, i) = Mx(:);
                Xr(2:m_in+1, j, i) = Mx(:);


                i_idx = (i-1)*k_ob + j;
                XI(1:m_in, i_idx) = Mx(:);
                I(i_idx) = i;
            end
        end

        if(norm_flo)
            for j = 1:k_ob
                % extract and scale observation sequence
                idx = (i-1)*l_sess + j;
            
                Myw = M(idx+1:idx+t_in, x_in+1:x_in+y_out);
                % bounds over session
                %MinSessos = min( Bos(1,:,:,i), [], 3); 
                %MaxSessos = max( Bos(2,:,:,i), [], 3);
                MeanSessos = mean( Bos(3,:,:,i), 3);
                StdSessos = mean( Bos(4,:,:,i), 3);

                Myw = generic_mean_std_scale2D(Myw, MeanSessos, StdSessos);
                %Myw = generic_mean_minmax_scale2D(Myw, MeanSessos, MinSessos, MaxSessos);
                %Myw = generic_minmax_scale2D(Myw, MinSessos, MaxSessos);

                My = reshape( Myw', [n_in,1] );
                Ys(:, j, i) = My(:);


                Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
                % bounds over session
                %MinSesso = min( Bo(1,:,:,i), [], 3); 
                %MaxSesso = max( Bo(2,:,:,i), [], 3);
                MeanSesso = mean( Bo(3,:,:,i), 3);
                StdSesso = mean( Bo(4,:,:,i), 3);

                Myw = generic_mean_std_scale2D(Myw, MeanSesso, StdSesso);
                %Myw = generic_mean_minmax_scale2D(Myw, MeanSesso, MinSesso, MaxSesso);
                %Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

                My = reshape( Myw', [n_out,1] );
                Y(:, j, i) = My(:);
            end
        end
    end

    C = categorical(I);
end