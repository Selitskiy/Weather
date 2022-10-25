function [X, Xc, Xr, Ys, Y, Bi, Bo, XI, C, k_ob] = generic_train_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, n_sess, norm_fli, norm_flo)

    % Number of observations in a session
    k_ob = l_sess - t_in + 1;

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
    Bi = zeros([2, x_in, k_ob, n_sess]);
    Bo = zeros([2, y_out, k_ob, n_sess]);
    Bos = zeros([2, y_out, k_ob, n_sess]);

    k_iob = k_ob * n_sess;
    XI = zeros([m_in, k_iob]);
    I = zeros([k_iob, 1]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;
            
            Mxw = M(idx:idx+t_in-1, 1:x_in);
            % scale bounds over observation span
            [Bi(1,:,j,i), Bi(2,:,j,i)] = bounds(Mxw,1);

            Mx = reshape( Mxw', [m_in,1] );
            X(1:m_in, j, i) = Mx(:);
            Xc(1:m_in, 1, 1, j, i) = Mx(:);
            Xr(2:m_in+1, j, i) = Mx(:);


            Myw = M(idx+1:idx+t_in, x_in+1:x_in+y_out);
            [Bos(1,:,j,i), Bos(2,:,j,i)] = bounds(Myw,1);

            My = reshape( Myw', [n_in,1] );
            Ys(:, j, i) = My(:);

            Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
            [Bo(1,:,j,i), Bo(2,:,j,i)] = bounds(Myw,1);

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
                MinSessi = min( Bi(1,:,:,i), [], 3); 
                MaxSessi = max( Bi(2,:,:,i), [], 3);

                Mxw = generic_minmax_scale2D(Mxw, MinSessi, MaxSessi);

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
                MinSessos = min( Bos(1,:,:,i), [], 3); 
                MaxSessos = max( Bos(2,:,:,i), [], 3);

                Myw = generic_minmax_scale2D(Myw, MinSessos, MaxSessos);

                My = reshape( Myw', [n_in,1] );
                Ys(:, j, i) = My(:);


                Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
                % bounds over session
                MinSesso = min( Bo(1,:,:,i), [], 3); 
                MaxSesso = max( Bo(2,:,:,i), [], 3);

                Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

                My = reshape( Myw', [n_out,1] );
                Y(:, j, i) = My(:);
            end
        end
    end

    C = categorical(I);
end