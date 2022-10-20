function [X, Xc, Xr, Ys, Y, Bi, Bo, XI, C, k_ob] = generic_train_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, n_sess, norm_fl)

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
            if(norm_fl)
                Mxw = generic_minmax_scale2D(Mxw, Bi(1,:,j,i), Bi(2,:,j,i));
            end
            Mx = reshape( Mxw', [m_in,1] );
            X(1:m_in, j, i) = Mx(:);
            Xc(1:m_in, 1, 1, j, i) = Mx(:);
            Xr(2:m_in+1, j, i) = Mx(:);

            Myw = M(idx+1:idx+t_in, x_in+1:x_in+y_out);
            [Bo(1,:,j,i), Bo(2,:,j,i)] = bounds(Myw,1);
            %My = reshape( M(idx+1:idx+t_in, x_in+1:x_in+y_out)', [n_in,1] );
            if(norm_fl)
                Myw = generic_minmax_scale2D(Myw, Bo(1,:,j,i), Bo(2,:,j,i));
            end
            My = reshape( Myw', [n_in,1] );
            Ys(:, j, i) = My(:);

            Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
            [Bo(1,:,j,i), Bo(2,:,j,i)] = bounds(Myw,1);
            %My = reshape( M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out)', [n_out,1] );
            if(norm_fl)
                Myw = generic_minmax_scale2D(Myw, Bo(1,:,j,i), Bo(2,:,j,i));
            end
            My = reshape( Myw', [n_out,1] );
            Y(:, j, i) = My(:);


            i_idx = (i-1)*k_ob + j;
            XI(1:m_in, i_idx) = Mx(:);
            I(i_idx) = i;
        end
    end

    C = categorical(I);
end