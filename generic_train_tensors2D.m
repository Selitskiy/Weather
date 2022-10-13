function [X, Xc, Xr, Ys, Y, B, XI, C, k_ob] = generic_train_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, n_sess, norm_fl)

    % Number of observations in a session
    k_ob = l_sess - t_in + 1;

    m_in = x_in * t_in;
    n_out = y_out * t_out;

    % Re-format input into session tensor
    % ('ones' (not 'zeros') for X are for bias 'trick'
    X = zeros([m_in, k_ob, n_sess]);
    Xc = zeros([m_in, 1, 1, k_ob, n_sess]);
    Xr = ones([m_in+1, k_ob, n_sess]);
    Ys = zeros([n_out, k_ob, n_sess]);
    Y = zeros([n_out, k_ob, n_sess]);
    B = zeros([2, k_ob, n_sess]);

    k_iob = k_ob * n_sess;
    XI = zeros([m_in, k_iob]);
    I = zeros([k_iob, 1]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;
            
            Mx = reshape( M(idx:idx+t_in-1, 1:x_in), [m_in,1] );
            % scale bounds over observation span
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series_generic_minmax_scale(Mx, B(1,j,i), B(2,j,i));
            end
            X(1:m_in, j, i) = Mx(:);
            Xc(1:m_in, 1, 1, j, i) = Mx(:);
            Xr(2:m_in+1, j, i) = Mx(:);

            My = reshape( M(idx+1:idx+t_in, x_in+1:x_in+y_out), [n_out,1] );
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
            end
            Ys(:, j, i) = My(:);

            My = reshape( M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out), [n_out,1] );
            if(norm_fl)
                My = w_series_generic_minmax_scale(My, B(1,j,i), B(2,j,i));
            end
            Y(:, j, i) = My(:);


            i_idx = (i-1)*k_ob + j;
            XI(1:m_in, i_idx) = Mx(:);
            I(i_idx) = i;
        end
    end

    C = categorical(I);
end