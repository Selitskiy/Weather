function [X2, Xc2, Xr2, Y2s, Y2, Yh2, Bi, Bo, k_tob] = generic_test_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, k_tob)
    %% Test regression ANN
    if(k_tob == 0)
        k_tob = ceil(l_sess/t_out);
    end

    m_in = x_in * t_in;
    n_out = y_out * t_out;
    n_in = y_out * t_in;

    X2 = zeros([m_in, k_tob, t_sess-sess_off]);
    Xc2 = zeros([m_in, 1, 1, k_tob, t_sess-sess_off]);
    Xr2 = ones([m_in+1, k_tob, t_sess]);
    Y2s = zeros([n_in, k_tob, t_sess-sess_off]);
    Y2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Yh2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Bi = zeros([2, x_in, k_tob, t_sess-sess_off]);
    Bo = zeros([2, y_out, k_tob, t_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset;

            Mw = M(idx:idx+t_in-1, 1:x_in);
            [Bi(1,:,j,i), Bi(2,:,j,i)] = bounds(Mw,1);
            if(norm_fl)
                Mw = generic_minmax_scale2D(Mw, Bi(1,:,j,i), Bi(2,:,j,i));
            end
            Mx = reshape( Mw', [m_in,1] );
            X2(1:m_in, j, i) = Mx(:);
            Xc2(1:m_in, 1, 1, j, i) = Mx(:);
            Xr2(2:m_in+1, j, i) = Mx(:);

            Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
            [Bo(1,:,j,i), Bo(2,:,j,i)] = bounds(Myw,1);
            %My = reshape( M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out)', [n_out,1] );
            %if(norm_fl)
            %    Myw = generic_minmax_scale2D(Myw, Bo(1,:,j,i), Bo(2,:, j,i));
            %end
            My = reshape( Myw', [n_out,1] );
            Yh2(:, j, i) = My(:);
        end
    end
end