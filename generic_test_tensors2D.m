function [X2, Xc2, Xr2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, k_tob] = generic_test_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
    %% Test regression ANN
    if(k_tob == 0)
        k_tob = floor(l_sess/t_out); %ceil
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
    Yhs2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Bti = zeros([2, x_in, k_tob, t_sess-sess_off]);
    Bto = zeros([2, y_out, k_tob, t_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

            Mw = M(idx:idx+t_in-1, 1:x_in);
            [Bti(1,:,j,i), Bti(2,:,j,i)] = bounds(Mw,1);

            Mx = reshape( Mw', [m_in,1] );
            X2(1:m_in, j, i) = Mx(:);
            Xc2(1:m_in, 1, 1, j, i) = Mx(:);
            Xr2(2:m_in+1, j, i) = Mx(:);

            Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
            My = reshape( Myw', [n_out,1] );
            Yh2(:, j, i) = My(:);

            [Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);

            My = reshape( Myw', [n_out,1] );
            Yhs2(:, j, i) = My(:);
        end

        if(norm_fli)
             for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Mw = M(idx:idx+t_in-1, 1:x_in);
                % bounds over session
                MinSessi = min( Bi(1,:,:,i), [], 3); 
                MaxSessi = max( Bi(2,:,:,i), [], 3);

                Mw = generic_minmax_scale2D(Mw, MinSessi, MaxSessi);
            
                Mx = reshape( Mw', [m_in,1] );
                X2(1:m_in, j, i) = Mx(:);
                Xc2(1:m_in, 1, 1, j, i) = Mx(:);
                Xr2(2:m_in+1, j, i) = Mx(:);
             end
        end
        
        if(norm_flo)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Myw = M(idx+t_in:idx+t_in+t_out-1, x_in+1:x_in+y_out);
                % bounds over session
                MinSesso = min( Bo(1,:,:,i), [], 3); 
                MaxSesso = max( Bo(2,:,:,i), [], 3);

                Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

                My = reshape( Myw', [n_out,1] );
                Yhs2(:, j, i) = My(:);
            end
        end

    end
end