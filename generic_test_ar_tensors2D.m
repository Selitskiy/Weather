function [X2, Xc2, Xr2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_ar_tensors2D(M, x_off, x_in, t_in, y_off, y_out, t_out, n_xy, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob)
    %% Test regression ANN
    if(k_tob == 0)
        [m,~] = size(M);
        k_tob = floor(l_sess/t_out); %ceil
        if(l_sess + k_tob*t_out*(t_sess-sess_off)) > m
            k_tob = k_tob - 1;
        end
    end

    m_in = x_in * t_in;
    n_out = n_xy * t_out; %y_out * t_out;
    n_in = y_out * t_in;

    X2 = zeros([m_in, k_tob, t_sess-sess_off]);
    Xc2 = zeros([m_in, 1, 1, k_tob, t_sess-sess_off]);
    Xr2 = ones([m_in+1, k_tob, t_sess]);
    Y2s = zeros([n_in, k_tob, t_sess-sess_off]);
    Y2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Yh2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Yhs2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Bti = zeros([4, x_in, k_tob, t_sess-sess_off]);
    Bto = zeros([4, n_xy, k_tob, t_sess-sess_off]);

    %Segment boundaries
    Sx2 = zeros([2, k_tob, t_sess-sess_off]);
    Sy2 = zeros([2, k_tob, t_sess-sess_off]);


    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

            st_idx = idx;
            end_idx = idx+t_in-1;
            Sx2(1,j,i) = st_idx;
            Sx2(2,j,i) = end_idx;

            Mw = M(st_idx:end_idx, x_off+1:x_off+x_in);
            [Bti(1,:,j,i), Bti(2,:,j,i)] = bounds(Mw,1);
            Bti(3,:,j,i) = mean(Mw,1);
            Bti(4,:,j,i) = std(Mw,0,1);

            Myw = M(st_idx:end_idx, x_off+1:x_off+n_xy);
            [Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);
            Bto(3,:,j,i) = mean(Myw,1);
            Bto(4,:,j,i) = std(Myw,0,1);


            Mx = reshape( Mw', [m_in,1] );
            X2(1:m_in, j, i) = Mx(:);
            Xc2(1:m_in, 1, 1, j, i) = Mx(:);
            Xr2(1:m_in, j, i) = Mx(:);


            st_idx = idx+t_in;
            end_idx = idx+t_in+t_out-1;
            Sy2(1,j,i) = st_idx;
            Sy2(2,j,i) = end_idx;

            Myw = M(st_idx:end_idx, x_off+1:x_off+n_xy);
            My = reshape( Myw', [n_out,1] );
            Yh2(:, j, i) = My(:);

            %[Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);

            My = reshape( Myw', [n_out,1] );
            Yhs2(:, j, i) = My(:);
        end

        if(norm_fli)
             for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Mw = M(idx:idx+t_in-1, x_off+1:x_off+x_in);
                % bounds over session
                MeanSessi = Bti(3,:,j,i);
                StdSessi = Bti(4,:,j,i);
                %MeanSessi = Bi(3,:,i);
                %StdSessi = Bi(4,:,i);

                Mw = generic_mean_std_scale2D(Mw, MeanSessi, StdSessi);
                %Mw = generic_mean_minmax_scale2D(Mw, MeanSessi, MinSessi, MaxSessi);
                %Mw = generic_minmax_scale2D(Mw, MinSessi, MaxSessi);
            
                Mx = reshape( Mw', [m_in,1] );
                X2(1:m_in, j, i) = Mx(:);
                Xc2(1:m_in, 1, 1, j, i) = Mx(:);
                Xr2(1:m_in, j, i) = Mx(:);
             end
        end
        
        if(norm_flo)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Myw = M(idx+t_in:idx+t_in+t_out-1, x_off+1:x_off+n_xy);
                % bounds over session
                MeanSesso = Bto(3,:,j,i);
                StdSesso = Bto(4,:,j,i);
                %MeanSesso = Bo(3,:,i);
                %StdSesso = Bo(4,:,i);

                Myw = generic_mean_std_scale2D(Myw, MeanSesso, StdSesso);
                %Myw = generic_mean_minmax_scale2D(Myw, MeanSesso, MinSesso, MaxSesso);
                %Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);

                My = reshape( Myw', [n_out,1] );
                Yhs2(:, j, i) = My(:);
            end
        end

    end
end