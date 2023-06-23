function [X2, Xc2, Xr2, Xs2, Ys2, Ysh2, Yshs2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob, Xp2, Xcp2, Xrp2, Xsp2] = generic_test_tensors2D(M, x_off, x_in, t_in, y_off, y_out, t_out, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob, x_pca, Vit)
    %% Test regression ANN
    if(k_tob == 0)
        [m,~] = size(M);
        k_tob = floor(l_sess/t_out); %ceil
        if(l_sess + k_tob*t_out*(t_sess-sess_off)) > m
            k_tob = k_tob - 1;
        end
    end

    m_in = x_in * t_in;
    n_out = y_out * t_out;
    %n_in = y_out * t_in;

    m_pca = m_in;
    if x_pca
        m_pca = x_pca * t_in;
    end

    X2 = zeros([m_in, k_tob, t_sess-sess_off]);
    Xc2 = zeros([x_in, t_out, 1, k_tob, t_sess-sess_off]);
    Xr2 = ones([m_in+1, k_tob, t_sess-sess_off]);
    Xs2 = zeros([x_in, t_in, k_tob, t_sess-sess_off]);
    Ys2 = zeros([y_out, t_out, k_tob, t_sess-sess_off]);
    Ysh2 = zeros([y_out, t_out, k_tob, t_sess-sess_off]);
    Yshs2 = zeros([y_out, t_out, k_tob, t_sess-sess_off]);
    %Y2s = zeros([n_in, k_tob, t_sess-sess_off]);
    Y2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Yh2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Yhs2 = zeros([n_out, k_tob, t_sess-sess_off]);
    Bti = zeros([4, x_in, k_tob, t_sess-sess_off]);
    Bto = zeros([4, y_out, k_tob, t_sess-sess_off]);

    %Segment boundaries
    Sx2 = zeros([2, k_tob, t_sess-sess_off]);
    Sy2 = zeros([2, k_tob, t_sess-sess_off]);


    %PCA
    Xp2 = zeros([m_pca, k_tob, t_sess-sess_off]);
    Xcp2 = zeros([x_pca, t_out, 1, k_tob, t_sess-sess_off]);
    Xrp2 = ones([m_pca+1, k_tob, t_sess-sess_off]);
    Xsp2 = zeros([x_pca, t_in, k_tob, t_sess-sess_off]);


    k_iob = k_tob * (t_sess-sess_off);
    XI2 = zeros([m_in, k_iob]);


    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

            st_idx = idx;
            end_idx = idx+t_in-1;
            Sx2(1,j,i) = st_idx;
            Sx2(2,j,i) = end_idx;

            %Normalize test data (both input and output) on input period
            Mw = M(st_idx:end_idx, x_off+1:x_off+x_in);
            [Bti(1,:,j,i), Bti(2,:,j,i)] = bounds(Mw,1);
            Bti(3,:,j,i) = mean(Mw,1);
            Bti(4,:,j,i) = std(Mw,0,1);
            
            Myw = M(st_idx:end_idx, y_off+1:y_off+y_out);
            [Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);
            Bto(3,:,j,i) = mean(Myw,1);
            Bto(4,:,j,i) = std(Myw,0,1);

            Mx = reshape( Mw', [m_in,1] );
            X2(1:m_in, j, i) = Mx(:);
            Xr2(1:m_in, j, i) = Mx(:);
            Xc2(:, :, 1, j, i) = Mw';
            Xs2(:,:,j,i) = Mw';


            st_idx = idx+t_in;
            end_idx = idx+t_in+t_out-1;
            Sy2(1,j,i) = st_idx;
            Sy2(2,j,i) = end_idx;

            Myw = M(st_idx:end_idx, y_off+1:y_off+y_out);
            My = reshape( Myw', [n_out,1] );
            Yh2(:, j, i) = My(:);

            %[Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);

            %My = reshape( Myw', [n_out,1] );
            Yhs2(:, j, i) = My(:);

            Ysh2(:,:, j, i) = Myw';
            Yshs2(:,:, j, i) = Myw';


            i_idx = (i-1)*k_tob + j;
            XI2(1:m_in, i_idx) = Mx(:);
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
                Xr2(1:m_in, j, i) = Mx(:);
                Xc2(:, :, 1, j, i) = Mw';
                Xs2(:,:,j,i) = Mw';


                %PCA
                if Vit
                    Mxwp = pca_map(Mw', Vit(:,:,i));
                    Mxp = reshape( Mxwp, [m_pca,1] );
                    Xp2(1:m_pca, j, i) = Mxp;
                    Xrp2(1:m_pca, j, i) = Mxp;
                    Xcp2(:, :, 1, j, i) = Mxwp;
                    Xsp2(:,:,j,i) = Mxwp;
                end

                i_idx = (i-1)*k_tob + j;
                XI2(1:m_in, i_idx) = Mx(:);
                
             end
        else
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Mw = M(idx:idx+t_in-1, x_off+1:x_off+x_in);

                %PCA
                if Vit
                    Mxwp = pca_map(Mw', Vit(:,:,i));
                    Mxp = reshape( Mxwp, [m_pca,1] );
                    Xp2(1:m_pca, j, i) = Mxp;
                    Xrp2(1:m_pca, j, i) = Mxp;
                    Xcp2(:, :, 1, j, i) = Mxwp;
                    Xsp2(:,:,j,i) = Mxwp;
                end

             end
        end
        
        if(norm_flo)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                Myw = M(idx+t_in:idx+t_in+t_out-1, y_off+1:y_off+y_out);
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

                Yshs2(:,:, j, i) = Myw';
            end
        end

    end
end