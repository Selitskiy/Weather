function [X2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, k_tob] = generic_test_seq_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, l_test, t_sess, sess_off, offset,...
    norm_fli, norm_flo, Bi, Bo, k_ob, k_tob)

    %% Test regression ANN
    if(k_tob == 0)
        k_tob = floor(l_sess/t_out); %ceil
    end

    n_xy = x_in + y_out; 

    X2 = ones([n_xy, k_ob+1, k_tob, t_sess-sess_off]);
    Y2s = zeros([n_xy, k_ob+1, k_tob, t_sess-sess_off]);
    Y2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Yh2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Yhs2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Bti = zeros([2, n_xy, k_tob, t_sess-sess_off]);
    Bto = zeros([2, n_xy, k_tob, t_sess-sess_off]);


    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;


            Mxw = M(idx-1:idx+k_ob-1, 1:n_xy);
            % scale bounds over observation span
            [Bti(1,:,j,i), Bti(2,:,j,i)] = bounds(Mxw,1);

            %Mx = reshape( Mxw', [m_in,1] );
            X2(:, :, j, i) = Mxw';


            Myw = M(idx+k_ob:idx+k_ob-1+t_out, 1:n_xy);
            % scale bounds over observation span
            [Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);

            %Mx = reshape( Mxw', [m_in,1] );
            Yh2(:, :, j,i) = Myw';
        end


        if(norm_fli)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;


                Mxw = M(idx-1:idx+k_ob-1, 1:n_xy);
                % bounds over session
                MinSessi = Bi(1,:,i); 
                MaxSessi = Bi(2,:,i);

                Mxw = generic_minmax_scale2D(Mxw, MinSessi, MaxSessi);
                X2(:, :, j, i) = Mxw';
            end
        end


        if(norm_flo)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;


                Myw = M(idx+k_ob:idx+k_ob-1+t_out, 1:n_xy);
                % bounds over session
                MinSesso = Bo(1,:,i); 
                MaxSesso = Bo(2,:,i);

                Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);
                Yh2(:, :, j,i) = Myw';
            end
        end

    end

end