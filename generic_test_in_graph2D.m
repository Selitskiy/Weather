function generic_test_in_graph2D(M, l_m, X2, l_y, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, k_start)

    %m_in = x_in * t_in;

    f = figure();

    for k = 1:x_in

        % Re-format sessions back into through array
        M2 = M(:,k);

        for i = 1:t_sess-sess_off
            for j = 1:k_tob

                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                %Mw = M(idx:idx+t_in-1, 1:x_in);
                %Mx = reshape( Mw', [m_in,1] );
                %X2(1:m_in, j, i) = Mx(:);

                Mxw = reshape( X2(:, j, i), [x_in, t_in])';
                M2(idx:idx+t_in-1, 1) = Mxw(:,k);

                %Myw = reshape( Y2(:, j, i), [y_out, t_out])';
                %M2(idx+t_in:idx+t_in+t_out-1, 1) = Myw(:, k);

            end
        end

        if(k_start==0)
            k_start=1;
        end

        lp = plot(k_start:l_y, M2(k_start:l_y), 'r:', k_start:l_m, M(k_start:l_m,k), 'b','LineWidth', 2);

        hold on;

    end

    %title(strcat("Model ",modelName))
    xlabel('Observations')
    ylabel('Inputs')
    legend('split+joined', 'whole')
end