function generic_seq_err_graph2D(M, l_m, Y2, l_y, l_sess, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, k_start, modelName)
    
    f = figure();

    for k = 1:y_out

        % Re-format sessions back into through array
        M2 = M(:,x_in+k);

        for i = 1:t_sess-sess_off
            for j = 1:k_tob

                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;

                %Myw = reshape( Y2(:, j, i), [y_out, t_out])';

                Myw = Y2(x_in+k, :, j, i)';
                M2(idx+t_in:idx+t_in+t_out-1, 1) = Myw;

            end
        end

        if(k_start==0)
            k_start=1;
        end

        lp = plot(k_start:l_y, M2(k_start:l_y), 'r:', k_start:l_m, M(k_start:l_m,x_in+k), 'b','LineWidth', 2);

        hold on;

    end

    title(strcat("Model ",modelName))
    xlabel('Observations')
    ylabel('Moistrure %')
    legend('prediction', 'observation')
end