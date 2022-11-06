function generic_ar_err_graph2D(M, Em, l_m, Y2, Sy2, l_y, l_sess, x_off, x_in, t_in, y_off, y_out, t_out, n_xy, k_tob, t_sess, sess_off, offset, k_start, modelName)

    %m_in = x_in * t_in;
    
    legItems = strings(0);

    f = figure();
        
    if(k_start==0)
        k_start=1;
    end

    [m,n]=size(M);



    M3 = zeros([m,n]);
    M4 = zeros([m,n]);

    yyaxis left

    for k = 1:y_out
        lp = plot(k_start:l_m, M(k_start:l_m, y_off+k), 'b', 'MarkerSize', 2, 'LineWidth', 1);
        hold on;
        legIt = strcat('observation ', num2str(k));
        legItems = [legItems, legIt];
    end


    for i = 1:t_sess-sess_off
        y_st2 = min(Sy2(1,:,i));
        y_end2 = max(Sy2(2,:,i));

        for j = 1:k_tob
            Myw = reshape(Y2(:, j, i), [n_xy, t_out])';
            %lp = plot(Sy(1,j,i):Sy(2,j,i), Myw(:,k)+1, 'm', 'LineWidth', 2);
            M3(Sy2(1,j,i):Sy2(2,j,i), x_off+1:x_off+n_xy) = Myw;
        end

        for k = 1:y_out
            lp = plot(y_st2:y_end2, M3(y_st2:y_end2, y_off+k), 'r', 'MarkerSize', 2,'LineWidth', 1);
            hold on;
            legIt = strcat('prediction ', num2str(k));
            legItems = [legItems, legIt];
        end
    end

    ylabel('Moistrure %')


    yyaxis right

    for i = 1:t_sess-sess_off
        y_st2 = min(Sy2(1,:,i));
        y_end2 = max(Sy2(2,:,i));

        for j = 1:k_tob
            Mew = reshape(Em(:, j, i), [y_out, t_out])';
            M4(Sy2(1,j,i):Sy2(2,j,i), y_off+1:y_off+y_out) = Mew;
        end

        for k = 1:y_out
            lp = plot(y_st2:y_end2, M4(y_st2:y_end2, y_off+k), 'g', 'MarkerSize', 2,'LineWidth', 1);
            hold on;
            legIt = strcat('sess ', num2str(i), ', error ', num2str(k));
            legItems = [legItems, legIt];
        end
    end

    title(strcat("Model ",modelName))
    xlabel('Observations')
    ylabel('Error')
    legend(legItems)
end