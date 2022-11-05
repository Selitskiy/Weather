function generic_seq_err_graph2D(M, l_m, Y2, Sy2, l_y, l_sess, x_off, x_in, t_in, y_off, y_out, t_out, n_xy, k_tob, t_sess, sess_off, offset, k_start, modelName)
    
    %m_in = x_in * t_in;
    legItems = strings(0);

    f = figure();
        
    if(k_start==0)
        k_start=1;
    end

    [m,n]=size(M);


    M3 = zeros([m,n]);

    for k = 1:y_out
        lp = plot(k_start:l_m, M(k_start:l_m, y_off+k), 'b', 'LineWidth', 2);
        hold on;
        legIt = strcat('observation ', num2str(k));
        legItems = [legItems, legIt];
    end


    for i = 1:t_sess-sess_off
        y_st2 = min(Sy2(1,:,i));
        y_end2 = max(Sy2(2,:,i));

        for j = 1:k_tob
            Myw = Y2(:, :, j, i)';
            %lp = plot(Sy(1,j,i):Sy(2,j,i), Myw(:,k)+1, 'm', 'LineWidth', 2);
            M3(Sy2(1,j,i):Sy2(2,j,i), x_off+1:x_off+n_xy) = Myw;
        end

        for k = 1:y_out
            lp = plot(y_st2:y_end2, M3(y_st2:y_end2, y_off+k), 'r','LineWidth', 2);
            hold on;
            legIt = strcat('sess ', num2str(i), ', prediction ', num2str(k));
            legItems = [legItems, legIt];
        end
    end


    
    %f = figure();

    %for k = 1:y_out

        % Re-format sessions back into through array
    %    M2 = M(:,y_off+k);

    %    for i = 1:t_sess-sess_off
    %        for j = 1:k_tob

    %            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - t_in;
    %            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;
                
                %Myw = reshape( Y2(:, j, i), [y_out, t_out])';

    %            Myw = Y2(y_off+k, :, j, i)';
    %            M2(idx+t_in:idx+t_in+t_out-1, 1) = Myw;

    %        end
    %    end

    %    if(k_start==0)
    %        k_start=1;
    %    end

    %    lp = plot(k_start:l_y, M2(k_start:l_y), 'r:', k_start:l_m, M(k_start:l_m,y_off+k), 'b','LineWidth', 2);

    %    hold on;

    %end

    title(strcat("Model ",modelName))
    xlabel('Observations')
    ylabel('Moistrure %')
    legend(legItems)
end