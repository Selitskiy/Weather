function generic_test_in_graph2D(M, l_m, X, Y, X2, Y2, Sx, Sy, Sx2, Sy2, l_y, n_sess, l_sess, k_ob, x_in, t_in, y_out, t_out, k_tob, t_sess, sess_off, offset, k_start, model_name)

    m_in = x_in * t_in;

    f = figure();
        
    if(k_start==0)
        k_start=1;
    end

    [m,n]=size(M);

    M2 = zeros([m,n]);
    M3 = zeros([m,n]);

    %x_st = min(Sx(1,:,:));
    %x_end = max(Sx(2,:,:));
    
    %y_st = min(Sy(1,:,:));
    %y_end = max(Sy(2,:,:));

    %x_st2 = min(Sx2(1,:,:));
    %x_end2 = max(Sx2(2,:,:));
    
    %y_st2 = min(Sy2(1,:,:));
    %y_end2 = max(Sy2(2,:,:));

    for k = 1:x_in
        lp = plot(k_start:l_m, M(k_start:l_m, k), 'b', 'LineWidth', 2);
        hold on;
    end


    for i = 1:n_sess
        x_st = min(Sx(1,:,i));
        x_end = max(Sx(2,:,i));

        for j = 1:k_ob
            Mxw = reshape(X(1:m_in, j, i), [x_in, t_in])';

            %lp = plot(Sx(1,j,i):Sx(2,j,i), Mxw(:,k), 'g', 'LineWidth', 2);
            M2(Sx(1,j,i):Sx(2,j,i), 1:x_in) = Mxw;
        end

        for k = 1:x_in
            lp = plot(x_st:x_end, M2(x_st:x_end, k), 'g','LineWidth', 2);
            hold on;
        end
    end


    for i = 1:t_sess-sess_off
        x_st2 = min(Sx2(1,:,i));
        x_end2 = max(Sx2(2,:,i));

        for j = 1:k_tob
            Mxw = reshape(X2(1:m_in, j, i), [x_in, t_in])';
            %lp = plot(Sx2(1,j,i):Sx2(2,j,i), Mxw(:,k), 'c', 'LineWidth', 2);
            M3(Sx2(1,j,i):Sx2(2,j,i), 1:x_in) = Mxw;
        end

        for k = 1:x_in
            lp = plot(x_st2:x_end2, M3(x_st2:x_end2,k), 'm','LineWidth', 2);
            hold on;
        end
    end



    M2 = zeros([m,n]);
    M3 = zeros([m,n]);

    f2 = figure();

    for k = 1:y_out
        lp = plot(k_start:l_m, M(k_start:l_m, x_in+k), 'b', 'LineWidth', 2);
        hold on;
    end


    for i = 1:n_sess
        y_st = min(Sy(1,:,i));
        y_end = max(Sy(2,:,i));

        for j = 1:k_ob
            Myw = reshape(Y(:, j, i), [y_out, t_out])';
            %lp = plot(Sy(1,j,i):Sy(2,j,i), Myw(:,k)+1, 'm', 'LineWidth', 2);
            M2(Sy(1,j,i):Sy(2,j,i), x_in+1:x_in+y_out) = Myw;            
        end

        for k = 1:y_out
            lp = plot(y_st:y_end, M2(y_st:y_end, x_in+k), 'c','LineWidth', 2);
            hold on;
        end
    end


    for i = 1:t_sess-sess_off
        y_st2 = min(Sy2(1,:,i));
        y_end2 = max(Sy2(2,:,i));

        for j = 1:k_tob
            Myw = reshape(Y2(:, j, i), [y_out, t_out])';
            %lp = plot(Sy(1,j,i):Sy(2,j,i), Myw(:,k)+1, 'm', 'LineWidth', 2);
            M3(Sy2(1,j,i):Sy2(2,j,i), x_in+1:x_in+y_out) = Myw;
        end

        for k = 1:y_out
            lp = plot(y_st2:y_end2, M3(y_st2:y_end2, x_in+k), 'r','LineWidth', 2);
            hold on;
        end
    end



    title(strcat("Model ",model_name))
    xlabel('Observations')
    ylabel('Inputs')
    legend('split+joined', 'whole')
end