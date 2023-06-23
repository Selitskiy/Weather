function [E2f, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx]=generic_seq_calc_rmse2D(Y2, Yh2, x_in, y_out, t_out)

    E2f = ((Y2(end-y_out+1:end, 1:t_out, :, :) - Yh2(end-y_out+1:end, 1:t_out, :, :))).^2;
    %E2f = ((Y2(end-y_out+1:end, 1:t_out, :, :) - Yh2(end-y_out+1:end, 1:t_out, :, :))).^2;
    [skn, skf, sjf, sif] = size(E2f);

    %Remove NaNs
    nn = 0;
    for n = 1:skn
    for k = 1:skf
        for j = 1:sjf
            for i = 1:sif
                if isnan(E2f(n,k,j,i))
                    E2f(n,k,j,i) = 0;
                    nn = nn + 1;
                end
            end
        end
    end
    end

    S2 = sum(E2f, [2, 3, 4]);
    Sn2 = skf*sjf*sif - nn;
    
    S2 = sum(sqrt(S2/Sn2), 1)/skn;

    [m, i] = max(E2f, [], [2, 4], 'linear');
    [ma_err, sess_ma_idx]=max(m);
    ob_ma_idx = i(sess_ma_idx);

    [m, i] = min(E2f, [], [2, 4], 'linear');
    [mi_err, sess_mi_idx]=min(m);
    ob_mi_idx = i(sess_mi_idx);

    % Per session mean+std
    S2s = sum(E2f, [2,3]);
    Sn2s = skf*sjf;
    
    S2s = sum(sqrt(S2s/Sn2s), 1)/skn;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
end