function [E2f, S2, S2Mean, S2Std, S2s, ma_rmse, sess_ma_idx, ob_ma_idx, mi_rmse, sess_mi_idx, ob_mi_idx]=generic_calc_rmse2D(Y2, Yh2, n_out)
  
    %E2f(:, :, :) = ((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :)) ./ Yh2(1:n_out, :, :)).^2;
    E2f(:, :, :) = ((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :))).^2; %square
    [skn, sjf, sif] = size(E2f);

    S2 = sum(E2f, [2,3]); %mean
    Sn2 = sjf*sif;
    
    S2 = sum(sqrt(S2/Sn2), 1)/skn; %root and mean over channels

    [ma_errs, i] = max(E2f, [], [1 2],"linear");
    [ma_err, sess_ma_idx] = max(sqrt(ma_errs));
    ob_ma_idx = i(sess_ma_idx);
    ma_rmse = sqrt(sum(ma_errs)/sif);

    [mi_errs, i] = min(E2f, [], [1 2],"linear");
    [mi_err, sess_mi_idx]=min(sqrt(mi_errs));
    ob_mi_idx = i(sess_mi_idx);
    mi_rmse = sqrt(sum(mi_errs)/sif);

    % Per session mean+std
    S2s = sum(E2f, 2);
    Sn2s = sjf;
    
    S2s = sum(sqrt(S2s/Sn2s), 1)/skn;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
end