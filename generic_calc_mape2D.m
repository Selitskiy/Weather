function [E2f, S2, S2Mean, S2Std, S2s, ma_mape, sess_ma_idx, ob_ma_idx, mi_mape, sess_mi_idx, ob_mi_idx]=generic_calc_mape2D(Y2, Yh2, n_out)
  
    E2f(:, :, :) = abs((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :)) ./ Yh2(1:n_out, :, :));
    [skf, sjf, sif] = size(E2f);
    S2 = sum(E2f, 'all');
    Sn2 = skf*sjf*sif;
    
    S2 = S2/Sn2;

    [ma_errs, i] = max(E2f, [], [1 2],"linear");
    [ma_err, sess_ma_idx]=max(ma_errs);
    ob_ma_idx = i(sess_ma_idx);
    ma_mape = mean(ma_errs);

    [mi_errs, i] = min(E2f, [], [1 2],"linear");
    [mi_err, sess_mi_idx]=min(mi_errs);
    ob_mi_idx = i(sess_mi_idx);
    mi_mape = mean(mi_errs);

    % Per session mean+std
    S2s = sum(E2f, [1,2]);
    Sn2s = skf*sjf;
    
    S2s = S2s/Sn2s;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
    
end