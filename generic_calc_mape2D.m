function [E2f, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx]=generic_calc_mape2D(Y2, Yh2, n_out)
  
    E2f(:, :, :) = abs((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :)) ./ Yh2(1:n_out, :, :));
    [skf, sjf, sif] = size(E2f);
    S2 = sum(E2f, 'all');
    Sn2 = skf*sjf*sif;
    
    S2 = S2/Sn2;

    [m, i] = max(E2f);
    [ma_err, sess_ma_idx]=max(m);
    ob_ma_idx = i(sess_ma_idx);

    [m, i] = min(E2f);
    [mi_err, sess_mi_idx]=min(m);
    ob_mi_idx = i(sess_mi_idx);

    % Per session mean+std
    S2s = sum(E2f, [1,2]);
    Sn2s = skf*sjf;
    
    S2s = S2s/Sn2s;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
    
end