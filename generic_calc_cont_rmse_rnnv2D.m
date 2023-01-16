function [E2f, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx]=generic_calc_cont_rmse_rnnv2D(Y2, Yh2, n_out, y_out)

    [~, t_out, ktob, nsess]=size(Y2);
    E2f = zeros([y_out, 1, ktob-1, nsess]);
  
    for j=2:ktob
        E2f(:, :, j-1, :) = ((Y2(:, 1, j, :) - Y2(:, t_out, j-1, :))).^2;
    end
    
    [skn, skf, sjf, sif] = size(E2f);
    S2 = sum(E2f, [2,3,4]);
    Sn2 = skn*skf*sjf*sif;
    
    S2 = sum(sqrt(S2/Sn2), 1)/skn;

    [m, i] = max(E2f);
    [ma_err, sess_ma_idx]=max(m);
    ob_ma_idx = i(sess_ma_idx);

    [m, i] = min(E2f);
    [mi_err, sess_mi_idx]=min(m);
    ob_mi_idx = i(sess_mi_idx);

    % Per session mean+std
    S2s = sum(E2f, [2,3]);
    Sn2s = skf*sjf;
    
    S2s = sum(sqrt(S2s/Sn2s), 1)/skn;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
end