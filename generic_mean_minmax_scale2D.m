function Mn = generic_mean_minmax_scale2D(M, Mean, Min, Max)
    [m, n] = size(M);
    Mn = zeros([m, n]);

    eps = 0.00001;

    for i = 1:n
        Mn(:,i) = (M(:,i) - Mean(i)) / (abs(Max(i) - Min(i)) + eps);
    end
end