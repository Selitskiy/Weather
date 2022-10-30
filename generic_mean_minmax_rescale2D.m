function M = generic_mean_minmax_rescale2D(Mn, Mean, Min, Max)
    [m, n] = size(Mn);
    M = zeros([m, n]);

    eps = 0.00001;

    for i = 1:n
        M(:,i) = Mn(:,i) * (abs(Max(i) - Min(i)) + eps) + Mean(i);
    end
end