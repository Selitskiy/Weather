function M = generic_mean_std_rescale2D(Mn, Mean, Std)
    [m, n] = size(Mn);
    M = zeros([m, n]);

    eps = 0.00001;

    for i = 1:n
        M(:,i) = Mn(:,i) * (Std(i) + eps) + Mean(i);
    end
end