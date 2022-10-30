function Mn = generic_mean_std_scale2D(M, Mean, Std)
    [m, n] = size(M);
    Mn = zeros([m, n]);

    eps = 0.00001;

    for i = 1:n
        Mn(:,i) = (M(:,i) - Mean(i)) / (Std(i) + eps);
    end
end