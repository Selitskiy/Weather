function M = generic_minmax_rescale2D(Mn, Min, Max)
    [m, n] = size(Mn);
    M = zeros([m, n]);
    for i = 1:n
        M(:,i) = Mn(:,i) * (Max(i) - Min(i)) + Min(i);
    end
end