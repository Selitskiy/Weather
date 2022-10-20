function Mn = generic_minmax_scale2D(M, Min, Max)
    [m, n] = size(M);
    Mn = zeros([m, n]);
    for i = 1:n
        Mn(:,i) = (M(:,i) - Min(i)) / (Max(i) - Min(i));
    end
end