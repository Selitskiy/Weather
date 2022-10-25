function Mn = generic_minmax_scale2D(M, Min, Max)
    [m, n] = size(M);
    Mn = zeros([m, n]);
    for i = 1:n
        if Max(i) == Min(i)
            if Max(i) == 0
                Mn(:,i) = 0;
            else
                Mn(:,i) = 1;
            end
        else
            Mn(:,i) = (M(:,i) - Min(i)) / (Max(i) - Min(i));
        end
    end
end