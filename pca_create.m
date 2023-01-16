function [Xet, Vit, Vi, V, It] = pca_create(X, nThresh, vThresh)

    C = cov( X' );
    [V, ~, ~] = eig(C);
    %Vi = flipud(inv(V));
    Vi = inv(V);
    %Xe = Vi * X;
    Xe = V\X;

    % Remove minor components with normalized var < thresh
    Expl = var(Xe, 0, 2)/sum(var(Xe, 0, 2));
    [n,~] = size(Expl);

    if nThresh
        It = logical(zeros([n,1]));
        It(end-nThresh+1:end) = logical(ones([nThresh,1]));
    else
        It = Expl >= vThresh;
    end
    
    Vit = Vi(It, :);
    Xet = Xe(It, :);


end