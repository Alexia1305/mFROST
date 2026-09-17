
% USECNC method used in our initialization for mfrost to obtain a community
% assignements consensus 
% Combine the L node partitions in v_list to obtain the final clustering (concensus node partition) result (with r clusters).
% based on the paper "Ultra-scalable spectral clustering and ensemble
% clustering", IEEE TKDE, 2019. 
function v = USENC(v_list, r)

    % v_list : L × n
    [L, n] = size(v_list);

    v_list = double(v_list);

    % Vérification : les labels doivent commencer à 1
    if any(v_list(:) < 1) || any(mod(v_list(:),1) ~= 0)
        error('v_list must contain positive integer labels.');
    end

    % Nombre de communautés dans chaque partition
    r_list = max(v_list, [], 2);

    % Décalage propre à chaque partition
    offsets = [0; cumsum(r_list(1:end-1))];

    % Nouveaux indices de communautés
    v_list_new = v_list + offsets;

    total_clusters = sum(r_list);

    % Graphe biparti nœuds-communautés
    rows = repelem((1:n)', L);
    cols = v_list_new(:);

    B = sparse( ...
        rows, ...
        cols, ...
        ones(n*L,1), ...
        n, ...
        total_clusters ...
    );

    % Supprimer les communautés vides
    B = B(:, full(sum(B,1)) > 0);

    v = Tcut_for_bipartite_graph(B, r, 100, 20);
end


function labels = Tcut_for_bipartite_graph( ...
    B, Nseg, maxKmIters, cntReps)

    if nargin < 3
        maxKmIters = 100;
    end

    if nargin < 4
        cntReps = 20;
    end

    [Nx, Ny] = size(B);

    if Ny < Nseg
        error('Need more columns!');
    end

    % Degrés des nœuds
    dx = full(sum(B,2));
    dx(dx == 0) = 1e-10;

    Dx = spdiags(1./dx, 0, Nx, Nx);

    % Affinité entre communautés
    Wy = B' * Dx * B;

    % Normalisation
    d = full(sum(Wy,2));
    d(d == 0) = 1e-10;

    D = spdiags(1./sqrt(d), 0, Ny, Ny);

    nWy = D * Wy * D;
    nWy = (nWy + nWy')/2;

    % Décomposition spectrale symétrique
    [evecs, evals] = eig(full(nWy), 'vector');
    [~, order] = sort(evals, 'descend');

    Ncut_evec = D * evecs(:, order(1:Nseg));

    % Projection vers les nœuds
    evec = Dx * B * Ncut_evec;

    % Normalisation des lignes
    row_norms = sqrt(sum(evec.^2,2));
    row_norms(row_norms == 0) = 1e-10;

    evec = evec ./ row_norms;

    % K-means
    
    options = statset('MaxIter', maxKmIters);

    labels = kmeans( ...
        evec, ...
        Nseg, ...
        'Start', 'plus', ...
        'Replicates', cntReps, ...
        'Options', options, ...
        'EmptyAction', 'singleton' ...
    );
end