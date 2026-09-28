function [w_best, v_best, S_best, error_best, time_global, time_iteration] = mfrost(X_list, r, varargin)
%MFROST Joint nonnegative matrix trifactorization for multilayer graphs.
%   X_list is a cell array of nonnegative, symmetric n-by-n matrices.
%   Z_l is represented by Z_l(i,v(i)) = w(l,i), with all other entries zero.
%
%   Name-value options:
%     'numTrials'  : number of restarts                     (default 10)
%     'init'       : 'USENC' or 'random'                    (default 'USENC')
%     'maxiter'    : maximum iterations per restart         (default 100)
%     'delta'      : absolute change in relative error      (default 1e-6)
%     'time_limit' : total time budget, in seconds          (default Inf)
%     'verbosity'  : print progress                         (default 1)
%     'cardano'    : 'batch' or 'scalar'                     (default 'batch')
%
%   'scalar' uses the original scalar root solver, with the same cached
%   coefficients, for comparison with the vectorized 'batch' solver.
%   Set rng(seed) before calling this function for reproducible restarts.
%
%   Output shapes: w_best is L-by-n; v_best is 1-by-n;
%   S_best is L-by-r-by-r. time_iteration{t}{j} is the duration of iteration
%   j in restart t. time_global includes preprocessing and initialization.
%   time_limit is checked between iterations and restarts, so a running
%   initialization or iteration can exceed this budget.
%
%   The reported error keeps the original convention:
%       sum_l sqrt(1e-9 + ||X_l||_F^2 - ||S_l||_F^2) / sum_l ||X_l||_F.
%   This is NOT sqrt(sum_l residual_l^2 / sum_l ||X_l||_F^2).
%   A zero data norm returns error zero. Empty columns remain zero.
%
%   Optimizations:
%     * Cache off-diagonal neighbours, diagonals and edge lists once.
%     * Store each S_l as an r-by-r matrix throughout the optimization.
%     * Aggregate squared weights by community when initializing p.
%     * Solve all L*r quartics together, retaining sequential node updates.
%     * For nonnegative X,S,w, the cubic constant is nonpositive. Only
%       the largest nonnegative stationary root and the boundary are needed.
%
%   USENC initialization requires the existing SSPA, orthNNLS and USENC
%   functions on the MATLAB path. No new toolbox is required.

start_time = tic;
options = struct('numTrials', 10, 'init', 'USENC', 'maxiter', 100, ...
                 'delta', 1e-6, 'time_limit', Inf, 'verbosity', 1, ...
                 'cardano', 'batch');
if mod(numel(varargin), 2) ~= 0
    error('mfrost:Options', 'Options must be name-value pairs.');
end
for k = 1:2:numel(varargin)
    name = char(varargin{k});
    if ~isfield(options, name)
        error('mfrost:Options', 'Unknown option: %s.', name);
    end
    options.(name) = varargin{k+1};
end
if ~iscell(X_list) || isempty(X_list)
    error('mfrost:Input', 'X_list must be a nonempty cell array.');
end
L = numel(X_list);
n = size(X_list{1}, 1);
validateattributes(r, {'numeric'}, {'scalar', 'integer', '>=', 1, '<=', n});
validateattributes(options.numTrials, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(options.maxiter, {'numeric'}, {'scalar', 'integer', '>=', 0});
validateattributes(options.delta, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0});
validateattributes(options.time_limit, {'numeric'}, {'scalar', 'real', 'nonnan', '>=', 0});
init_method = char(options.init);
if ~any(strcmpi(init_method, {'random', 'USENC'}))
    error('mfrost:Initialization', 'init must be ''random'' or ''USENC''.');
end
switch lower(char(options.cardano))
    case 'batch'
        solve_quartics = @cardano_batch;
    case 'scalar'
        solve_quartics = @cardano_scalar_batch;
    otherwise
        error('mfrost:Options', 'cardano must be ''batch'' or ''scalar''.');
end

% X is fixed across ALL restarts. MATLAB sparse storage is column-oriented;
% one find(X.') groups the entries of each original row contiguously.
[X_list, neighbours, neighbour_values, diagonal_X, degrees, I, J, VAL, normX2] = ...
    prepare_layers(X_list, n, L);
norm_sum = sum(sqrt(normX2));
has_degree = degrees ~= 0;

error_best = Inf;
w_best = zeros(L, n);
v_best = zeros(1, n);
S_best = zeros(L, r, r);
time_iteration = cell(options.numTrials, 1);
completed_trials = 0;
if options.verbosity > 0
    fprintf('Running %u Trials in Series\n', options.numTrials);
end

for trial = 1:options.numTrials
    if trial > 1 && toc(start_time) >= options.time_limit
        break;
    end

    if norm_sum == 0
        % An empty multilayer graph contains no information about labels.
        v = mod(0:n-1, r) + 1;
    elseif strcmpi(init_method, 'random')
        % Generate ONE partition, rather than overwriting it n times.
        v = [1:r, randi(r, 1, n-r)];
        v = v(randperm(n));
    else
        v_layer = zeros(L, n);
        for l = 1:L
            v_layer(l,:) = reshape(community_detection_SVCA(X_list{l}, r), 1, n);
        end
        v = reshape(USENC(v_layer, r), 1, n);
    end
    if any(~isfinite(v)) || any(v < 1 | v > r | v ~= fix(v))
        error('mfrost:Labels', 'Initialization must return integer labels in 1:r.');
    end

    w = zeros(L, n);
    for l = 1:L
        community_degree = accumarray(v(:), degrees(l,:).', [r,1], @sum, 0);
        denom = reshape(community_degree(v), 1, n);
        nonzero = denom ~= 0;
        w(l,nonzero) = degrees(l,nonzero) ./ denom(nonzero);
    end
    w = normalize_weights(w, v, r);
    [S_layer, normS2] = update_interactions(I, J, VAL, w, v, r, L);
    current_error = relative_error(normX2, normS2, norm_sum);
    iteration_times = zeros(1, options.maxiter);
    completed_iterations = 0;

    for iteration = 1:options.maxiter
        if toc(start_time) >= options.time_limit
            break;
        end
        start_iteration = tic;

        % S is fixed during this entire coordinate sweep. Precompute only
        % once per iteration, using an r-vector of community squared norms.
        p = zeros(L, r);
        diagonal_S = zeros(L, r);
        for l = 1:L
            Sl = S_layer{l};
            mass = accumarray(v(:), (w(l,:).^2).', [r,1], @sum, 0);
            p(l,:) = mass.' * (Sl.^2);
            diagonal_S(l,:) = diag(Sl).';
        end
        cubic_a = 4 * diagonal_S.^2;
        cubic_b = zeros(L, r);
        cubic_c = zeros(L, r);
        old_contribution = zeros(L, r);

        for i = randperm(n)
            old_k = v(i);
            for l = 1:L
                Sl = S_layer{l};
                old_contribution(l,:) = (w(l,i) * Sl(old_k,:)).^2;
                if ~has_degree(l,i)
                    cubic_b(l,:) = 0;
                    cubic_c(l,:) = 0;
                    continue;
                end

                % Derivative coefficients a*x^3 + b*x + c = 0.
                cubic_b(l,:) = 4 * (p(l,:) - old_contribution(l,:)) ...
                               - 4 * diagonal_X(l,i) * diagonal_S(l,:);
                cols = neighbours{l,i};
                weighted_edges = neighbour_values{l,i} .* w(l,cols);
                cubic_c(l,:) = -4 * (weighted_edges * Sl(v(cols),:));
            end

            % Solve all candidates before changing ANY node assignment.
            [candidate_w, candidate_cost] = solve_quartics(cubic_a, cubic_b, cubic_c);
            [~, best_k] = min(sum(candidate_cost, 1));
            best_w = candidate_w(:,best_k);

            % Maintain p incrementally. Subsequent nodes must use this new
            % w and v, so the loop over nodes is intentionally sequential.
            for l = 1:L
                Sl = S_layer{l};
                p(l,:) = p(l,:) - old_contribution(l,:) ...
                         + (best_w(l) * Sl(best_k,:)).^2;
            end
            w(:,i) = best_w;
            v(i) = best_k;
        end

        w = normalize_weights(w, v, r);
        [S_layer, normS2] = update_interactions(I, J, VAL, w, v, r, L);
        previous_error = current_error;
        current_error = relative_error(normX2, normS2, norm_sum);
        iteration_times(iteration) = toc(start_iteration);
        completed_iterations = iteration;
        if current_error < options.delta || abs(previous_error-current_error) < options.delta
            break;
        end
    end

    completed_trials = trial;
    time_iteration{trial} = num2cell(iteration_times(1:completed_iterations));
    if current_error <= error_best
        w_best = w;
        v_best = v;
        for l = 1:L
            S_best(l,:,:) = reshape(S_layer{l}, 1, r, r);
        end
        error_best = current_error;
    end
    if options.verbosity > 0
        fprintf('Trial %u of %u with %s: %.4e | Best: %.4e\n', ...
                trial, options.numTrials, init_method, current_error, error_best);
    end
    if error_best <= options.delta || toc(start_time) >= options.time_limit
        break;
    end
end
time_iteration = time_iteration(1:completed_trials);
time_global = toc(start_time);
end


function [X_list, neighbours, values, diagonal_X, degrees, I, J, VAL, normX2] = prepare_layers(X_list, n, L)
neighbours = cell(L, n);
values = cell(L, n);
I = cell(L, 1);
J = cell(L, 1);
VAL = cell(L, 1);
diagonal_X = zeros(L, n);
degrees = zeros(L, n);
normX2 = zeros(L, 1);
for l = 1:L
    X = sparse(double(X_list{l}));
    if ~isequal(size(X), [n,n]) || ~isreal(X)
        error('mfrost:Input', 'Each layer must be a real n-by-n matrix.');
    end
    [I{l}, J{l}, VAL{l}] = find(X);
    val = VAL{l};
    if any(~isfinite(val)) || any(val < 0)
        error('mfrost:Input', 'Each layer must be finite and nonnegative.');
    end
    X_list{l} = X;
    diagonal_X(l,:) = full(diag(X)).';
    degrees(l,:) = full(sum(X, 1));
    normX2(l) = sum(val.^2);

    [cols, row, val] = find(X.');
    off_diagonal = cols ~= row;
    cols = cols(off_diagonal);
    row = row(off_diagonal);
    val = val(off_diagonal);
    counts = accumarray(row, ones(size(row)), [n,1], @sum, 0);
    offsets = [0; cumsum(counts)];
    for i = 1:n
        idx = offsets(i)+1:offsets(i+1);
        neighbours{l,i} = reshape(cols(idx), 1, []);
        values{l,i} = reshape(val(idx), 1, []);
    end
end
end


function w = normalize_weights(w, v, r)
for l = 1:size(w,1)
    col_norm = sqrt(accumarray(v(:), (w(l,:).^2).', [r,1], @sum, 0));
    denom = reshape(col_norm(v), 1, []);
    nonzero = denom ~= 0;
    w(l,nonzero) = w(l,nonzero) ./ denom(nonzero);
    w(l,~nonzero) = 0;
end
end


function [S_layer, normS2] = update_interactions(I, J, VAL, w, v, r, L)
S_layer = cell(L, 1);
normS2 = zeros(L, 1);
for l = 1:L
    rows = reshape(v(I{l}), [], 1);
    cols = reshape(v(J{l}), [], 1);
    weights = reshape(w(l,I{l}), [], 1) .* reshape(w(l,J{l}), [], 1) .* VAL{l};
    Sl = accumarray([rows,cols], weights, [r,r], @sum, 0);
    S_layer{l} = Sl;
    normS2(l) = sum(Sl(:).^2);
end
end


function err = relative_error(normX2, normS2, norm_sum)
if norm_sum == 0
    err = 0;
else
    % Protect sqrt against negative roundoff at an exact factorization.
    err = sum(sqrt(max(1e-9 + normX2 - normS2, 0))) / norm_sum;
end
end


function [x, f] = cardano_batch(a, b, c)
% Solve min_{x>=0} a*x^4/4 + b*x^2/2 + c*x elementwise.
% In this algorithm a>=0 and c<=0. For a>0,c<0 the derivative has exactly
% one positive root; if c=0 the minimizer is 0 or sqrt(-b/a). Thus negative
% roots and complex arithmetic are unnecessary. Never use this specialized
% solver for problems with positive c without adapting the root selection.
% The 1e-12 linear-degeneracy threshold is kept from the original code.
tol = 1e-12;
x = zeros(size(a));

linear = a < tol;
take = linear & b > tol;
x(take) = -c(take) ./ b(take);

quartic = ~linear;
zero_constant = quartic & c == 0 & b < 0;
x(zero_constant) = sqrt(-b(zero_constant) ./ a(zero_constant));
take = quartic & c < 0;
if any(take(:))
    p = b(take) ./ a(take);
    q = c(take) ./ a(take);
    discriminant = (q/2).^2 + (p/3).^3;
    roots = zeros(size(p));

    one_real = discriminant >= 0;
    if any(one_real)
        pp = p(one_real);
        qq = q(one_real);
        u = (-qq/2 + sqrt(discriminant(one_real))).^(1/3);
        z = -pp ./ (3*u);
        root = u + z;
        % For p>0, u and z have opposite signs. This equivalent expression
        % avoids cancellation in u+z when the positive root is small.
        stable = pp > 0;
        root(stable) = -qq(stable) ./ ...
            (u(stable).^2 - u(stable).*z(stable) + z(stable).^2);
        roots(one_real) = root;
    end
    three_real = ~one_real;
    if any(three_real)
        pp = p(three_real);
        qq = q(three_real);
        cos_arg = (-qq/2) ./ sqrt(-(pp/3).^3);
        cos_arg = min(1, max(-1, cos_arg));
        roots(three_real) = 2*sqrt(-pp/3) .* cos(acos(cos_arg)/3);
    end
    x(take) = roots;
end

% Evaluate the same quartic as the scalar code, and compare with x=0.
x2 = x.^2;
f = ((a/4).*x2 + b/2).*x2 + c.*x;
reject = x <= 0 | ~isfinite(x) | ~isfinite(f) | f >= 0;
x(reject) = 0;
f(reject) = 0;
end


function [x, f] = cardano_scalar_batch(a, b, c)
% Reference mode: original scalar Cardano formula applied to each entry.
x = zeros(size(a));
f = zeros(size(a));
for j = 1:numel(a)
    [x(j), f(j)] = cardano_scalar(a(j), b(j), c(j));
end
end


function [x_opt, f_opt] = cardano_scalar(c3, c1, c0)
x_opt = 0;
f_opt = 0;
if abs(c3) < 1e-12
    if abs(c1) > 1e-12
        x = -c0/c1;
        if x > 0
            f = (c3/4)*x^4 + (c1/2)*x^2 + c0*x;
            if f < f_opt
                x_opt = x;
                f_opt = f;
            end
        end
    end
else
    p = c1/c3;
    q = c0/c3;
    discriminant = 4*p^3 + 27*q^2;
    d = 0.5*(-q + sqrt(discriminant/27));
    if discriminant <= 0
        radius = 2*abs(d)^(1/3);
        angle = atan2(imag(d), real(d))/3;
        roots = radius * [cos(angle), cos(angle+2*pi/3), cos(angle+4*pi/3)];
        roots = roots(roots > 0);
        if ~isempty(roots)
            values = (c3/4)*roots.^4 + (c1/2)*roots.^2 + c0*roots;
            [value, j] = min(values);
            if value < f_opt
                x_opt = roots(j);
                f_opt = value;
            end
        end
    else
        d2 = 0.5*(-q - sqrt(discriminant/27));
        x = sign(d)*abs(d)^(1/3) + sign(d2)*abs(d2)^(1/3);
        if x > 0
            f = (c3/4)*x^4 + (c1/2)*x^2 + c0*x;
            if f < f_opt
                x_opt = x;
                f_opt = f;
            end
        end
    end
end
end


function v = community_detection_SVCA(X, r)
n = size(X, 1);
p = max(2, floor(0.1*n/r));
options.average = 1;
[ZO, ~] = SSPA(X, r, p, options);
norm2x = full(sqrt(sum(X.^2, 1)));
% Sparse column scaling avoids an implicit dense n-by-n intermediate.
Xn = X * spdiags((1 ./ (norm2x + 1e-16)).', 0, n, n);
HO = orthNNLS(X, ZO, Xn);
Z = HO';
v = max((Z ~= 0) .* (1:r), [], 2);
zero_idx = find(v == 0);
v(zero_idx) = randi(r, size(zero_idx));
v = reshape(v, 1, n);
end
