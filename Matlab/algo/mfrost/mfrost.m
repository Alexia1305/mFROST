%  Heuristic algorithm for multilayer community detection via joint nonnegative matrix trifactorization.
% Estimates nonnegative matrices S_l>=0 and Z_l>=0 that minimize:
% 
%     sum_{l=1..L} ||X_l-Z_l S_l Z_l^T||_F^T
% 
% subject to the constraints:
%     Z_l = D_l V
%     Z_l^T Z_l = I
%     S_l >= 0
%     Z_l >= 0
% where:
% - X_l is the adjacency matrix of layer l
% - S_l is a nonnegative community interaction matrix for layer l
% - Z_l is a nonnegative orthogonal matrix encoding community memberships
% - D_l is a diagonal scaling matrix specific to layer l
% - V is a shared binary community assignment matrix across all layers
% 
% Notes
% -----
% The matrices Z_l=D_l V are not stored explicitly but represented with:
% 
% - v : ndarray of shape (n,) — layer-independent community assignments defining V,
%   where v[i] is the community index of node i.
% 
% - w : ndarray of shape (L, n) — layer-dependent diagonal values defining D_l,
%   where w[l, i] is the scaling factor for node i in layer l.
% 
% Thus, each row i of Z_l has a single nonzero element:
% 
%     Z_l[i, v[i]] = w[l, i]
% 
% The vector v is shared across layers, while w is layer-dependent.
%
% INPUTS
%    X_list : cell array of length L
%       List of adjacency matrices, one for each layer.
%       X_list{l} is an n-by-n sparse matrix representing layer l.
%       Each matrix must be nonnegative and symmetric, corresponding to
%       the adjacency matrix of an undirected graph.
%   r: Number of communities
%
% Options (varargin)
%   numTrials: Number of restarts with different initializations
%              Default: 10
%   init: Initialization method for Z
%         Options: 'random', 'USENC' (default)
%   maxiter: Maximum number of iterations per trial
%            Default: 100
%   delta: Convergence tolerance
%          Default: 1e-6
%          (Stop if the change in error between iterations is < delta or if error < delta)
%   time_limit: Time limit for a trial in seconds
%               Default: Inf
%   verbosity: Display messages (1) or not (0)
%              Default: 1

%
% OUTPUTS
%
%   w_best: Vector of length n; w(i) gives the value of the non-zero element
%           in the i-th row of Z
%   v_best: Vector of length n; v(i) gives the index of the column of Z
%           corresponding to the non-zero element in the i-th row
%   S_best: Central matrix of size r x r 
%   error_best: Relative error ||X-ZSZ||_F/||X||_F
%   time_global: Total Runtime
%   time_iteration: time_iteration{t}{i} time of iteration i in trial t
%
% This code is a supplementary material to the paper
%  TOCOMPLETE 

function [w_best,v_best,S_best,error_best,time_global,time_iteration] = mfrost(X_list,r,varargin)

if nargin <= 2
    options = [];
else
    for k = 1:2:length(varargin)
        options.(varargin{k}) = varargin{k+1};
    end
end
% Default Value 
if ~isfield(options, 'numTrials')
    options.numTrials = 10;
end
if ~isfield(options, 'time_limit')
    options.time_limit = Inf;
end 
if ~isfield(options, 'delta')
    options.delta = 1e-6;
end 
if ~isfield(options, 'maxiter')
    options.maxiter = 100;
end 
if ~isfield(options, 'verbosity')
    options.verbosity = 1;
end 
time_iteration = {};
start = tic;
error_best = 'inf';


 if options.verbosity > 0
        fprintf('Running %u Trials in Series \n', options.numTrials);
 end
 
% PRECOMPUTATION 
L = numel(X_list);
n = size(X_list{1},1);


I = cell(L, 1);
J = cell(L, 1);
VAL = cell(L, 1);

for l = 1:L
    [I{l}, J{l}, VAL{l}] = find(X_list{l});
end

normX = zeros(L, 1);
normX2 = zeros(L, 1);
for l = 1:L
    normX(l) = norm(X_list{l}, 'fro');
    normX2(l)=normX(l)^2;
end
degrees = zeros(L, n);

for l = 1:L
    degrees(l, :) = sum(X_list{l}, 1);
end
 
 
for trials = 1:options.numTrials
    
    %INITIALISATION
    time = {};
    w = zeros(L,n);
    v = zeros(1,n);
    S = zeros(L, r, r);
    if isfield(options, 'init')
        init_algo = options.init;
    else
        init_algo = "USENC";
    end 
    if init_algo == "random"
        for i = 1:n
           v = [1:r, randi(r,1,n-r)];
           v = v(randperm(n));
        end 
        
    elseif init_algo == "USENC"
        % Community detection in ecah layer with SVCA then concensus with
        % USENC
        v_l = zeros(L,n); % to store the communities in each layer 
        % Community detection in each layer with SVCA 
        for l = 1:L
           v_l(l,:) = community_detection_SVCA(X_list{l},r);
        end
        
        % censencus with USENC 
        v = USENC(v_l,r);
        
        
    end 
    % w set proportionnal to node degrees in each layer 
    for l = 1:L
        d_r = accumarray(v(:), degrees(l, :)', [r, 1], @sum, 0);
    
        denominator = d_r(v);
    
        w(l, :) = 0;
        idx = denominator ~= 0;
        w(l, idx) = degrees(l, idx) ./ denominator(idx)';
     end 
    
    % Normalization of wl 
    for l = 1:L
        colNorm = sqrt(accumarray(v(:),w(l,:).^2,[r 1],@sum,0));
        denom = colNorm(v).';
        mask = denom ~= 0;
        w(l,mask) = w(l,mask)./denom(mask);
        w(l,~mask) = 0;
    end 
    % Construction of Sl
    for l = 1:L
        [i, j, val] = deal(I{l}, J{l}, VAL{l});

        rows   = v(i(:));
        cols   = v(j(:));
        values = w(l,i(:)).' .* w(l,j(:)).' .* val(:);

        S(l,:,:) = accumarray([rows(:),cols(:)], values(:), [r,r], @sum, 0);
    

    end 
    % UPDATE error 
    error_pre=0;
    for l=1:L
        error_pre = error_pre + sqrt(1e-9+normX2(l)-norm(S(l,:,:),'fro')^2);
    end 
    error_pre = error_pre/sum(normX);
    error = error_pre;

    
    
    for itt = 1:options.maxiter
        start_it = tic;
        
        if toc(start) > options.time_limit
            disp('Time limit passed');
            break;
        end
        
        % Precomputation
        p  = zeros(L,r);
        S2 = S.^2;
        for l=1:L
            for k = 1:r
                p(l,k) = sum(w(l,:).^2 .* S2(l,v,k));
            end
        end
        dgS=zeros(L,r);
        for l=1:L
            S_l = squeeze(S(l,:,:));
            dgS(l,:) = diag(S_l);
        end 

        % UPDATE Z

        for i = randperm(n)
            % compute the coefficients for the L times r problems 
            b=zeros(L,r);
            c=zeros(L,r);
            for l=1:L
                X = X_list{l};
                S_l = squeeze(S(l,:,:));
                % b coefficients of the r problems  min_x ax^4+bx^2+cx
                b(l,:) = 2*(p(l,:)-(w(l,i)*S_l(v(i),:)).^2)-2*X(i,i)*dgS(l,:); % O(r)
    
                % c coefficients of the r problems  min_x ax^4+bx^2+cx
                [~, cols, vals] = find(X(i, :));
                mask = cols ~= i;
                cols_i = cols(mask);                % indices nonzeros de X(i,:) without i !
                xip    = vals(mask);                %  X values for non zeros entries indices X(i,:) without i !
                c(l,:)      = -4 * ( (xip(:)'.*w(l,cols_i)) * S_l( v(cols_i) , : ) );   % O( nnz dans X(i,:) )
            end 
            % Solve r problems  min_x ax^4+bx^2+cx
            best_f = inf; best_x=zeros(L) ; best_k = 1;
            x=zeros(L,1);
            for k = 1:r
                f=0;
                for l=1:L
                    if degrees(l,i) == 0
                        x(l) = 0;
                        fl = 0;
                    else
                        [x(l),fl] = cardan_depressed(4*S2(l,k,k),2*b(l,k),c(l,k),0);
                    end 
                    f=f+fl;
                    
                end 
                if f < best_f
                    best_f = f; best_x = x; best_k = k;
                end
            end

            % Update of p before updating w(i) (O(r))
            for l=1:L
                S_l = squeeze(S(l,:,:));
                p(l,:) = p(l,:) - (w(l,i)*S_l(v(i),:)).^2 + (best_x(l)*S_l(best_k,:)).^2;
            end 
            % Update w(i)
            w(:,i) = best_x;
            v(i) = best_k;
        end

        % Normalization of wl 
        for l = 1:L
            colNorm = sqrt(accumarray(v(:),w(l,:).^2,[r 1],@sum,0));
            denom = colNorm(v).';
            mask = denom ~= 0;
            w(l,mask) = w(l,mask)./denom(mask);
            w(l,~mask) = 0;
        end 

        % Construction of Sl
        for l = 1:L
            [i, j, val] = deal(I{l}, J{l}, VAL{l});
    
            rows   = v(i(:));
            cols   = v(j(:));
            values = w(l,i(:)).' .* w(l,j(:)).' .* val(:);
    
            S(l,:,:) = accumarray([rows(:),cols(:)], values(:), [r,r], @sum, 0);
        
    
        end 
    
        error_pre = error;
        % UPDATE error 
        error=0;
        for l=1:L
            error = error + sqrt(1e-9+normX2(l)-norm(S(l,:,:),'fro')^2);
        end 
        error = error/sum(normX);
        time{end+1} = toc(start_it);
        if error < options.delta
            break;
        end
        if abs(error_pre-error) < options.delta
            break;
        end
    end

    time_iteration{end+1} = time;
    time_global = toc(start);

    if error<=error_best
        w_best = w;
        v_best = v;
        S_best = S;
        error_best = error;
        if error_best <= options.delta
            break;
        end 
         if toc(start) > options.time_limit
            if options.verbosity > 0
                fprintf('Time_limit reached \n')
            end 
            break;
        end
    end 
    if options.verbosity > 0
        if itt == options.maxiter
                fprintf('Not converged \n')
        end 
        fprintf('Trial %u of %u with %s : %2.4e | Best: %2.4e \n',...
            trials, options.numTrials, init_algo, error, error_best);
            
    end
    
end 
end

function [x_opt,f_opt] = cardan_depressed(c3,c1,c0,default_x)
% find x that min c3/4*x^4+c1/2*x^2+c0*x and x>0
% default value otherwise
% solve by cardano formula 

    x_opt=default_x;
    f_opt=(c3/4)*(x_opt^4)+(c1/2)*(x_opt^2)+c0*x_opt;

    if abs(c3)<1e-12
        if abs(c1)>1e-12
            x = -c0/c1;
            if x>0
                f=(c3/4)*x^4+(c1/2)*x^2+c0*x;
                if f<f_opt
                    x_opt=x;
                    f_opt=f;
                end
            end

            
        end 
    else

        %Si c2=0
        np=c1/c3;
        nq=c0/c3;

        Delta = 4*np^3+27*nq^2;
        d     = 0.5*(-nq+sqrt(Delta/27));

        % For values where Delta is <= 0
        if Delta <=0 %-> 3 solutions réelles distinctes ou une solution multiple mais toutes réelles
            r3        = 2*(abs(d)^(1/3)); %racine cubique du module de d multipliée par 2
            th3       = (atan2(imag(d),real(d)))/3; %angle(dneg)/3;       %argument de d divisé par 3
            x         = (r3)*[cos(th3) cos(th3+(2*pi/3)) cos(th3+(4*pi/3))];
            x_pos = x(x > 0);
            if ~isempty(x_pos)
                f= (c3/4)*x_pos.^4 + (c1/2)*x_pos.^2 + c0*x_pos;
                [f_s, idx] = min(f);
                x_s = x_pos(idx);
                if f_s<f_opt
                    x_opt=x_s;
                    f_opt=f_s;
                end


                
            end 
            



        else
            d2pos     = 0.5*(-nq-sqrt(Delta/27));
            x      = sign(d)*(abs(d))^(1/3) + sign(d2pos)*(abs(d2pos))^(1/3);
            if x>0
                f=(c3/4)*x^4+(c1/2)*x^2+c0*x;
                if f<f_opt
                    x_opt=x;
                    f_opt=f;
                end
            end
        end
    end
end

function [v] = community_detection_SVCA(X,r)
    [n,~] = size(X);
    % Estimation of ZO=ZS by SSPA
    p = max(2,floor(0.1*n/r));
    options1.average = 1;
    [ZO,~] = SSPA(X,r,p,options1);
    % Compute Z s.t. min ||X-ZO*Z'||_F Z'Z=I
    norm2x = sqrt(sum(X.^2, 1));
    Xn = X .* (1 ./ (norm2x + 1e-16));
    HO = orthNNLS(X, ZO, Xn);
    Z = HO';
    % Compute v and w given Z
    v = max ((Z ~= 0).*(1:r),[],2);
    zero_idx = find(v == 0);
    v(zero_idx) = randi(r, size(zero_idx)); 
  
end


