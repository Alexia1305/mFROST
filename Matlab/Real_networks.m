
clear all; clc; close all;
%% Experiments on real-world multilayer networks 


for network= ["AUCS","Cora","Citeseer","Caltech"]
    % load network 
    fprintf("Network: %s\n", network);
   
   
    rng(2);
    filename = fullfile("data", network + ".mat");
    data = load(filename);

    n = double(data.n);
    L = double(data.L);
    labels = double(data.labels(:));
    % some nodes have no labels 
    labelled= isfinite(labels) & labels > -1e9 & labels < 1e9; % None value 
    
    
    Adj_list = cell(L, 1);
    
    for l = 1:L
        i = double(data.(sprintf("i%d", l)));
        j = double(data.(sprintf("j%d", l)));
        x = double(data.(sprintf("x%d", l)));
    
        Adj_list{l} = sparse(i, j, x, n, n);
    end
       
    % Experiments: metrics of 10 restarts of mFROST
    n_restart=10;
    r=max(labels);
    
    %mFROST 
	[w,v,S,erreur,  ~,~] = mfrost(Adj_list,r,'verbosity',1);
	
	nmi= nmi(labels(labelled),v(labelled))

end 







