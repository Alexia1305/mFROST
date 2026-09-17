clear all; clc; close all;
%% Example of community detection on AUCS 

%% load AUCS network 

[A_list, labels] = build_AUCS("data/AUCS/aucs_edgelist.txt", "data/AUCS/aucs_nodelist.txt");
r=max(labels);
L=numel(A_list);

%% Community detection with mFROST 
% INPUT: A_list is a cell array with the adjacency
% matrices of each layer, r is the number of communities we want 
% OUTPUT: v a vector giving the community of each node
[w,v,S,erreur,  ~,~] = mfrost(A_list,r,'verbosity',1);


%% Plot the multilayer network 
L = numel(A_list);
r = max(v);
n=length(v);
community_colors = lines(r);

% Community centers arranged on a grid
nCols = ceil(sqrt(r));
spacing = 4;

x = zeros(n, 1);
y = zeros(n, 1);

rng(42);  % Reproducible positions

for k = 1:r
    nodes_k = find(v == k);

    row = floor((k - 1) / nCols);
    col = mod(k - 1, nCols);

    center_x = spacing * col;
    center_y = -spacing * row;

    % Scatter nodes around their community center
    x(nodes_k) = center_x + 0.7 * randn(numel(nodes_k), 1);
    y(nodes_k) = center_y + 0.7 * randn(numel(nodes_k), 1);
end

% One figure per layer
for l = 1:L
    figure;

    G = graph(A_list{l});

    plot(G, ...
        "XData", x, ...
        "YData", y, ...
        "NodeCData", v(:), ...
        "MarkerSize", 7, ...
        "EdgeColor", [0.75 0.75 0.75], ...
        "LineWidth", 0.5);

    colormap(community_colors);
    clim([0.5, r + 0.5]);

    title(sprintf("Layer %d", l));
    axis equal off;
end

  
function [A_list, labels] = build_AUCS(edge_file, node_file)

edges = readtable(edge_file, TextType="string");
nodes = readtable(node_file, TextType="string");

node_names = nodes.node;
n = height(nodes);
layers = unique(edges.layer, "stable");
A_list = cell(numel(layers), 1);

for l = 1:numel(layers)
    e = edges(edges.layer == layers(l), :);

    [~, i] = ismember(e.source, node_names);
    [~, j] = ismember(e.target, node_names);

    % Undirected, binary sparse adjacency matrix
    A = sparse([i; j], [j; i], 1, n, n);
    A = spones(A);
    A = A - spdiags(diag(A), 0, n, n);

    A_list{l} = A;
end

% Convert G1, G2, ... to consecutive labels starting at 1.
% Nodes with group "NA" receive NaN.
group_strings = erase(nodes.group, "G");
groups = str2double(group_strings);

labels = nan(n, 1);
valid = ~isnan(groups);

valid_groups = unique(groups(valid), "stable");
[~, labels(valid)] = ismember(groups(valid), valid_groups);

end

