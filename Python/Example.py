import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from mfrost import mfrost
import matplotlib.pyplot as plt
from mfrost import mfrost
from scipy.sparse import csr_matrix

# Build the multilayer AUCS network 
def build_AUCS(edge_file, node_file):

    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)

    node_names = nodes["node"].tolist()
    N = len(node_names)

    node_index = {node: i for i, node in enumerate(node_names)}

    layers = edges["layer"].unique()

    adjacency_list = []

    for layer_name in layers:

        e = edges[edges["layer"] == layer_name]

        rows = []
        cols = []

        for _, row in e.iterrows():

            i = node_index[row["source"]]
            j = node_index[row["target"]]

            rows.extend([i, j])
            cols.extend([j, i])

        data = np.ones(len(rows), dtype=int)

        A = csr_matrix((data, (rows, cols)), shape=(N, N))
        A.data = np.ones_like(A.data)

        adjacency_list.append(A)

    groups = (
    nodes["group"]
    .str.replace("G", "", regex=False)
    .astype(float)
    .to_numpy()
)

    return {
        "A": adjacency_list,
        "labels": groups
    }
   

if __name__ == "__main__":

    # Build the multilayer AUCS network 
    data=build_AUCS("Data/AUCS/aucs_edgelist.txt","Data/AUCS/aucs_nodelist.txt")
    labels = np.array(data["labels"])
    A_list = data["A"]
    np.random.seed(12)
    r=int(max(labels))
    # mFROST returns v the community assignments (v[i] is the community of the node i)

    w, v, S, error_best= mfrost.mfrost(A_list,r)

    # Normalized mutual information for nodes with a label (not Nan)
    indices = np.where(~np.isnan(labels))[0]
    nmi = normalized_mutual_info_score(labels[indices], v[indices])
    print(f"NMI = {nmi:.4f}")


    # Plot the multilayer network
    L = len(A_list)
    n = len(v)

   
    n_cols = int(np.ceil(np.sqrt(r)))
    spacing = 4

    x = np.zeros(n)
    y = np.zeros(n)

    rng = np.random.default_rng(42)  # Reproducible positions

    for k in range(r):
        nodes_k = np.where(v == k)[0]

        row = (k - 1) // n_cols
        col = (k - 1) % n_cols

        center_x = spacing * col
        center_y = -spacing * row

        
        x[nodes_k] = center_x + 0.7 * rng.standard_normal(len(nodes_k))
        y[nodes_k] = center_y + 0.7 * rng.standard_normal(len(nodes_k))

    
    pos = {i: (x[i], y[i]) for i in range(n)}

    # One figure per layer
    for l in range(L):

        plt.figure()

        G = nx.from_scipy_sparse_array(A_list[l])

        nx.draw(
            G,
            pos=pos,
            node_color=v,
            cmap=plt.cm.tab10,
            node_size=70,
            edge_color="lightgray",
            width=0.5,
            with_labels=False
        )

        plt.title(f"Layer {l + 1}")
        plt.axis("off")
        plt.savefig(
        f"layer_{l + 1}.png",
        dpi=300,
        bbox_inches="tight"
    )
       
        plt.show()