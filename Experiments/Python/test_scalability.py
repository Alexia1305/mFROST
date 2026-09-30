import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix 
from sklearn.metrics import normalized_mutual_info_score
from mfrost import mfrost
import matplotlib.pyplot as plt
import json
from pathlib import Path
import csv


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

def read_highschool2011(contact_file, metadata_file):
    contacts = pd.read_csv(
        contact_file,
        sep=r"\s+",
        header=None,
        names=["time", "id1", "id2", "class1", "class2"],
        dtype={
            "time": float,
            "id1": str,
            "id2": str,
            "class1": str,
            "class2": str,
        },
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=["NA"],
    )

    metadata = pd.read_csv(
        metadata_file,
        sep=r"\s+",
        header=None,
        names=["id", "class", "gender"],
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=["NA"],
    )

    if contacts.empty or metadata.empty:
        raise ValueError("Les fichiers ne doivent pas être vides.")

    if not np.isfinite(contacts["time"].to_numpy()).all():
        raise ValueError("Les timestamps doivent être finis.")

    if metadata["id"].duplicated().any():
        raise ValueError("Les identifiants des métadonnées doivent être uniques.")

    if metadata[["id", "class"]].isna().any().any():
        raise ValueError("Les identifiants et classes ne doivent pas être manquants.")

 
    node_ids = metadata["id"].to_numpy()
    n = len(node_ids)
    node_index = pd.Index(node_ids)

    i = node_index.get_indexer(contacts["id1"])
    j = node_index.get_indexer(contacts["id2"])

    if (i < 0).any() or (j < 0).any():
        raise ValueError(
            "Certains nœuds des contacts sont absents des métadonnées."
        )

    timestamps = pd.to_datetime(
        contacts["time"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("Europe/Paris")

    contact_days = timestamps.dt.strftime("%Y-%m-%d").to_numpy()
    days = sorted(np.unique(contact_days))


    Adj_list = []

    for day in days:
        mask = (contact_days == day) & (i != j)
        rows = np.concatenate([i[mask], j[mask]])
        cols = np.concatenate([j[mask], i[mask]])

        A = coo_matrix(
            (np.ones(len(rows), dtype=float), (rows, cols)),
            shape=(n, n),
        ).tocsr()


        A.data.fill(1.0)
        Adj_list.append(A)

    node_classes = metadata["class"].str.strip()
    is_teacher = node_classes.str.lower().isin(["teacher", "teachers"])
    node_classes = node_classes.mask(is_teacher, "teacher")

    classes = sorted(node_classes.unique())
    class_to_label = {
        class_name: label
        for label, class_name in enumerate(classes, start=1)
    }
    labels = node_classes.map(class_to_label).to_numpy(dtype=int)

    return {
        "A": Adj_list,
        "labels": labels
    }

def build_cora_multilayer(content_path, cites_path, k=20):

   
    # Load content file
   
    content = pd.read_csv(
        content_path,
        sep=r"\s+",
        header=None
    )

    paper_id = content.iloc[:, 0].astype(str).values
    labels_raw = content.iloc[:, -1].values

    X = content.iloc[:, 1:-1].to_numpy(dtype=float)

   
    # KEEP ONLY 3 CLASSES
   
    keep_classes = [
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods"
    ]

    keep_idx = np.isin(labels_raw, keep_classes)

    paper_id = paper_id[keep_idx]
    labels_raw = labels_raw[keep_idx]
    X = X[keep_idx, :]

    # Convert labels to 1..3
    label_mapping = {
        class_name: i + 1
        for i, class_name in enumerate(keep_classes)
    }

    labels = np.array([
        label_mapping[label]
        for label in labels_raw
    ])
    labels = labels.tolist()

    n = len(paper_id)

   
    # CITATION LAYER
   
    cites = pd.read_csv(
        cites_path,
        sep=r"\s+",
        header=None,
        names=["cited", "citing"]
    )

    cites["cited"] = cites["cited"].astype(str)
    cites["citing"] = cites["citing"].astype(str)

    # Initialize adjacency matrix
    adj_citation = np.zeros((n, n), dtype=int)

    # Mapping paper IDs -> matrix indices
    paper_to_idx = {
        paper: i
        for i, paper in enumerate(paper_id)
    }

    # Filter edges to kept nodes
    cites = cites[
        cites["citing"].isin(paper_to_idx)
        & cites["cited"].isin(paper_to_idx)
    ]

    # Add citation edges
    for _, row in cites.iterrows():

        citing_idx = paper_to_idx[row["citing"]]
        cited_idx = paper_to_idx[row["cited"]]

        adj_citation[citing_idx, cited_idx] = 1
        adj_citation[cited_idx, citing_idx] = 1

   
    # SIMILARITY LAYER
   

    # Row-wise L2 norm
    norm_X = np.sqrt(np.sum(X**2, axis=1))

    # Avoid division by zero
    norm_X[norm_X == 0] = 1

    X_norm = X / norm_X[:, None]

    # Cosine similarity
    sim = X_norm @ X_norm.T

    # Remove self-similarity
    np.fill_diagonal(sim, 0)

   
    # kNN GRAPH
   
    adj_similarity = np.zeros((n, n), dtype=int)

    for i in range(n):

        # Same logic as:
        # order(sim[i, ], decreasing = TRUE)[1:min(k, n)]
        top_k = np.argsort(sim[i])[::-1][:min(k, n)]

        adj_similarity[i, top_k] = 1

    # Symmetrize
    adj_similarity = np.maximum(
        adj_similarity,
        adj_similarity.T
    )

    # Remove diagonal just in case
    np.fill_diagonal(adj_similarity, 0)


    A = [
        adj_citation,
        adj_similarity
    ]

    return {
        "A": A,
        "labels": labels,
    }
 
def build_caltech(labels_file, edges_file):
    
    #  Read labels 
    labels_data = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None
    )
    
    # column 1 = node id, column 2 = label
    labels = labels_data.iloc[:, 1].to_numpy()
    
    n = len(labels)
    
    
    #  Read edges 
    edges = pd.read_csv(
        edges_file,
        sep=r"\s+",
        header=None
    )
    
    # columns:
    # column 1 = layer
    # column 2 = node i
    # column 3 = node j
    
    num_layers = edges.iloc[:, 0].max()
    
    A_list = []
    
    
    #  Build adjacency matrices 
    for l in range(1, num_layers + 1):
        
        edges_l = edges[edges.iloc[:, 0] == l]
        
        i = edges_l.iloc[:, 1].to_numpy() - 1
        j = edges_l.iloc[:, 2].to_numpy() - 1
        
        # Undirected graph
        rows = np.concatenate([i, j])
        cols = np.concatenate([j, i])
        
        data = np.ones(len(rows))
        
        A = coo_matrix(
            (data, (rows, cols)),
            shape=(n, n)
        ).tocsr()
        
        # Remove duplicates / set values to 1
        A.data[:] = 1
        
        A_list.append(A)
    
    
    return {
        "labels": labels,
        "A": A_list
    }

def test_scalability(A_list,r,labels):
    w_best, v_best, S_best, error_best,convergence_data= mfrost.mfrost(A_list,r,init_method='USENC',numTrials=10,convergence_test=True,true_labels=labels)
    
    output_dir = Path("scalability_data")
    output_dir.mkdir(exist_ok=True)

    for trial, trial_results in enumerate(convergence_data):
        file_path = output_dir / f"trial_{trial + 1}.json"

        with open(file_path, "w") as f:
            json.dump(trial_results, f)

    output_file = output_dir / "results.txt"

    with open(output_file, "w") as f:

        f.write("Convergence results\n")
        f.write("===================\n\n")

        # Final NMI, ARI and Time
        for quant in ["NMI", "ARI", "time"]:

            values = [trial[quant][-1] for trial in convergence_data]

            mean_value = np.mean(values)
            std_value = np.std(values, ddof=1)

            f.write(
                f"{quant}: {mean_value:.4f} ± {std_value:.4f}\n"
            )

        # Number of iterations
        n_iterations = [
            len(trial["NMI"])
            for trial in convergence_data
        ]

        mean_iterations = np.mean(n_iterations)
        std_iterations = np.std(n_iterations, ddof=1)

        f.write(
            f"Iterations: {mean_iterations:.2f} ± {std_iterations:.2f}\n"
        )

        print(f"Results saved to {output_file}")



if __name__ == "__main__":

    #data = build_caltech("../Data/caltech_all/labels.txt", "../Data/caltech_all/edges.txt")
    #data= build_cora_multilayer("../Data/cora/cora.content","../Data/cora/cora.cites",k=20)
    data = read_highschool2011("../Data/HighSchool/thiers_2011.csv","../Data/HighSchool/highschool_2011_metadata.txt")
    #data=build_AUCS("../Data/AUCS/aucs_edgelist.txt","../Data/AUCS/aucs_nodelist.txt")
    labels = np.array(data["labels"])
    A_list = data["A"]
    np.random.seed(101)
    print(int(max(labels)))
    test_scalability(A_list,int(max(labels)), labels)
