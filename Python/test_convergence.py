import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import normalized_mutual_info_score
from frost import frost_multilayer
import matplotlib.pyplot as plt
import json
from pathlib import Path

def build_cora_multilayer(content_path, cites_path, k=20):

    # -------------------------
    # 1. Load content file
    # -------------------------
    content = pd.read_csv(
        content_path,
        sep=r"\s+",
        header=None
    )

    paper_id = content.iloc[:, 0].astype(str).values
    labels_raw = content.iloc[:, -1].values

    X = content.iloc[:, 1:-1].to_numpy(dtype=float)

    # -------------------------
    # KEEP ONLY 3 CLASSES
    # -------------------------
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

    # -------------------------
    # 2. CITATION LAYER
    # -------------------------
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

    # -------------------------
    # 3. SIMILARITY LAYER
    # -------------------------

    # Row-wise L2 norm
    norm_X = np.sqrt(np.sum(X**2, axis=1))

    # Avoid division by zero
    norm_X[norm_X == 0] = 1

    X_norm = X / norm_X[:, None]

    # Cosine similarity
    sim = X_norm @ X_norm.T

    # Remove self-similarity
    np.fill_diagonal(sim, 0)

    # -------------------------
    # 4. kNN GRAPH
    # -------------------------
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

    # -------------------------
    # 5. RETURN
    # -------------------------

    A = [
        adj_citation,
        adj_similarity
    ]

    return {
        "A": A,
        "labels": labels,
    }

def build_caltech(labels_file, edges_file):
    
    # ---- Read labels ----
    labels_data = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None
    )
    
    # column 1 = node id, column 2 = label
    labels = labels_data.iloc[:, 1].to_numpy()
    
    n = len(labels)
    
    
    # ---- Read edges ----
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
    
    
    # ---- Build adjacency matrices ----
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

def test_convergence(A_list,r,labels):
    w_best, v_best, S_best, error_best,convergence_data= frost_multilayer.frost_multilayer(A_list,r,init_method='USENC',numTrials=10,maxiter=30,delta=None,time_limit=30000,convergence_test=True,verbosity=1,true_labels=labels)
    w_best_random, v_best_random, S_best, error_best,convergence_data_random= frost_multilayer.frost_multilayer(A_list,r,init_method='random',numTrials=10,maxiter=30,delta=None,time_limit=30000,convergence_test=True,verbosity=1,true_labels=labels)

    output_dir = Path("convergence_data")
    output_dir.mkdir(exist_ok=True)

    for trial, trial_results in enumerate(convergence_data):
        file_path = output_dir / f"trial_init{trial + 1}.json"

        with open(file_path, "w") as f:
            json.dump(trial_results, f)

    for trial, trial_results in enumerate(convergence_data_random):
        file_path = output_dir / f"trial_rdm{trial + 1}.json"

        with open(file_path, "w") as f:
            json.dump(trial_results, f)
    plot_dir = Path("convergence_plots")
    plot_dir.mkdir(exist_ok=True)
    quantities = {
    "error": "Reconstruction error",
    "Z_change": "Relative change in Z",
    "n_changed": "Number of reassigned nodes",
    "NMI": "NMI",
    "ARI":"ARI"
    }

    results = {
    "initialization": {},
    "random": {}
    }

    for quantity, ylabel in quantities.items():

        # Standard initialization
        values_init = np.array([
            trial_results[quantity]
            for trial_results in convergence_data
        ])

        mean_init = []
        std_init = []
        min_init = []
        max_init = []
        mean_random = []
        std_random = []
        min_random = []
        max_random = []

        for col in zip(*values_init):
            if any(v is None for v in col):
                mean_init.append(None)
                std_init.append(None)
                min_init.append(None)
                max_init.append(None)
            else:
                mean_init.append(np.mean(col))
                std_init.append(np.std(col, ddof=1))
                min_init.append(np.min(col))
                max_init.append(np.max(col))

        results["initialization"][quantity] = {
            "mean": mean_init,
            "std": std_init,
            "min": min_init,
            "max": max_init
        }

        # Random initialization
        values_random = np.array([
            trial_results[quantity]
            for trial_results in convergence_data_random
        ])

        for col in zip(*values_random):
            if any(v is None for v in col):
                mean_random.append(None)
                std_random.append(None)
                min_random.append(None)
                max_random.append(None)
            else:
                mean_random.append(np.mean(col))
                std_random.append(np.std(col, ddof=1))
                min_random.append(np.min(col))
                max_random.append(np.max(col))

        results["random"][quantity] = {
            "mean": mean_random,
            "std": std_random,
            "min": min_random,
            "max": max_random
        }

    for init_name, data in results.items():
        df = pd.DataFrame({
            "iteration": range(1, len(data["error"]["mean"]) + 1),
            "error_mean": data["error"]["mean"],
            "error_std": data["error"]["std"],
            "error_min": data["error"]["min"],
            "error_max": data["error"]["max"],
            "NMI_mean": data["NMI"]["mean"],
            "NMI_std": data["NMI"]["std"],
            "NMI_min": data["NMI"]["min"],
            "NMI_max": data["NMI"]["max"],
        })

        df.to_csv(f"convergence_{init_name}.csv", index=False)
        

        

    for data in [convergence_data_random, convergence_data]:

        last_nmi = [trial["NMI"][-1] for trial in data]

        mean_nmi = np.mean(last_nmi)
        std_nmi = np.std(last_nmi)

        print("Mean NMI:", mean_nmi)
        print("Std NMI:", std_nmi)  

    for quantity, ylabel in quantities.items():

        plt.figure(figsize=(7, 5))

        for trial, trial_results in enumerate(convergence_data):
            values = trial_results[quantity]
            values_rdm=convergence_data_random[trial][quantity]
            iterations = range(1, len(values) + 1)

            plt.plot(
                iterations,
                values,
                label=f"Trial init{trial + 1}"
            )
            plt.plot(
                iterations,
                values_rdm,
                label=f"Trial rdm{trial + 1}",
                linestyle="--"
                        )

        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
        plot_dir / f"{quantity}_convergence.png",
        dpi=300,
        bbox_inches="tight"
        )
        plt.close()
    for quantity, ylabel in quantities.items():
    
            plt.figure(figsize=(7, 5))
            if quantity in ["error","NMI","ARI"]:
                            iterations = np.arange(0, len(mean_init))
            else:
                            iterations = np.arange(1, len(mean_init))
    
            mean_init = np.asarray(results["initialization"][quantity]["mean"], dtype=float)[iterations]
            std_init = np.asarray(results["initialization"][quantity]["std"], dtype=float)[iterations]
            mean_random = np.asarray(results["random"][quantity]["mean"], dtype=float)[iterations]
            std_random = np.asarray(results["random"][quantity]["std"], dtype=float)[iterations]
                
    
            plt.plot(
                iterations,
                mean_init,
                label="SVCA-USENC"
            )
    
            plt.fill_between(
                iterations,
                mean_init-std_init,
                mean_init+std_init,
                alpha=0.2
            )
    
            # -----------------
            # Random
            # -----------------
            
    
            plt.plot(
                iterations,
                mean_random,
                label="Random",
                linestyle="--"
            )
    
            plt.fill_between(
                iterations,
                mean_random-std_random,
                mean_random+std_random,
                alpha=0.2
            )
    
            plt.xlabel("Iteration")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
    
            plt.savefig(
                plot_dir / f"{quantity}_convergence_mean.png",
                dpi=300,
                bbox_inches="tight"
            )
    
            plt.close()




    return v_best
if __name__ == "__main__":

    data = build_caltech("../Data/caltech_all/labels.txt", "../Data/caltech_all/edges.txt")
    #data= build_cora_multilayer("../Data/cora/cora.content","../Data/cora/cora.cites",k=20)
    labels = np.array(data["labels"])
    A_list = data["A"]
   
    np.random.seed(42)

    v= test_convergence(A_list,max(labels), labels)
    nmi = normalized_mutual_info_score(labels, v)
    print(f"NMI = {nmi:.4f}")