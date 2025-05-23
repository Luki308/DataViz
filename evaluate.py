import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import genieclust
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from typing import Union, Dict, List, Optional, Tuple, Literal
from io import StringIO

class ClusteringEvaluator:
    
    # stores scores from evaluation in dataframe with columns: battery, dataset, method, rand_score, silhoutte score, accuracy score
    results_df:pd.DataFrame = pd.DataFrame({
        'battery': pd.Series(dtype='str'),
        'dataset': pd.Series(dtype='str'),
        'method': pd.Series(dtype='str'),
        'n_clusters': pd.Series(dtype='int'),
        'rand_score': pd.Series(dtype='float'),
        'silhouette_score': pd.Series(dtype='float'),
        'NCA': pd.Series(dtype='float'),
    })


    def __init__(self):
        """
        Initialize the ClusteringEvaluator with the path to the clustering data.
        
        Args:
            data_path: Path to the clustering data directory
        """
        self.data_path = os.path.abspath("clustering-data-v1")
        self.methods = ['kmeans', 'dbscan', 'agglomerative', 'genie']
        self.g = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.linkage = ['ward', 'single', 'complete', 'average']
    
    def load_data(self, battery: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        self.b = clustbench.load_dataset(battery, dataset, path=self.data_path)
        self.data = self.b.data
        self.labels = self.b.labels
        self.n_clusters = self.b.n_clusters

    def get_clusterer(self, method: str, **kwargs) -> Union[KMeans, DBSCAN, AgglomerativeClustering]:
        """
        Get the clustering algorithm based on the method name.
        
        Args:
            method: Name of the clustering method
        
        Returns:
            An instance of the clustering algorithm
        """
        params = None
        if method == 'kmeans':
            return KMeans(), params
        elif method == 'dbscan':
            # DBSCAN requires eps and min_samples parameters
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            if eps != 0.5:
                params = {'eps': eps}
            elif min_samples != 5:
                params = {'min_samples': min_samples}
            elif eps != 0.5 and min_samples != 5:
                params = {'eps': eps, 'min_samples': min_samples}
            return DBSCAN(eps=eps, min_samples=min_samples), params
        elif method == 'agglomerative':
            linkage = kwargs.get('linkage', 'ward')
            if linkage != 'ward':
                params = {'linkage': linkage}
            return AgglomerativeClustering(linkage=linkage), params
        elif method == 'genie':
            gini_threshold = kwargs.get('gini_threshold', 0.3)
            if gini_threshold != 0.3:
                params = {'gini_threshold': gini_threshold}
            return genieclust.Genie(gini_threshold=gini_threshold), params
        else:
            raise ValueError(f"Unknown method: {method}")

    def do_all(self, battery: str, dataset: str, plot:bool) -> None:

        self.load_data(battery, dataset)
        print(f"Loaded data for {battery} - {dataset}")

        method_params = []
            # Create a list of (method, params_dict) tuples to iterate over
        for method in self.methods:
            if method == 'genie':
                # For genie, use different gini_threshold values
                for g in self.g:
                    method_params.append((method, {'gini_threshold': g}))
            elif method == 'agglomerative':
                # For agglomerative, use different linkage values
                for linkage in self.linkage:
                    method_params.append((method, {'linkage': linkage}))
            else:
                # For other methods, use default params
                method_params.append((method, {}))

        # Number of rows in the subplot grid should be based on method_params length
        num_rows = len(method_params) + 1  # +1 for the true labels row
        
        if plot:
            plt.figure(figsize=(10, 30))
            for i in range(len(self.labels)):
                plt.subplot(num_rows, len(self.labels), i + 1)
                genieclust.plots.plot_scatter(
                    self.data,
                    labels=self.labels[i]-1,
                    axis='equal',
                    title=f"True Labels (k = {self.n_clusters[i]}) ",
                )

        for iter, (method, params) in enumerate(method_params):
            # Results shoulf be stored in a dictionary with keys as the number of clusters and list of labels as values

            # Create the clusterer
            if method == 'dbscan':
                results = {}
                for i, k in enumerate(self.n_clusters):
                    clusterer, param_dict = self.get_clusterer(method, **params)
                    pred_labels = clusterer.fit_predict(self.data)+1
                    results[int(k)] = pred_labels
            else:
                clusterer, param_dict = self.get_clusterer(method, **params)
                results = clustbench.fit_predict_many(clusterer, self.data, self.n_clusters)

            # plot results vs true labels
            if plot:
                for i, k in enumerate(results):
                    # Update subplot position using num_rows instead of len(self.methods)+1
                    ax = plt.subplot(num_rows, len(results), (iter+1) * len(results) + i + 1)
                    
                    # Create method title with parameters
                    if method == 'genie' and 'gini_threshold' in params:
                        title = f"{method.capitalize()} (g={params['gini_threshold']}) k={k}"
                    elif method == 'agglomerative' and 'linkage' in params:
                        title = f"{method.capitalize()} ({params['linkage']}) k={k}"
                    else:
                        title = f"{method.capitalize()} Labels (k = {k})"
                    
                    genieclust.plots.plot_scatter(
                        self.data,
                        labels=results[k]-1,
                        axis='equal',
                        title=title,
                    )
                    
                    # The rest of your confusion matrix code remains the same
                    confusion_matrix = genieclust.compare_partitions.confusion_matrix(
                        self.labels[i], results[k]
                    )
                    cm_str = StringIO()
                    np.savetxt(cm_str, confusion_matrix, fmt='%d', delimiter=' | ', footer="\nTrue \\\\ Pred", comments = '')
                    cm_text = cm_str.getvalue()
                    ax.text(
                        0.95, 0.05, cm_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                    )

            # store results in dataframe
            for i, k in enumerate(results):
                df = pd.DataFrame({
                    "battery": battery,
                    "dataset": dataset,
                    "method": method,
                    "n_clusters": k,
                    "rand_score": adjusted_rand_score(self.labels[i], results[k]),
                    "silhouette_score": silhouette_score(self.data, results[k]),
                    # "NCA": clustbench.get_score(self.labels[i], results[k])
                    "NCA": genieclust.compare_partitions.normalized_clustering_accuracy(
                        self.labels[i], results[k]),
                    'params': 'default',
                }, index=[0])
                if params is not None:
                    for key, value in params.items():
                        df[key] = value

                self.results_df = pd.concat([self.results_df, df], ignore_index=True)
        if plot:
            plt.tight_layout()
            plt.show()

