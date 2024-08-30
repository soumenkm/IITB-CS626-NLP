import numpy as np
from typing import List, Tuple
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class Utils:
    
    @staticmethod
    def _get_metrics(conf_matrix: np.array) -> dict:
        weight = conf_matrix.sum(axis=1)
        weight = weight / weight.sum()
        acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
        recall = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
        prec = np.diag(conf_matrix) / conf_matrix.sum(axis=0)
        f1 = ((1+1**2) * prec * recall) / ((1**2 * prec) + recall)
        f0p5 = ((1+0.5**2) * prec * recall) / ((0.5**2 * prec) + recall)
        f2 = ((1+2**2) * prec * recall) / ((2**2 * prec) + recall)
        
        wt_recall = (np.diag(conf_matrix) / conf_matrix.sum(axis=1)) * weight
        wt_prec = (np.diag(conf_matrix) / conf_matrix.sum(axis=0)) * weight
        wt_f1 = (((1+1**2) * prec * recall) / ((1**2 * prec) + recall)) * weight
        wt_f0p5 = (((1+0.5**2) * prec * recall) / ((0.5**2 * prec) + recall)) * weight
        wt_f2 = (((1+2**2) * prec * recall) / ((2**2 * prec) + recall)) * weight
        
        return {"weight": weight, "acc": acc, "precision": prec, "recall": recall, "f1": f1, "f0.5": f0p5, "f2": f2,
                "wt_precision": wt_prec.sum(), "wt_recall": wt_recall.sum(), "wt_f1": wt_f1.sum(), "wt_f0.5": wt_f0p5.sum(), "wt_f2": wt_f2.sum()}
    
    @staticmethod
    def _plot_conf_matrix(conf_matrix: np.array, labels: List[str], file_name: str, name: str) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix/conf_matrix.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted POS Tags")
        plt.ylabel("Actual POS tags")
        plt.title(f"Confusion Matrix for {name} (Normalized by 'True' Tags)")
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Confusion matrix is saved to {file_name}")
        
    @staticmethod
    def save_metrics(conf_matrix: np.array, labels: List[str], save_dir: str, name: str) -> None:
        Utils._plot_conf_matrix(conf_matrix=conf_matrix, labels=labels, file_name=save_dir + f"/{name}_conf_matrix.png", name=name)
        metrics = Utils._get_metrics(conf_matrix=conf_matrix)

        # Figure 1: Precision and Recall
        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
        fig1.suptitle(f"Precision and Recall by {name} POS Tagger", fontsize=16)
        metric_names_1 = ["Precision", "Recall"]
        metric_keys_1 = ["precision", "recall"]
        
        for i, (ax, metric_name, metric_key) in enumerate(zip(axes1, metric_names_1, metric_keys_1)):
            values = metrics[metric_key]
            bars = ax.bar(labels, values)
            ax.set_title(f"{metric_name} per POS Tag for {name}")
            ax.set_xlabel("POS Tag")
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='top', color='black')
        
        fig1.savefig(save_dir + f"/{name}_precision_recall.png", bbox_inches='tight')
        
        # Figure 2: F-Scores
        fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))
        fig2.suptitle(f"F-Scores by {name} POS Tagger", fontsize=16)
        metric_names_2 = ["F1-Score", "F0.5-Score", "F2-Score"]
        metric_keys_2 = ["f1", "f0.5", "f2"]
        
        for i, (ax, metric_name, metric_key) in enumerate(zip(axes2, metric_names_2, metric_keys_2)):
            values = metrics[metric_key]
            bars = ax.bar(labels, values)
            ax.set_title(f"{metric_name} per POS Tag for {name}")
            ax.set_xlabel("POS Tag")
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='top', color='black')
        
        fig2.savefig(save_dir + f"/{name}_f_scores.png", bbox_inches='tight')

        # Figure 3: Weighted Metrics
        fig3, ax3 = plt.subplots(figsize=(6, 8))
        fig3.suptitle(f"Weighted Metrics by {name} POS Tagger", fontsize=16)
        weighted_metric_names = ["Precision", "Recall", "F1", "F0.5", "F2"]
        weighted_values = [metrics[key] for key in ["wt_precision", "wt_recall", "wt_f1", "wt_f0.5", "wt_f2"]]
        
        bars = ax3.bar(weighted_metric_names, weighted_values)
        ax3.set_xlabel("Metrics")
        ax3.set_ylabel("Value")
        ax3.set_ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 4), ha='center', va='top', color='black')
        
        fig3.savefig(save_dir + f"/{name}_weighted_metrics.png", bbox_inches='tight')
          