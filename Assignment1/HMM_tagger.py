from dataset import Dataset
import numpy as np
from typing import List, Tuple
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class HMM:
    def __init__(self, dataset: Dataset, fold_index: int):
        self.fold_index = fold_index
        self.dataset = dataset
        self._train()
        self.S = self.params["S"]
        self.mod_S = [t for t in self.S if t not in ["<s>", "</s>"]]
        self.M = self._get_confusion_matrix() # (|mod_S|, |mod_S|) -> excludes <s> and </s> tags

    def _train(self) -> None:
        data = self.dataset.get_train_test_dataset(fold_index=self.fold_index)
        self.train_data = data["train"]
        self.test_data = data["test"]
        self.params = self.dataset.get_prob_with_smoothing(train_data=self.train_data)

    def viterbi_decode(self, word_seq: List[str]) -> List[str]:
        if word_seq[0] != "<s>" and word_seq[-1] != "</s>":
            word_seq = ["<s>"] + word_seq + ["</s>"]
        V = self.params["V"] # list of words
        for i, w in enumerate(word_seq):
            if w not in V:
                word_seq[i] = "<unk>"
                
        n = len(word_seq) # num of observed words
        S = self.params["S"] # list of tag states
        T = len(S) # num of hidden tag states

        A = self.params["transition"] # transition prob of tags P(tj | ti) => A(ti, tj)
        B = self.params["emission"] # emission prob of word P(wi | ti) => B(ti, wi)
        DP = np.zeros(shape=(T, n))
        BP = np.zeros(shape=(T, n))
        
        # Initialization
        w0 = word_seq[0]
        for idx, s in enumerate(S):
            DP[idx, 0] = 1.0 * B[(s, w0)]
            BP[idx, 0] = -1
        
        # Memorization
        for i in range(1, n):
            for idx, s in enumerate(S):
                wi = word_seq[i]
                array = np.array([DP[j, i-1] * A[(sp, s)] * B[(s, wi)] for j, sp in enumerate(S)])
                max_index = int(array.argmax())
                DP[idx, i] = array[max_index]
                BP[idx, i] = max_index
        
        # Decoding
        tag_seq = [None] * len(word_seq)
        tag_seq[-1] = int(DP[:, -1].argmax())
        for i in range(n-1, 0, -1):
            tag_seq[i-1] = int(BP[tag_seq[i], i])
        
        return [S[t] for t in tag_seq]

    def _evaluate_sent(self, pred_tag_sent: List[str], target_tag_sent: List[str], tag_conf_matrix: dict) -> bool:
        lp = len(pred_tag_sent)
        lt = len(target_tag_sent)
        assert lp == lt, f"The length of pred (= {lp}) and target (= {lt}) tag sent must match!"
        
        is_correct = True
        for pred, target in zip(pred_tag_sent, target_tag_sent):
            if target not in ["<s>", "</s>"]:  # Ignore <s> and </s>
                tag_conf_matrix[target][pred] += 1
            if target != pred:  
                is_correct = False      
        return is_correct
    
    def _get_confusion_matrix(self) -> np.array:
        test_dl = Dataset.prepare_dataloader(data=self.test_data)
        words_dl = test_dl["input_words"]
        tags_dl = test_dl["target_tags"]
        tag_conf_matrix = {tt: {tp: 0 for tp in self.mod_S} for tt in self.mod_S} # Matrix[Target, Pred]

        with tqdm.tqdm(iterable=zip(words_dl, tags_dl), desc=f"Evaluating test sents for fold {self.fold_index}", total=len(words_dl), unit=" sents", colour="green") as pbar:
            for input_word_seq, target_tag_seq in pbar:
                pred_tag_seq = self.viterbi_decode(word_seq=input_word_seq)
                res = self._evaluate_sent(pred_tag_sent=pred_tag_seq, target_tag_sent=target_tag_seq, tag_conf_matrix=tag_conf_matrix)
                pbar.set_postfix(prediction=res)
        
        M = np.zeros(shape=(len(self.mod_S), len(self.mod_S)), dtype=np.int64)
        for i, tt in enumerate(self.mod_S):
            for j, tp in enumerate(self.mod_S):
                M[i,j] = tag_conf_matrix[tt][tp]  
        return M
    
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
    def _plot_conf_matrix(conf_matrix: np.array, labels: List[str], file_name: str) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix/conf_matrix.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted POS Tags")
        plt.ylabel("Actual POS tags")
        plt.title("Confusion Matrix (Normalized by 'True' Tags)")
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Confusion matrix is saved to {file_name}")
        
    @staticmethod
    def save_metrics(conf_matrix: np.array, labels: List[str], save_dir: str) -> None:
        HMM._plot_conf_matrix(conf_matrix=conf_matrix, labels=labels, file_name=save_dir + "/conf_matrix.png")
        metrics = HMM._get_metrics(conf_matrix=conf_matrix)

        # Figure 1: Precision and Recall
        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
        fig1.suptitle("Precision and Recall by HMM POS Tagger", fontsize=16)
        metric_names_1 = ["Precision", "Recall"]
        metric_keys_1 = ["precision", "recall"]
        
        for i, (ax, metric_name, metric_key) in enumerate(zip(axes1, metric_names_1, metric_keys_1)):
            values = metrics[metric_key]
            bars = ax.bar(labels, values, color=plt.cm.Paired(np.linspace(0, 1, len(labels))))
            ax.set_title(f"{metric_name} per POS Tag")
            ax.set_xlabel("POS Tag")
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='top', color='black')
        
        fig1.savefig(save_dir + "/precision_recall.png", bbox_inches='tight')
        
        # Figure 2: F-Scores
        fig2, axes2 = plt.subplots(1, 3, figsize=(24, 6))
        fig2.suptitle("F-Scores by HMM POS Tagger", fontsize=16)
        metric_names_2 = ["F1-Score", "F0.5-Score", "F2-Score"]
        metric_keys_2 = ["f1", "f0.5", "f2"]
        
        for i, (ax, metric_name, metric_key) in enumerate(zip(axes2, metric_names_2, metric_keys_2)):
            values = metrics[metric_key]
            bars = ax.bar(labels, values, color=plt.cm.Paired(np.linspace(0, 1, len(labels))))
            ax.set_title(f"{metric_name} per POS Tag")
            ax.set_xlabel("POS Tag")
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='top', color='black')
        
        fig2.savefig(save_dir + "/f_scores.png", bbox_inches='tight')

        # Figure 3: Weighted Metrics
        fig3, ax3 = plt.subplots(figsize=(6, 8))
        fig3.suptitle("Weighted Metrics by HMM POS Tagger", fontsize=16)
        weighted_metric_names = ["Wt Precision", "Wt Recall", "Wt F1", "Wt F0.5", "Wt F2"]
        weighted_values = [metrics[key] for key in ["wt_precision", "wt_recall", "wt_f1", "wt_f0.5", "wt_f2"]]
        
        bars = ax3.bar(weighted_metric_names, weighted_values, color=plt.cm.Paired(np.linspace(0, 1, len(weighted_metric_names))))
        ax3.set_xlabel("Metrics")
        ax3.set_ylabel("Value")
        ax3.set_ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 4), ha='center', va='top', color='black')
        
        fig3.savefig(save_dir + "/weighted_metrics.png", bbox_inches='tight')
                      
def main():
    num_folds = 5
    dataset = Dataset(num_folds=num_folds, unk_threshold=2)
    M_list = []
    for fold_index in range(num_folds):
        hmm = HMM(fold_index=fold_index, dataset=dataset)
        M_list.append(hmm.M)
    M = np.stack(M_list, axis=0).sum(axis=0)
    HMM.save_metrics(conf_matrix=M, labels=hmm.mod_S, save_dir="Assignment1/outputs")
    
if __name__ == "__main__":
    main()