from dataset import Dataset
import numpy as np
from typing import List, Tuple
import tqdm, pickle
import seaborn as sns
import matplotlib.pyplot as plt
from utils import Utils
from pathlib import Path

class HMM:
    def __init__(self, dataset: Dataset, fold_index: int):
        self.S = [".", "ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", "X"]    
        self.hmm_path = Path(Path.cwd(), f"Assignment1/outputs/HMM/trained_hmm_model_{fold_index}.pkl")
        if self.hmm_path.exists():
            obj = pickle.load(open(self.hmm_path, "rb"))
            assert obj["fold_index"] == fold_index, "Fold index should match!"
            self.__dict__.update(obj)
            print(f"The trained HMM model is loaded from {self.hmm_path}")
        else:
            Path.mkdir(self.hmm_path.parent, parents=True, exist_ok=True)
            self.fold_index = fold_index
            self.dataset = dataset
            self._train()
            pickle.dump(self.__dict__, open(self.hmm_path, "wb"))
            print(f"The trained HMM model is saved to {self.hmm_path}")
    
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
    
    def get_confusion_matrix(self) -> np.array:
        test_dl = Dataset.prepare_dataloader(data=self.test_data)
        words_dl = test_dl["input_words"]
        tags_dl = test_dl["target_tags"]
        tag_conf_matrix = {tt: {tp: 0 for tp in self.S} for tt in self.S} # Matrix[Target, Pred]

        with tqdm.tqdm(iterable=zip(words_dl, tags_dl), desc=f"Evaluating test sents for fold {self.fold_index}", total=len(words_dl), unit=" sents", colour="green") as pbar:
            for input_word_seq, target_tag_seq in pbar:
                pred_tag_seq = self.viterbi_decode(word_seq=input_word_seq)
                res = self._evaluate_sent(pred_tag_sent=pred_tag_seq, target_tag_sent=target_tag_seq, tag_conf_matrix=tag_conf_matrix)
                pbar.set_postfix(prediction=res)
        
        M = np.zeros(shape=(len(self.S), len(self.S)), dtype=np.int64)
        for i, tt in enumerate(self.S):
            for j, tp in enumerate(self.S):
                M[i,j] = tag_conf_matrix[tt][tp]  
        return M
                
def eval():
    num_folds = 5
    dataset = Dataset(num_folds=num_folds, unk_threshold=2)
    M_list = []
    for fold_index in range(num_folds):
        hmm = HMM(fold_index=fold_index, dataset=dataset)
        M = hmm.get_confusion_matrix()
        M_list.append(M)
    M = np.stack(M_list, axis=0).sum(axis=0)
    Utils.save_metrics(conf_matrix=M, labels=hmm.S, save_dir="Assignment1/outputs/HMM", name="HMM")

def main():
    num_folds = 5
    dataset = Dataset(num_folds=num_folds, unk_threshold=2)
    hmm = HMM(fold_index=0, dataset=dataset)
    sent = "Janet will back the bill .".split(" ")
    print(sent)
    print(hmm.viterbi_decode(word_seq=sent))

if __name__ == "__main__":
    main()