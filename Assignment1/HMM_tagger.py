import pickle
import math
from dataset import Dataset
import numpy as np
from typing import List, Tuple

class HMM:
    def __init__(self):
        self.dataset = Dataset()
    
    def train(self, fold_index: int) -> dict:
        train_data = self.dataset.get_train_test_dataset(fold_index=fold_index)["train"]
        params = self.dataset.get_prob_with_smoothing(train_data=train_data)
        return params
    
    def viterbi_decode(self, word_seq: List[str], params: dict) -> List[str]:
        if word_seq[0] != "<s>" and word_seq[-1] != "</s>":
            word_seq = ["<s>"] + word_seq + ["</s>"]
        V = params["V"] # list of words
        for i, w in enumerate(word_seq):
            if w not in V:
                word_seq[i] = "<unk>"
                
        n = len(word_seq) # num of observed words
        T = params["T"] # num of hidden tag states
        S = params["S"] # list of tag states
        A = params["transition"] # transition prob of tags P(tj | ti) => A(ti, tj)
        B = params["emission"] # emission prob of word P(wi | ti) => B(ti, wi)
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
                    
def main():
    hmm = HMM()
    params = hmm.train(fold_index=0)
    word_seq = "She is beautiful .".split(" ")
    out = hmm.viterbi_decode(word_seq=word_seq, params=params)
    print(word_seq)
    print(out)

if __name__ == "__main__":
    main()