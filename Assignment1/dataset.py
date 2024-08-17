import pickle, nltk
from typing import List, Tuple
from collections import defaultdict
import math

class Dataset:
    def __init__(self, num_folds=5):
        self.corpus = self._get_corpus()
        self.num_sents = len(self.corpus)
        self.k = num_folds
        self.dataset_folds = self._k_fold_cross_val(k=self.k)
    
    def _get_corpus(self) -> List[List[Tuple[str, str]]]:
        corpus = nltk.corpus.brown.tagged_sents(tagset='universal')
        pp_corpus = []
        for sent in corpus:
            item = [("<s>", "<s>")] + sent + [("</s>", "</s>")]
            pp_corpus.append(item)
        return pp_corpus
                
    def _k_fold_cross_val(self, k: int) -> List[dict]:
        fold_size = self.num_sents // k
        corpus = self.corpus[:fold_size * k]
        folds = [] # List[k x Dict{train, test}]
        for i in range(k):
            start = i * fold_size
            end = (i+1) * fold_size
            test_set = corpus[start:end]
            train_set = corpus[:start] + corpus[end:]
            folds.append({"train": train_set, "test": test_set})
        return folds
    
    def get_train_test_dataset(self, fold_index: int) -> dict:
        assert fold_index >= 0 and fold_index < self.k, f"fold_index (= {fold_index}) must be in between 0 to {self.k-1}!"
        return self.dataset_folds[fold_index]
    
    def _get_counts(self, train_data: List[List[Tuple[str, str]]]) -> dict:
        word_seq = []
        tag_seq = []
        for sent in train_data:
            for word, tag in sent:
                word_seq.append(word)
                tag_seq.append(tag)
        S = set(tag_seq)
        V = set(word_seq + ["<unk>"])
        
        unigram_count = defaultdict(int)
        bigram_count = defaultdict(int)
        emission_count = defaultdict(int)
        for i in range(len(tag_seq)):
            ti = tag_seq[i]
            wi = word_seq[i] 
          
            # Get unigram state counts: C(t_i) for all t_i in S
            unigram_count[ti] += 1

            # Get bigram state counts: C(t_i, t_j) for all (t_i, t_j) in S x S 
            if i < len(tag_seq) - 1: 
                tj = tag_seq[i+1]
                bigram_count[(ti, tj)] += 1
            
            # Get emission counts: C(w_i, t_i) for all (w_i, t_i) in V x S
            emission_count[(wi, ti)] += 1
            if ("<unk>", ti) not in emission_count:
                emission_count[("<unk>", ti)] = 0 # Handles unknown word emission
        
        # Ensures that all possible pair of bigram exists
        for ti in S:
            for tj in S:
                if (ti, tj) not in bigram_count:
                    bigram_count[(ti, tj)] = 0
        
        # Ensures that all possible pair of emission exists
        for wi in V:
            for ti in S:
                if (wi, ti) not in emission_count:
                    emission_count[(wi, ti)] = 0
        
        return {"S": S, "V": V, "ugc": unigram_count, "bgc": bigram_count, "emc": emission_count}
    
    def get_prob_with_smoothing(self, train_data: List[List[Tuple[str, str]]]) -> dict:
        counts = self._get_counts(train_data=train_data)
        S = counts["S"]
        V = counts["V"]
        unigram_count = counts["ugc"]
        bigram_count = counts["bgc"]
        emission_count = counts["emc"]
        
        bigram_prob = defaultdict(float)
        emission_prob = defaultdict(float)
        
        # Get bigram prob with Laplace smoothing
        for (ti, tj) in bigram_count.keys():
            bigram_prob[(ti, tj)] = (bigram_count[(ti, tj)] + 1) / (unigram_count[ti] + len(S))
        
        # Get emission prob with Laplace smoothing
        for (wi, ti) in emission_count.keys():
            emission_prob[(wi, ti)] = (emission_count[(wi, ti)] + 1) / (unigram_count[ti] + len(V))
            
        # Check if the prob dist are valid
        a = sum([sum([v for k,v in bigram_prob.items() if k[0] == t]) for t in S])
        b = sum([sum([v for k,v in emission_prob.items() if k[1] == t]) for t in S])
        assert abs(a - len(S)) < 1e-3, f"a ({a}) must be equal to {len(S)}"
        assert abs(b - len(S)) < 1e-3, f"b ({b}) must be equal to {len(S)}"
        
        return {"S": S, "V": V, "T": len(S), "W": len(V), "bgp": bigram_prob, "emp": emission_prob}

def main():
    dataset = Dataset()
    train_data = dataset.get_train_test_dataset(fold_index=0)["train"]
    b = dataset.get_prob_with_smoothing(train_data=train_data)
    print(b)
    
    print("DONE")

if __name__ == "__main__":
    main()