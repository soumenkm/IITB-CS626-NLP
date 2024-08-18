import nltk
from typing import List, Tuple
from collections import defaultdict

class Dataset:
    def __init__(self, unk_threshold: int, num_folds: int):
        self.corpus = self._get_corpus()
        self.num_sents = len(self.corpus)
        self.k = num_folds
        self.unk_threshold = unk_threshold
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
    
    def _get_data_stream(self, train_data: List[List[Tuple[str, str]]]) -> dict:
        word_seq = []
        tag_seq = []
        word_count = defaultdict(int) # Get unigram word counts: C(wi) for all wi in Vocabulary
        for sent in train_data:
            for word, tag in sent:
                word_seq.append(word)
                tag_seq.append(tag)
                word_count[word] += 1
        word_seq = ["<unk>" if word_count[w] <= self.unk_threshold else w for w in word_seq] # Replace the rare words by <unk>
        return {"tag_seq": tag_seq, "word_seq": word_seq}
        
    def _get_counts(self, train_data: List[List[Tuple[str, str]]]) -> dict:
        data_stream = self._get_data_stream(train_data=train_data)
        tag_seq = data_stream["tag_seq"]
        word_seq = data_stream["word_seq"]
        S = sorted(list(set(tag_seq)))
        V = sorted(list(set(word_seq + ["<unk>"]))) # For safety purpose if all words are above unk threshold
        
        unigram_count = {ti: 0 for ti in S} # Get unigram state counts: C(ti) for all ti in S
        bigram_count = {(ti, tj): 0 for ti in S for tj in S} # Get bigram state counts: C(ti, tj) for all (ti, tj) in S x S 
        emission_count = {(ti, wi): 0 for ti in S for wi in V} # Get emission counts: C(ti, wi) for all (ti, wi) in S x V
        for i in range(len(tag_seq)):
            ti = tag_seq[i]
            wi = word_seq[i] 
            unigram_count[ti] += 1
            if i < len(tag_seq) - 1: 
                tj = tag_seq[i+1]
                bigram_count[(ti, tj)] += 1
            emission_count[(ti, wi)] += 1
        
        return {"S": S, "V": V, "ugc": unigram_count, "bgc": bigram_count, "emc": emission_count}
    
    def get_prob_with_smoothing(self, train_data: List[List[Tuple[str, str]]]) -> dict:
        counts = self._get_counts(train_data=train_data)
        S = counts["S"]
        V = counts["V"]
        unigram_count = counts["ugc"]
        bigram_count = counts["bgc"]
        emission_count = counts["emc"]
        
        bigram_prob = {key: 0.0 for key in bigram_count.keys()} # Get transition prob: P(tj | ti) for all (ti, tj) in S x S
        emission_prob = {key: 0.0 for key in emission_count.keys()} # Get emission prob: P(wi | ti) for all (ti, wi) in S x V
        for (ti, tj) in bigram_count.keys():
            bigram_prob[(ti, tj)] = (bigram_count[(ti, tj)] + 1) / (unigram_count[ti] + len(S)) # Laplace smoothing
        for (ti, wi) in emission_count.keys():
            emission_prob[(ti, wi)] = (emission_count[(ti, wi)] + 1) / (unigram_count[ti] + len(V)) # Laplace smoothing
        
        # Fix the special probability due to smoothing
        for t in S:
            bigram_prob[(t, "<s>")] = 0.0
            bigram_prob[("</s>", t)] = 1.0 if t == "</s>" else 0.0
        bigram_prob[("<s>", "</s>")] = 0.0
        
        for w in V:
            emission_prob[("<s>", w)] = 1.0 if w == "<s>" else 0.0
            emission_prob["</s>", w] = 1.0 if w == "</s>" else 0.0
            emission_prob[w, "<s>"] = 1.0 if w == "<s>" else 0.0
            emission_prob[w, "</s>"] = 1.0 if w == "</s>" else 0.0
          
        # Check if the prob dist are valid
        a = sum([sum([v for k,v in bigram_prob.items() if k[0] == t]) for t in S])
        b = sum([sum([v for k,v in emission_prob.items() if k[0] == t]) for t in S])
        assert abs(a - len(S)) < 0.01, f"a ({a}) must be equal to {len(S)}"
        assert abs(b - len(S)) < 0.01, f"b ({b}) must be equal to {len(S)}"
        
        return {"S": S, "V": V, "transition": bigram_prob, "emission": emission_prob}

def main():
    dataset = Dataset(num_folds=5, unk_threshold=2)
    train_data = dataset.get_train_test_dataset(fold_index=0)["train"]
    b = dataset.get_prob_with_smoothing(train_data=train_data)
    print(b)
    
    print("DONE")

if __name__ == "__main__":
    main()