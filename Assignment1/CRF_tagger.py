import sklearn_crfsuite, string, tqdm, pickle, time
from sklearn_crfsuite import metrics
from dataset import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from typing import List, Tuple, Union
from utils import Utils

class CRF:
    def __init__(self, dataset: Union[Dataset, None], fold_index: int, data_frac: Union[float, None]):  
        self.S = [".", "ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", "X"]    
        self.crf_path = Path(Path.cwd(), f"Assignment1/outputs/CRF/trained_crf_model_{fold_index}.pkl")
        if self.crf_path.exists():
            obj = pickle.load(open(self.crf_path, "rb"))
            assert obj["fold_index"] == fold_index, "Fold index should match!"
            self.__dict__.update(obj)
            print(f"The trained CRF model is loaded from {self.crf_path}")
        else:
            Path.mkdir(self.crf_path.parent, parents=True, exist_ok=True)
            self.fold_index = fold_index
            self.data_frac = data_frac
            self.dataset = dataset  
            self.model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',  
                c1=0.1,            
                c2=0.1,            
                max_iterations=100, 
                all_possible_transitions=True
                )
            self._train()
            pickle.dump(self.__dict__, open(self.crf_path, "wb"))
            print(f"The trained CRF model is saved to {self.crf_path}")
    
    def _get_prefix_features(self, word: str) -> dict:
        prefixes = [
            'un', 're', 'in', 'im', 'dis', 'over', 'mis', 'sub', 'pre', 'inter', 'fore', 'de', 'trans', 'super', 'semi', 'anti', 'mid', 'under'
        ]
        word_lower = word.lower()  
        prefix_features = {}
        for prefix in prefixes:
            prefix_features[f'prefix_{prefix}'] = word_lower.startswith(prefix)
        return prefix_features

    def _get_suffix_features(self, word: str) -> dict:
        suffixes = [
            'ing', 'ly', 'ed', 'ious', 'es', 's', 'ment', 'tion', 'ness', 'ful', 'able', 'ible', 'al', 'er', 'ish', 'ism', 'ate', 'ify', 'ous', 'ive'
        ]
        word_lower = word.lower()  
        suffix_features = {}
        for suffix in suffixes:
            suffix_features[f'suffix_{suffix}'] = word_lower.endswith(suffix)
        return suffix_features
    
    def _get_basic_features(self, word: str) -> dict:
        stemmer = PorterStemmer()
        features = {
            'word': word,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.ispunct()': word in string.punctuation,
            'word.length': len(word),
            'word[-1:]': word[-1:],
            'word[-2:]': word[-2:],
            'word[-3:]': word[-3:],
            'word[:1]': word[:1],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            'is_noun_suffix': word.endswith(('s', 'es', 'tion', 'ment', 'ness', 'ity', 'ant')),
            'is_verb_suffix': word.endswith(('ed', 'ing', 'ize', 'ise', 'ify')),
            'is_adjective_suffix': word.endswith(('ous', 'al', 'ic', 'ive', 'y', 'able', 'ible')),
            'is_adverb_suffix': word.endswith('ly'),
            'root_word': stemmer.stem(word),  
            'has_inflection': any(x in word.lower() for x in ['ed', 's', 'ing']),  
            'is_compound': '-' in word or ' ' in word,  
        }
        features.update(self._get_prefix_features(word))
        features.update(self._get_suffix_features(word))
        return features
        
    def _get_local_features(self, x: List[str], t: int) -> None:
        word = x[t]
        features = self._get_basic_features(word)
        if len(x) > 2:
            if t > 1:
                features.update({f'prev2_{k}': v for k, v in self._get_basic_features(x[t-2]).items()})
            if t > 0:
                features.update({f'prev1_{k}': v for k, v in self._get_basic_features(x[t-1]).items()})
            if t < len(x) - 2:
                features.update({f'next2_{k}': v for k, v in self._get_basic_features(x[t+2]).items()})
            if t < len(x) - 1:
                features.update({f'next1_{k}': v for k, v in self._get_basic_features(x[t+1]).items()})
        features.update({"BOS": t == 0, "EOS": t == len(x) - 1})
        return features
    
    def _get_features_labels(self, data: List[List[Tuple[str, str]]]):
        X, y = [], []
        with tqdm.tqdm(iterable=data, desc=f"Preparing dataset for fold {self.fold_index}...", total=len(data), unit=" sents", colour="green") as pbar:
            for item in pbar:
                x_features = []
                word_seq = [w for w, t in item]
                tag_seq = [t for w, t in item]
                for t in range(len(word_seq)):
                    features = self._get_local_features(word_seq, t)
                    x_features.append(features)
                X.append(x_features)
                y.append(tag_seq)
        return X, y
    
    def _get_data_split(self, is_train: bool) -> Tuple[List]:
        split_type = "train" if is_train else "test"
        data = self.dataset.get_train_test_dataset(fold_index=self.fold_index)
        data = data[split_type] # List[List[Tuple[str, str]]] (N, T, 2)
        data = [i[1:-1] for i in data]
        index = int(len(data) * self.data_frac)
        data = data[:index]
        X, y = self._get_features_labels(data)
        if is_train:
            return X, y
        else:
            X = []
            y = []
            for item in data:
                word_seq = [w for w, t in item]
                tag_seq = [t for w, t in item]
                X.append(word_seq)
                y.append(tag_seq)
            return X, y
        
    def _train(self) -> None:
        print("Training the CRF model...")
        t1 = time.time()
        X_train, y_train = self._get_data_split(is_train=True)
        self.model.fit(X_train, y_train)
        t2 = time.time()
        print(f"Training of CRF model is completed in {(t2-t1)/60:.2f} mints")
    
    def decode(self, word_seq: List[str]) -> List[str]:
        word_features = [self._get_local_features(word_seq, i) for i in range(len(word_seq))]
        pos_tags = self.model.predict_single(word_features)
        return pos_tags
    
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
        X_test, y_test = self._get_data_split(is_train=False)
        tag_conf_matrix = {tt: {tp: 0 for tp in self.S} for tt in self.S} # Matrix[Target, Pred]

        with tqdm.tqdm(iterable=zip(X_test, y_test), desc=f"Evaluating test sents for fold {self.fold_index}", total=len(X_test), unit=" sents", colour="green") as pbar:
            for input_word_seq, target_tag_seq in pbar:
                pred_tag_seq = self.decode(word_seq=input_word_seq)
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
        crf = CRF(dataset=dataset, fold_index=fold_index, data_frac=1)
        M = crf.get_confusion_matrix()
        M_list.append(M)
    M = np.stack(M_list, axis=0).sum(axis=0)
    Utils.save_metrics(conf_matrix=M, labels=crf.S, save_dir="Assignment1/outputs/CRF", name="CRF")

def main():
    num_folds = 5
    dataset = Dataset(num_folds=num_folds, unk_threshold=2)
    crf = CRF(dataset=dataset, fold_index=0, data_frac=1)
    sent = "Janet will back the bill .".split(" ")
    print(sent)
    print(crf.decode(word_seq=sent))
    
if __name__ == "__main__":
    main()
