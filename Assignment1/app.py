import streamlit as st
from HMM_tagger import HMM
from dataset import Dataset

st.title("CS 626-Assignment 1")

col1, col2 = st.columns(2)
col1.title("Input")
col2.title("Output")
with col1:
    x = st.text_area("Enter Sentence...")
    submit = st.button("Submit text")
    dataset = Dataset(num_folds=5, unk_threshold=2)
    x = x.split(" ")
    tagger = HMM(fold_index=0,dataset=dataset )
    op = tagger.viterbi_decode(word_seq=x)

with col2:
    y = st.text_area("Output Displayed here", value = op if submit else "")
    
    