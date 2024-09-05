import streamlit as st
from HMM_tagger import HMM
from dataset import Dataset
from CRF_tagger import CRF

st.title("CS 626-Assignment 1")
tab1, tab2 = st. tabs(["HMM","CRF"])

with st.sidebar:
    pass



with tab1:
    col1, col2 = st.columns(2)
    col1.title("Input")
    col2.title("Output")
    with col1:
        x = st.text_area("Enter Sentence...")
        submit = st.button("Submit text")
        dataset = Dataset(num_folds=5, unk_threshold=2)
        x = x.split(" ")
        tagger = HMM(fold_index=0,dataset=dataset)
        op = tagger.viterbi_decode(word_seq=x)[1:-1]

    with col2:
        y = st.text_area("Output Displayed here", value = op if submit else "")

with tab2:
    col1, col2 = st.columns(2)
    col1.title("Input")
    col2.title("Output")
    with col1:
        x = st.text_area("Enter CRF Sentence...")
        crf_submit = st.button("Submit text",key='crf')
        dataset = Dataset(num_folds=5, unk_threshold=2)
        x = x.split(" ")
        tagger = CRF(dataset=dataset, fold_index=0, data_frac=1)
        op = tagger.decode(word_seq=x)

    with col2:
        y = st.text_area("Output Displayed here", key='op2', value = op if crf_submit else "")

    
    