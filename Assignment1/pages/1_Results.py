import streamlit as st

# Set the title of the web app
st.title("Visualisation")

# Create tabs for HMM and CRF
tab1, tab2 = st.tabs(["HMM", "CRF"])

# Display PNG image in the HMM tab
with tab1:
    st.header("Visualisation for HMM Tagger")
    st.header("Confusion Matrix")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\HMM\HMM_conf_matrix.png", caption="HMM Confusion Matrix", use_column_width=True)

    st.header("F-Score")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\HMM\HMM_f_scores.png", caption="HMM F-Scores", use_column_width=True)

    st.header("Precision-Recall")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\HMM\HMM_precision_recall.png", caption="HMM Precision-Recall", use_column_width=True)

    st.header("Weighted Metrics")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\HMM\HMM_weighted_metrics.png", caption="HMM Weighted Metrics", use_column_width=True)

# Display PNG image in the CRF tab
with tab2:
    st.header("Visualisation for CRF Tagger")
    st.header("Confusion Matrix")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\CRF\CRF_conf_matrix.png", caption="Confusion Matrix", use_column_width=True)

    st.header("F-Score")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\CRF\CRF_f_scores.png", caption="F-Scores", use_column_width=True)

    st.header("Precision-Recall")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\CRF\CRF_precision_recall.png", caption="Precision-Recall", use_column_width=True)

    st.header("Weighted Metrics")
    st.image("C:\CMInDS\Sem3\CS-626\IITB-CS626-NLP\Assignment1\outputs\CRF\CRF_weighted_metrics.png", caption="Weighted Metrics", use_column_width=True)
