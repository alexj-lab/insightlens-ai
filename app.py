import streamlit as st
import os, re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

MAX_CHARS = 30000

def read_pdf(path):
    text = []
    with fitz.open(path) as doc:
        for p in doc:
            text.append(p.get_text())
    return "\n".join(text)

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(" ")

def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        t = read_pdf(path)
    elif ext in [".html", ".htm"]:
        t = read_html(path)
    else:
        t = read_txt(path)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t[:MAX_CHARS]

STOP = ["the", "and", "of", "to", "in", "for", "on", "with", "a", "an", "by", "is", "are",
        "les", "des", "et", "de", "la", "le", "pour", "dans", "sur", "aux", "au", "une", "un"]

def split_sentences(text):
    s = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [x.strip() for x in s if len(x.strip()) > 0]

def summarize_tfidf(text, max_sent=5):
    sents = split_sentences(text)
    if len(sents) <= max_sent:
        return sents
    
    try:
        vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), max_features=6000)
        X = vec.fit_transform(sents)
        scores = np.asarray(X.max(axis=1)).flatten()
        top_idx = scores.argsort()[-max_sent:][::-1]
        top_idx_sorted = sorted(top_idx)
        return [sents[i] for i in top_idx_sorted]
    except:
        return sents[:max_sent]

def top_keywords(text, k=12):
    paras = [p.strip() for p in re.split(r"\n\n+", text) if len(p.strip()) > 30]
    
    # FIX : Vérifications sans ambiguïté
    if len(paras) == 0:
        return []
    if len(paras) < 2:
        return []
    
    try:
        vec = TfidfVectorizer(stop_words=STOP, max_features=4000, ngram_range=(1, 2))
        X = vec.fit_transform(paras)
        
        # Conversion numpy sûre
        scores_matrix = X.max(axis=0)
        scores = np.array(scores_matrix).flatten()
        
        terms = vec.get_feature_names_out()
        pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return pairs[:k]
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

def bar_keywords(pairs):
    if len(pairs) == 0:
        return None
    terms = [t for t, _ in pairs[:10]]
    vals = [v for _, v in pairs[:10]]
    fig = plt.figure(figsize=(6, 3))
    plt.barh(terms[::-1], vals[::-1], color='#21808d')
    plt.xlabel("TF-IDF Score")
    plt.title("Top Keywords")
    plt.tight_layout()
    return fig

# ========== UI ==========
st.set_page_config(page_title="InsightLens AI", layout="wide")
st.title("🧾 InsightLens AI – Smart Document Summarizer")

with st.sidebar:
    st.header("⚙️ Settings")
    kwords = st.slider("Number of keywords", 5, 25, 12)
    st.markdown("---")
    st.caption("Powered by TF-IDF & scikit-learn")

upl = st.file_uploader("📄 Upload document (PDF/TXT/HTML)", type=["pdf", "txt", "html", "htm"])

col1, col2 = st.columns([2, 1])

if upl:
    try:
        # Save uploaded file
        path = f"/tmp/{upl.name}"
        with open(path, "wb") as f:
            f.write(upl.read())
        
        # Read content
        text = read_any(path)
        
        if len(text.strip()) < 100:
            st.warning("⚠️ File too short or empty")
            st.stop()
        
        st.success(f"✅ Loaded {len(text):,} characters")
        
        # Analysis
        with st.spinner("🤖 Analyzing..."):
            bullets = summarize_tfidf(text, max_sent=5)
            kw = top_keywords(text, k=kwords)
        
        # Display results
        with col1:
            st.subheader("📋 Executive Summary")
            for i, bullet in enumerate(bullets, 1):
                st.markdown


