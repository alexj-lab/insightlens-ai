import streamlit as st, uuid, os, re, fitz
from bs4 import BeautifulSoup
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# ---------- IO ----------
MAX_CHARS = 30000
def read_pdf(path):
    text=[]; 
    with fitz.open(path) as doc:
        for p in doc: text.append(p.get_text())
    return "\n".join(text)

def read_txt(path):
    with open(path,"r",encoding="utf-8",errors="ignore") as f: 
        return f.read()

def read_html(path):
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        soup=BeautifulSoup(f,"html.parser")
    return soup.get_text(" ")

def read_any(path):
    ext=os.path.splitext(path)[1].lower()
    t= read_pdf(path) if ext==".pdf" else read_html(path) if ext in [".html",".htm"] else read_txt(path)
    t=re.sub(r"\n{2,}","\n\n",t)
    return t[:MAX_CHARS]

# ---------- Simple NLP ----------
STOP = ["the","and","of","to","in","for","on","with","a","an","by","is","are",
        "les","des","et","de","la","le","pour","dans","sur","aux","au","une","un"]

def split_sentences(text):
    # simple, language-agnostic-ish splitter
    s = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [x.strip() for x in s if len(x.strip())>0]

def summarize_tfidf(text, max_sent=5):
    sents = split_sentences(text)
    if len(sents)<=max_sent: return sents
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1,2), max_features=6000)
    X = vec.fit_transform(sents)
    scores = np.asarray(X.max(axis=1)).ravel()  # importance per sentence
    top_idx = scores.argsort()[-max_sent:][::-1]
    # keep original order for readability
    top_idx_sorted = sorted(top_idx)
    return [sents[i] for i in top_idx_sorted]

def sentiment_vader(text):
    # VADER is tuned for English; still gives a coarse gauge for FR. OK for demo.
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text[:5000])
    # map to pos/neu/neg proportions (sum to 1 approx)
    pos = max(0.0, s.get("pos",0.0))
    neu = max(0.0, s.get("neu",0.0))
    neg = max(0.0, s.get("neg",0.0))
    total = pos+neu+neg or 1.0
    return {"positive": pos/total, "neutral": neu/total, "negative": neg/total}

def top_keywords(text, k=12):
    paras=[p.strip() for p in re.split(r"\n\n+", text) if len(p.strip())>30]
    if not paras: paras=[text]
    vec=TfidfVectorizer(stop_words=STOP, max_features=4000, ngram_range=(1,2))
    X=vec.fit_transform(paras)
    scores=X.max(axis=0).A1
    terms=vec.get_feature_names_out()
    pairs=sorted(zip(terms,scores), key=lambda x:x[1], reverse=True)
    return pairs[:k]

def bar_keywords(pairs):
    terms=[t for t,_ in pairs]; vals=[v for _,v in pairs]
    fig=plt.figure(figsize=(6,3))
    plt.barh(terms[::-1], vals[::-1]); plt.tight_layout()
    return fig

def bar_sentiment(scores):
    labels=["positive","neutral","negative"]
    vals=[scores.get("positive",0), scores.get("neutral",0), scores.get("negative",0)]
    fig=plt.figure(figsize=(4,3))
    plt.bar(labels, vals); plt.ylim(0,1); plt.tight_layout()
    return fig

# ---------- UI ----------
st.set_page_config(page_title="InsightLens AI (Lite)", layout="wide")
st.title("🧾 InsightLens AI – Smart Document Summarizer (Lite)")

with st.sidebar:
    st.header("Settings")
    kwords = st.slider("Number of keywords", 5, 25, 12)

upl = st.file_uploader("Upload a document (PDF/TXT/HTML)", type=["pdf","txt","html","htm"])
col1, col2 = st.columns([2,1])

if upl:
    path=f"/tmp/{upl.name}"
    with open(path,"wb") as f: f.write(upl.read())
    text=read_any(path)

    if len(text.strip())<200:
        st.warning("The file seems empty or scanned (image). Please use a text-based PDF/TXT/HTML.")
        st.stop()

    st.success(f"Loaded {len(text)} characters – lang hint: {detect(text[:1000] or 'en')}")

    with st.spinner("Analyzing..."):
        bullets = summarize_tfidf(text, max_sent=5)
        sent = sentiment_vader(text)
        kw = top_keywords(text, k=kwords)

    with col1:
        st.subheader("Summary")
        st.write("• " + "\n• ".join(bullets))

        st.subheader("Top Keywords")
        st.pyplot(bar_keywords(kw))

    with col2:
        st.subheader("Overall Sentiment")
        st.pyplot(bar_sentiment(sent))
else:
    st.info("Drop a PDF / TXT / HTML file to start.")

