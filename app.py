import streamlit as st, uuid, os, re, fitz
from bs4 import BeautifulSoup
from langdetect import detect
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# --------------------------
# Utils lecture documents
# --------------------------
MAX_CHARS = 30000

def read_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for p in doc:
            text.append(p.get_text())
    return "\n".join(text)

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(" ")

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        t = read_pdf(path)
    elif ext in [".html", ".htm"]:
        t = read_html(path)
    else:
        t = read_txt(path)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t[:MAX_CHARS]

# --------------------------
# Modèles HF (chargés une fois)
# --------------------------
@st.cache_resource(show_spinner=False)
def get_pipes():
    # Résumé: EN-only petit & rapide (stable sur Streamlit Cloud)
    summarizer_en = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Sentiment: multilingue (XLM-R)
    sentiment = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        return_all_scores=True,
    )
    return summarizer_en, sentiment

def summarize_text(text: str, lang_hint: str = None):
    lang = (lang_hint or detect(text[:1000] or "en")).lower()
    summarizer_en, _ = get_pipes()
    # Pour la V1: on passe par l'anglais (plus léger). Pour FR, ça marche si texte EN sinon output peut être mixte.
    out = summarizer_en(text[:4000], max_length=180, min_length=60, do_sample=False)
    return out[0]["summary_text"].strip(), lang

def sentiment_scores(text: str):
    _, sentiment = get_pipes()
    scores = sentiment(text[:4000])[0]
    m = {s["label"].lower(): s["score"] for s in scores}
    # Normaliser labels si besoin
    if not {"positive","neutral","negative"}.issubset(set(m.keys())):
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        names = ["positive","neutral","negative"]
        m = {names[i]: scores[i]["score"] for i in range(min(3, len(scores)))}
    return m

def top_keywords(text: str, k: int = 12):
    paras = [p.strip() for p in re.split(r"\n\n+", text) if len(p.strip()) > 30]
    if not paras: paras = [text]
    vec = TfidfVectorizer(stop_words=[
        "the","and","of","to","in","for","on","with","a","an","by","is","are",
        "les","des","et","de","la","le","pour","dans","sur","aux","au","une","un"
    ], max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(paras)
    scores = X.max(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return pairs[:k]

def bar_keywords(pairs):
    terms = [t for t,_ in pairs]
    vals = [v for _,v in pairs]
    fig = plt.figure(figsize=(6,3))
    plt.barh(terms[::-1], vals[::-1])
    plt.tight_layout()
    return fig

def bar_sentiment(scores):
    labels = ["positive","neutral","negative"]
    vals = [scores.get("positive",0), scores.get("neutral",0), scores.get("negative",0)]
    fig = plt.figure(figsize=(4,3))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.tight_layout()
    return fig

# --------------------------
# UI Streamlit
# --------------------------
st.set_page_config(page_title="InsightLens AI", layout="wide")
st.title("🧾 InsightLens AI – Smart Document Summarizer (V1)")

with st.sidebar:
    st.header("Settings")
    kwords = st.slider("Number of keywords", 5, 25, 12)

upl = st.file_uploader("Upload a document (PDF/TXT/HTML)", type=["pdf","txt","html","htm"])
col1, col2 = st.columns([2,1])

if upl:
    path = f"/tmp/{upl.name}"
    with open(path, "wb") as f: f.write(upl.read())
    text = read_any(path)

    st.success(f"Loaded {len(text)} characters")

    with st.spinner("Analyzing... (first run downloads models)"):
        summary, lang = summarize_text(text)
        sent = sentiment_scores(text)
        kw = top_keywords(text, k=kwords)

    with col1:
        st.subheader("Summary")
        bullets = [s.strip() for s in summary.split(".") if len(s.strip())>0][:5]
        st.write("• " + "\n• ".join(bullets))

        st.subheader("Top Keywords")
        fig_kw = bar_keywords(kw)
        st.pyplot(fig_kw)

    with col2:
        st.subheader(f"Overall Sentiment (lang={lang})")
        fig_s = bar_sentiment(sent)
        st.pyplot(fig_s)

    st.caption("V1 note: fast English summarizer used for reliability on Streamlit Cloud. We can switch to multilingual later if needed.")
else:
    st.info("Drop a PDF / TXT / HTML file to start.")
