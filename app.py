import os
import re
import json
from datetime import datetime

import streamlit as st
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────
APP_NAME = "InsightLens – AI Document Analyzer"
MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20
TOP_SENTENCES = 8
TOP_KEYWORDS = 15
WPM = 200  # reading-speed words per minute

st.set_page_config(page_title="InsightLens", layout="wide", page_icon="🧾")

# Professional minimal UI theme
st.markdown("""
<style>
:root {
  --brand:#0F62FE; /* IBM blue-ish */
  --ink:#121619;
  --muted:#6b7280;
  --card:#ffffff;
  --line:#e5e7eb;
}
html, body, [class*="block-container"] { background:#f7f8fa; color:var(--ink); }
h1,h2,h3,h4 { letter-spacing:0.2px }
.card {
  background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:18px 20px; box-shadow:0 1px 2px rgba(17,24,39,.05);
}
.kpi {
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:14px; text-align:center;
}
.kpi .v { font-size:1.4rem; font-weight:700; color:var(--brand) }
.kpi .l { font-size:.85rem; color:var(--muted) }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem;
  border:1px solid var(--line); background:#fff;
}
.summary-paragraph {
  line-height:1.7; font-size:1.03rem; color:#111827;
}
.footer-note { color:var(--muted); font-size:.85rem; }
hr { border:0; border-top:1px solid var(--line); margin: 8px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Minimal stopwords (EN + FR basics for robustness)
# ─────────────────────────────────────────────────────────────
STOP_WORDS = set([
    # EN
    "the","and","of","to","in","for","on","with","a","an","by","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","should","could","may","might","must","can","this",
    "that","these","those","i","you","he","she","it","we","they","what","which","who","when","where","why",
    "how","all","each","every","both","few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","but","or","as","at","from","into","through",
    # FR basics
    "le","la","les","de","des","du","un","une","et","à","au","aux","en","dans","sur","pour","par","avec","sans",
    "ce","cet","cette","ces","se","ses","son","sa","leurs","leur","plus","moins","a","ont","est","sont","été",
    "être","fait","faire","afin","ainsi","donc","car","ou","où"
])

# ─────────────────────────────────────────────────────────────
# File IO
# ─────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    try:
        out = []
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                try: doc.authenticate("")
                except Exception: return ""
            for page in doc:
                t = page.get_text("text")
                if t.strip(): out.append(t)
        return "\n\n".join(out)
    except Exception:
        return ""

def read_txt(path: str) -> str:
    for enc in ("utf-8","latin-1","cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def read_html(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script","style"]): tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        t = read_pdf(path)
    elif ext in (".html",".htm"):
        t = read_html(path)
    else:
        t = read_txt(path)
    t = clean_text(t)
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS]
        st.toast(f"Document truncated to {MAX_CHARS:,} characters", icon="⚠️")
    return t

# ─────────────────────────────────────────────────────────────
# NLP utilities (segmentation / scoring / MMR / rewrite)
# ─────────────────────────────────────────────────────────────
BULLET_PAT = re.compile(r'^\s*[\-\•\–\*]\s+')

def looks_like_heading(ln: str) -> bool:
    if not ln: return False
    if len(ln) <= 8: return True
    if ln.endswith(':'): return True
    letters = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]', '', ln)
    if letters:
        up = sum(c.isupper() for c in letters) / max(len(letters),1)
        if up > 0.6: return True
    return False

def split_sentences(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts, buf = [], []
    for ln in lines:
        if BULLET_PAT.match(ln) or looks_like_heading(ln):
            if buf: parts.append(" ".join(buf)); buf = []
            parts.append(ln)
        else:
            buf.append(ln)
    if buf: parts.append(" ".join(buf))

    sents = []
    for p in parts:
        if BULLET_PAT.match(p) or looks_like_heading(p):
            sents.append(p); continue
        chunks = re.split(r'(?<=[\.\!\?])\s+(?=[A-ZÉÈÊÀÂÎÔÛÇ])', p)
        for c in chunks:
            c = c.strip()
            if not c: continue
            if BULLET_PAT.match(c) or looks_like_heading(c) or len(c) >= MIN_SENTENCE_LENGTH:
                sents.append(c)

    seen, clean = set(), []
    for s in sents:
        key = s.lower()
        if key not in seen:
            seen.add(key); clean.append(s)
    return clean

def title_from_text(text: str) -> str:
    for ln in text.splitlines():
        t = ln.strip()
        if t: return t
    return ""

def build_tfidf(sentences, max_features=12000):
    vect = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1,2), max_features=max_features)
    X = vect.fit_transform(sentences)
    return vect, X

def sentence_position_bonus(n_sent, idx):
    return 1.0 - 0.4 * (idx / max(n_sent-1,1))

def mmr_select(X, k, diversity=0.7):
    sim = cosine_similarity(X)
    X_max = X.max(axis=1)
    rel = X_max.toarray().ravel() if hasattr(X_max, "toarray") else np.array(X_max).ravel()

    n = sim.shape[0]
    selected = []
    first = int(rel.argmax())
    selected.append(first)
    cand = set(range(n)) - {first}

    while len(selected) < min(k, n) and cand:
        best, best_val = None, -1e9
        for i in cand:
            redundancy = max(sim[i, selected]) if selected else 0.0
            val = (1 - diversity)*rel[i] - diversity*redundancy
            if val > best_val:
                best_val, best = val, i
        selected.append(best); cand.remove(best)
    return sorted(selected)

def rewrite_sentence(s: str) -> str:
    x = s.strip()
    x = re.sub(r'\s*\([^)]*\)', '', x)
    x = re.sub(r'^(However|Moreover|Furthermore|Additionally|In addition|Thus|Therefore)[,:]\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'^(According to|Per|As noted)\s+[^:]+:\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'\s*,\s*(which|that)\s+', ' ', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(due to|because of|owing to)\b', 'because of', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(in order to)\b', 'to', x, flags=re.IGNORECASE)
    x = re.sub(r'\s{2,}', ' ', x).strip(' -–—;:,')
    if not re.search(r'[.!?]$', x): x += '.'
    return x

def summarize_bullets(text: str, target_sentences=TOP_SENTENCES):
    sents = split_sentences(text)
    if not sents: return []
    if len(sents) <= target_sentences:
        return [rewrite_sentence(s) for s in sents]

    title = title_from_text(text)
    vect, X = build_tfidf(sents, max_features=12000)

    X_max = X.max(axis=1)
    centrality = X_max.toarray().ravel() if hasattr(X_max, "toarray") else np.array(X_max).ravel()

    title_sim = np.zeros(len(sents))
    if title:
        try:
            Xt = vect.transform([title])
            title_sim = np.asarray(cosine_similarity(X, Xt)).ravel()
        except Exception:
            title_sim = np.zeros(len(sents))

    pos_bonus = np.array([sentence_position_bonus(len(sents), i) for i in range(len(sents))], dtype=float)
    score = 0.60*centrality + 0.25*pos_bonus + 0.15*title_sim

    X_bias = X.multiply(score.reshape(-1,1))
    keep = mmr_select(X_bias, k=target_sentences, diversity=0.7)
    chosen = [rewrite_sentence(sents[i]) for i in keep]

    # dedupe + keep first 5 strong bullets
    out, seen = [], set()
    for b in chosen:
        k = b.lower()
        if k not in seen and len(b) >= 25:
            seen.add(k); out.append(b)
        if len(out) >= 5: break
    if not out:
        out = [rewrite_sentence(s) for s in sents[:5]]
    return out

def bullets_to_paragraph(bullets):
    """Always return a non-empty paragraph from bullets."""
    if not bullets: return ""
    lead = "This document discusses "
    cleaned = [re.sub(r'[ \t]*\.$', '', b).strip() for b in bullets]
    conns = [" It then explains", " Additionally,", " Moreover,", " Finally,"]
    parts = [lead + cleaned[0].lstrip("-.• ")]
    for i, sent in enumerate(cleaned[1:], start=1):
        parts.append(f"{conns[min(i-1, len(conns)-1)]} {sent.lstrip('-.• ')}")
    para = " ".join(parts).strip()
    if not para.endswith("."): para += "."
    return para

# ─────────────────────────────────────────────────────────────
# Keywords
# ─────────────────────────────────────────────────────────────
def extract_keywords(text: str, top_n=TOP_KEYWORDS):
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 60]
    if len(paragraphs) < 2:
        chunk_size = 600
        paragraphs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    if not paragraphs: paragraphs = [text]

    vect = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1,3), max_features=6000, min_df=1, max_df=0.85)
    X = vect.fit_transform(paragraphs)
    if X.shape[1] == 0: return []

    if isinstance(X, (coo_matrix, csr_matrix)): Xd = X.toarray()
    else: Xd = np.array(X)
    scores = Xd.max(axis=0)
    if len(scores.shape) > 1: scores = scores.flatten()

    terms = vect.get_feature_names_out()
    pairs = list(zip(terms, scores.tolist()))

    def ok(term):
        if len(term) < 3: return False
        if re.fullmatch(r'[\W_]+', term): return False
        if re.fullmatch(r'\d+(\.\d+)?', term): return False
        return True

    pairs = [(t, float(s)) for t, s in pairs if ok(t)]
    pairs.sort(key=lambda x: x[1], reverse=True)

    chosen, seen = [], set()
    for t, s in pairs:
        low = t.lower()
        if any(low in k or k in low for k in seen): continue
        seen.add(low); chosen.append((t, s))
        if len(chosen) >= top_n: break
    return chosen

# ─────────────────────────────────────────────────────────────
# Sentiment + explanation
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def text_stats(text: str):
    words = re.findall(r"\b\w+\b", text)
    wc = len(words)
    cc = len(text)
    paras = len([p for p in re.split(r"\n{2,}", text) if p.strip()])
    minutes = wc / WPM
    mm = int(minutes); ss = int((minutes - mm) * 60)
    rt = f"{mm}m{ss:02d}s" if mm > 0 else f"{ss}s"
    return {"words": wc, "chars": cc, "paragraphs": paras, "reading_time": rt}

def sentiment_scores(text: str):
    a = load_sentiment_analyzer()
    s = a.polarity_scores(text[:6000])
    pos = max(0.0, s.get("pos",0.0))
    neu = max(0.0, s.get("neu",0.0))
    neg = max(0.0, s.get("neg",0.0))
    total = pos+neu+neg or 1.0
    return {"positive": pos/total, "neutral": neu/total, "negative": neg/total, "compound": s.get("compound",0.0)}

def sentiment_explanation(text: str, max_words=6, max_sents=2):
    a = load_sentiment_analyzer()
    lex = a.lexicon
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    freq = {}
    for t in tokens:
        if t in STOP_WORDS: continue
        freq[t] = freq.get(t, 0) + 1

    pos_w, neg_w = [], []
    for tok, n in freq.items():
        if tok in lex:
            val = lex[tok] * n
            if val > 0: pos_w.append((tok, val))
            elif val < 0: neg_w.append((tok, -val))
    pos_w.sort(key=lambda x: x[1], reverse=True)
    neg_w.sort(key=lambda x: x[1], reverse=True)
    pos_words = [t for t,_ in pos_w[:max_words]]
    neg_words = [t for t,_ in neg_w[:max_words]]

    sents = split_sentences(text)
    scored = [(s, a.polarity_scores(s).get("compound",0.0)) for s in sents]
    top_pos = [s for s,_ in sorted(scored, key=lambda x:x[1], reverse=True)[:max_sents]]
    top_neg = [s for s,_ in sorted(scored, key=lambda x:x[1])[:max_sents]]

    return {"pos_words": pos_words, "neg_words": neg_words, "top_pos_sents": top_pos, "top_neg_sents": top_neg}

# ─────────────────────────────────────────────────────────────
# Viz helpers
# ─────────────────────────────────────────────────────────────
def fig_keywords(pairs):
    if not pairs:
        fig = plt.figure(figsize=(6,3))
        plt.text(0.5,0.5,"No keywords", ha="center", va="center"); plt.axis("off")
        return fig
    terms = [t for t,_ in pairs[:15]]
    vals = [v for _,v in pairs[:15]]
    fig = plt.figure(figsize=(6.6,3.6))
    plt.barh(terms[::-1], vals[::-1])
    plt.xlabel("TF-IDF (max per segment)")
    plt.tight_layout()
    return fig

def fig_sentiment(scores):
    labels = ["positive","neutral","negative"]
    vals = [scores.get("positive",0), scores.get("neutral",0), scores.get("negative",0)]
    fig = plt.figure(figsize=(4.2,3.2))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.ylabel("Share")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────
def save_json_md(base_dir: str, payload: dict, paragraph: str, keywords: list):
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join(base_dir, f"{run_id}.json")
    mpath = os.path.join(base_dir, f"{run_id}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        "# InsightLens Report",
        f"Date (UTC): {payload['meta']['timestamp']}",
        f"Characters: {payload['meta']['chars']}",
        f"Words: {payload['meta']['words']}",
        f"Reading time: {payload['meta']['reading_time']}",
        "",
        "## Summary (paragraph)",
        paragraph if paragraph else "(empty)",
        "",
        "## Top Keywords",
        ", ".join([t for t,_ in keywords]) if keywords else "(none)",
        ""
    ]
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return jpath, mpath

# ─────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────
st.title("🧾 " + APP_NAME)
st.caption("Upload a PDF / TXT / HTML. Get a professional single-paragraph summary, sentiment with explanations, keywords, and reading stats.")

with st.sidebar:
    st.header("Settings")
    kwords = st.slider("Number of keywords", 5, 25, TOP_KEYWORDS)
    st.caption("Tip: TXT/HTML are fastest. For PDFs, prefer text-based (not scanned).")

upload = st.file_uploader("Upload document", type=["pdf","txt","html","htm"])
main, side = st.columns([2,1], gap="large")

if upload:
    tmp = f"/tmp/{upload.name}"
    with open(tmp, "wb") as f: f.write(upload.read())

    text = read_any(tmp)
    if len(text.strip()) < 200:
        st.error("This file seems empty, scanned, or too short. Please use a text-based PDF or a .txt/.html file.")
        st.stop()

    stats = text_stats(text)
    st.caption(f"Loaded {len(text)} characters")

    with st.spinner("Analyzing document..."):
        bullets = summarize_bullets(text, target_sentences=TOP_SENTENCES)
        # hard fallback if bullets somehow empty:
        if not bullets:
            raw_sents = split_sentences(text)[:5]
            bullets = [rewrite_sentence(s) for s in raw_sents] if raw_sents else ["This document could not be summarized due to insufficient textual content."]
        paragraph = bullets_to_paragraph(bullets)
        if not paragraph.strip():
            # final fallback – never show an empty box
            paragraph = "This document discusses key points presented in the text. It outlines the main ideas, supporting arguments, and notable facts, then concludes with the most relevant implications."

        senti = sentiment_scores(text)
        senti_expl = sentiment_explanation(text, max_words=6, max_sents=2)
        keywords = extract_keywords(text, top_n=kwords)

    # ── Stats (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi"><div class="v">{stats["words"]:,}</div><div class="l">words</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="v">{stats["chars"]:,}</div><div class="l">characters</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="v">{stats["paragraphs"]}</div><div class="l">paragraphs</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="v">{stats["reading_time"]}</div><div class="l">reading time</div></div>', unsafe_allow_html=True)

    with main:
        st.subheader("Executive Summary")
        st.markdown(f'<div class="card summary-paragraph">{paragraph}</div>', unsafe_allow_html=True)

        st.subheader("Top Keywords")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(fig_keywords(keywords))
        st.markdown('</div>', unsafe_allow_html=True)

    with side:
        st.subheader("Overall Sentiment")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(fig_sentiment(senti))
        label = "Positive" if senti["compound"] >= 0.2 else "Negative" if senti["compound"] <= -0.2 else "Neutral"
        st.markdown(f'<span class="badge">Overall: <b>{label}</b> (compound={senti["compound"]:.3f})</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Why this sentiment?")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Positive drivers:** " + (", ".join(senti_expl["pos_words"]) if senti_expl["pos_words"] else "_none detected_"))
        st.markdown("**Negative drivers:** " + (", ".join(senti_expl["neg_words"]) if senti_expl["neg_words"] else "_none detected_"))
        if senti_expl["top_pos_sents"]:
            st.markdown("**Positive evidence:**")
            for s in senti_expl["top_pos_sents"]:
                st.markdown(f"- {s}")
        if senti_expl["top_neg_sents"]:
            st.markdown("**Negative evidence:**")
            for s in senti_expl["top_neg_sents"]:
                st.markdown(f"- {s}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Export
    payload = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "chars": len(text),
            "words": stats["words"],
            "reading_time": stats["reading_time"],
            "top_sentences": TOP_SENTENCES,
            "top_keywords": kwords
        },
        "summary_paragraph": paragraph,
        "summary_bullets": bullets,
        "sentiment": senti,
        "sentiment_explanation": senti_expl,
        "keywords": keywords
    }
    jpath, mpath = save_json_md("outputs", payload, paragraph, keywords)
    st.caption(f"Saved reports: {jpath}  ·  {mpath}")
    st.markdown('<div class="footer-note">Local files are stored in a temporary environment and may be cleared after the session.</div>', unsafe_allow_html=True)

else:
    st.info("Drop a PDF / TXT / HTML file to start. For best results, prefer text-based PDFs (not scanned images).")
