import streamlit as st
import os
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Configs & constants
# ─────────────────────────────────────────────────────────────
MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20
TOP_SENTENCES = 8          # nb de phrases candidates avant réécriture
TOP_KEYWORDS = 15
WPM = 200                  # words per minute (lecture)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

STOP_WORDS = set([
    # EN
    "the","and","of","to","in","for","on","with","a","an","by","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","should","could","may","might","must","can","this",
    "that","these","those","i","you","he","she","it","we","they","what","which","who","when","where","why",
    "how","all","each","every","both","few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","but","or","as","at","from","into","through",
    # FR (basiques)
    "le","la","les","de","des","du","un","une","et","à","au","aux","en","dans","sur","pour","par","avec","sans",
    "ce","cet","cette","ces","se","ses","son","sa","leurs","leur","plus","moins","a","ont","est","sont","été",
    "être","fait","faire","afin","ainsi","donc","car","ou","où"
])

# ─────────────────────────────────────────────────────────────
# UI style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="InsightLens AI (Lite+)", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a73e8; margin-bottom: 0.25rem; }
    .subtitle { font-size: 0.95rem; color: #5f6368; margin-bottom: 1rem; }
    .summary-box { background: #fff; border: 1px solid #dadce0; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
    .info-box { background: #e8f0fe; border-left: 4px solid #1967d2; border-radius: 4px; padding: 0.75rem 1rem; margin: 0.75rem 0; }
    .stat-card { background:#fff; border:1px solid #e8eaed; border-radius:10px; padding:0.9rem; text-align:center }
    .stat-val { font-size:1.4rem; font-weight:700; color:#1a73e8 }
    .stat-lb { font-size:0.85rem; color:#5f6368 }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# File reading helpers
# ─────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    """Read textual content from a PDF. Return empty string if nothing extractable."""
    try:
        text = []
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                try:
                    doc.authenticate("")  # try blank password
                except Exception:
                    return ""
            for page in doc:
                t = page.get_text("text")
                if t.strip():
                    text.append(t)
        return "\n\n".join(text)
    except Exception:
        return ""

def read_txt(path: str) -> str:
    for enc in ('utf-8', 'latin-1', 'cp1252'):
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
        for tag in soup(["script", "style"]):
            tag.decompose()
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
        text = read_pdf(path)
    elif ext in (".html", ".htm"):
        text = read_html(path)
    else:
        text = read_txt(path)
    text = clean_text(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        st.warning(f"⚠️ Document truncated to {MAX_CHARS:,} characters")
    return text

# ─────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────
BULLET_PAT = re.compile(r'^\s*[\-\•\–\*]\s+')

def looks_like_heading(ln: str) -> bool:
    if not ln: return False
    if len(ln) <= 8: return True
    if ln.endswith(':'): return True
    letters = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]', '', ln)
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
        if upper_ratio > 0.6:
            return True
    return False

def split_sentences(text: str):
    """Segment paragraphs, keep bullets and headings, split on .!? with capital after."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts, buf = [], []
    for ln in lines:
        if BULLET_PAT.match(ln) or looks_like_heading(ln):
            if buf:
                parts.append(' '.join(buf)); buf = []
            parts.append(ln)
        else:
            buf.append(ln)
    if buf: parts.append(' '.join(buf))

    sents = []
    for p in parts:
        if BULLET_PAT.match(p) or looks_like_heading(p):
            sents.append(p)
            continue
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

def text_stats(text: str):
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    paragraphs = len([p for p in re.split(r"\n{2,}", text) if p.strip()])
    minutes = word_count / WPM
    mm = int(minutes)
    ss = int((minutes - mm) * 60)
    reading_time = f"{mm}m{ss:02d}s" if mm > 0 else f"{ss}s"
    return {
        "words": word_count,
        "chars": char_count,
        "paragraphs": paragraphs,
        "reading_time": reading_time
    }

# ─────────────────────────────────────────────────────────────
# Scoring / MMR / rewrite / paragraph summary
# ─────────────────────────────────────────────────────────────
def build_tfidf(sentences, max_features=12000):
    vect = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1,2), max_features=max_features)
    X = vect.fit_transform(sentences)
    return vect, X

def sentence_position_bonus(n_sent, idx):
    return 1.0 - 0.4 * (idx / max(n_sent-1, 1))

def mmr_select(X, k, diversity=0.7):
    """Maximal Marginal Relevance to avoid redundancy."""
    sim = cosine_similarity(X)
    n = sim.shape[0]
    X_max = X.max(axis=1)
    rel = X_max.toarray().ravel() if hasattr(X_max, "toarray") else np.array(X_max).ravel()

    selected = []
    first = int(rel.argmax())
    selected.append(first)
    cand = set(range(n)) - {first}

    while len(selected) < min(k, n) and cand:
        best, best_val = None, -1e9
        for i in cand:
            redundancy = max(sim[i, selected]) if selected else 0.0
            val = (1 - diversity) * rel[i] - diversity * redundancy
            if val > best_val:
                best_val, best = val, i
        selected.append(best); cand.remove(best)

    return sorted(selected)

def rewrite_sentence(s: str) -> str:
    """Rule-based compression / rewrite for cleaner summary."""
    x = s.strip()
    x = re.sub(r'\s*\([^)]*\)', '', x)
    x = re.sub(r'^(However|Moreover|Furthermore|Additionally|In addition|Ainsi|Cependant|Toutefois)[,:]\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'^(According to|Per|As noted|Comme indiqué|D’après)\s+[^:]+:\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'\s*,\s*(which|that)\s+', ' ', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(due to|because of|owing to|en raison de|à cause de)\b', 'because of', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(in order to|afin de)\b', 'to', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(such as|comme|par exemple)\b', 'for example', x, flags=re.IGNORECASE)
    x = re.sub(r'\s{2,}', ' ', x).strip(' -–—;:,')
    if not re.search(r'[.!?]$', x):
        x += '.'
    return x

def summarize_text_bullets(text: str, target_sentences=TOP_SENTENCES):
    sentences = split_sentences(text)
    if len(sentences) == 0:
        return []
    if len(sentences) <= target_sentences:
        chosen = [rewrite_sentence(s) for s in sentences]
    else:
        title = title_from_text(text)
        vect, X = build_tfidf(sentences, max_features=12000)
        X_max = X.max(axis=1)
        centrality = X_max.toarray().ravel() if hasattr(X_max, "toarray") else np.array(X_max).ravel()

        title_sim = np.zeros(len(sentences))
        if title:
            try:
                Xt = vect.transform([title])
                title_sim = np.asarray(cosine_similarity(X, Xt)).ravel()
            except Exception:
                title_sim = np.zeros(len(sentences))

        pos_bonus = np.array([sentence_position_bonus(len(sentences), i) for i in range(len(sentences))], dtype=float)
        score = 0.60 * centrality + 0.25 * pos_bonus + 0.15 * title_sim
        X_bias = X.multiply(score.reshape(-1, 1))
        keep_idx = mmr_select(X_bias, k=target_sentences, diversity=0.7)
        chosen = [rewrite_sentence(sentences[i]) for i in keep_idx]

    bullets, seen = [], set()
    for b in chosen:
        key = b.lower()
        if key not in seen and len(b) >= 25:
            seen.add(key)
            bullets.append(b)
        if len(bullets) >= 5:
            break
    if not bullets:
        bullets = [rewrite_sentence(s) for s in sentences[:5]]
    return bullets

def bullets_to_paragraph(bullets):
    """Build one cohesive paragraph starting with 'Ce document parle de …'."""
    if not bullets: return ""
    lead = "Ce document parle de "
    # nettoyer les points finaux pour enchainer
    cleaned = [re.sub(r'[ \t]*\.$', '', b).strip() for b in bullets]
    # connecteurs simples pour fluidité
    connectors = [" Il explique ensuite que", " Par ailleurs,", " De plus,", " Enfin,"]
    parts = [lead + cleaned[0].lstrip("-.• ")]
    for i, sent in enumerate(cleaned[1:], start=1):
        conn = connectors[min(i-1, len(connectors)-1)]
        parts.append(f"{conn} {sent.lstrip('-.• ')}")
    para = " ".join(parts).strip()
    if not para.endswith("."):
        para += "."
    return para

# ─────────────────────────────────────────────────────────────
# Keywords
# ─────────────────────────────────────────────────────────────
def extract_keywords(text: str, top_n=TOP_KEYWORDS):
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 60]
    if len(paragraphs) < 2:
        chunk_size = 600
        paragraphs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    if not paragraphs:
        paragraphs = [text]

    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 3),
        max_features=6000,
        min_df=1,
        max_df=0.85
    )
    X = vectorizer.fit_transform(paragraphs)
    if X.shape[1] == 0:
        return []

    if isinstance(X, (coo_matrix, csr_matrix)):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    scores_array = X_dense.max(axis=0)
    if len(scores_array.shape) > 1:
        scores_array = scores_array.flatten()

    terms_array = vectorizer.get_feature_names_out()
    pairs = list(zip(terms_array, scores_array.tolist()))

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
        if any(low in k or k in low for k in seen):
            continue
        seen.add(low)
        chosen.append((t, s))
        if len(chosen) >= top_n:
            break
    return chosen

# ─────────────────────────────────────────────────────────────
# Sentiment + explanation (drivers & evidence)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def sentiment_scores(text: str):
    analyzer = load_sentiment_analyzer()
    s = analyzer.polarity_scores(text[:6000])
    pos = max(0.0, s.get("pos", 0.0))
    neu = max(0.0, s.get("neu", 0.0))
    neg = max(0.0, s.get("neg", 0.0))
    total = pos + neu + neg or 1.0
    return {"positive": pos/total, "neutral": neu/total, "negative": neg/total, "compound": s.get("compound", 0.0)}

def sentiment_explanation(text: str, max_words=6, max_sents=2):
    """Return top positive/negative tokens (by VADER lexicon score * freq) and evidence sentences."""
    analyzer = load_sentiment_analyzer()
    lex = analyzer.lexicon  # token -> valence
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    freq = {}
    for t in tokens:
        if t in STOP_WORDS: continue
        freq[t] = freq.get(t, 0) + 1

    pos_scores, neg_scores = [], []
    for tok, n in freq.items():
        if tok in lex:
            val = lex[tok] * n
            if val > 0:
                pos_scores.append((tok, val))
            elif val < 0:
                neg_scores.append((tok, -val))
    pos_scores.sort(key=lambda x: x[1], reverse=True)
    neg_scores.sort(key=lambda x: x[1], reverse=True)
    pos_words = [t for t, _ in pos_scores[:max_words]]
    neg_words = [t for t, _ in neg_scores[:max_words]]

    # evidence sentences
    sents = split_sentences(text)
    scored = []
    for s in sents:
        sc = analyzer.polarity_scores(s).get("compound", 0.0)
        scored.append((s, sc))
    top_pos = [s for s, sc in sorted(scored, key=lambda x: x[1], reverse=True)[:max_sents]]
    top_neg = [s for s, sc in sorted(scored, key=lambda x: x[1])[:max_sents]]

    return {
        "pos_words": pos_words,
        "neg_words": neg_words,
        "top_pos_sents": top_pos,
        "top_neg_sents": top_neg
    }

# ─────────────────────────────────────────────────────────────
# Viz
# ─────────────────────────────────────────────────────────────
def bar_keywords(pairs):
    if not pairs:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No keywords", ha='center', va='center')
        plt.axis('off'); return fig
    terms = [t for t, _ in pairs[:15]]
    vals = [v for _, v in pairs[:15]]
    fig = plt.figure(figsize=(6.5, 3.5))
    plt.barh(terms[::-1], vals[::-1])
    plt.xlabel("TF-IDF (max per segment)")
    plt.tight_layout()
    return fig

def bar_sentiment(scores):
    labels = ["positive", "neutral", "negative"]
    vals = [scores.get("positive",0), scores.get("neutral",0), scores.get("negative",0)]
    fig = plt.figure(figsize=(4.2, 3.2))
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Share")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────
def save_json_md(base_dir: str, payload: dict, paragraph: str, keywords: list):
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join(base_dir, f"{run_id}.json")
    mpath = os.path.join(base_dir, f"{run_id}.md")

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = ["# InsightLens Report",
          f"Date (UTC): {payload['meta']['timestamp']}",
          f"Characters: {payload['meta']['chars']}",
          f"Words: {payload['meta']['words']}",
          f"Reading time: {payload['meta']['reading_time']}",
          ""]
    md.append("## Summary (paragraph)")
    md.append(paragraph)
    md.append("")
    md.append("## Top Keywords")
    if keywords:
        md.append(", ".join([t for t, _ in keywords]))
    else:
        md.append("(none)")
    md.append("")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return jpath, mpath

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧾 InsightLens AI – Smart Document Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a PDF/TXT/HTML → get stats, a one-paragraph summary, sentiment with explanations, and keywords.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    kwords = st.slider("Number of keywords", 5, 25, TOP_KEYWORDS)
    st.caption("Tip: TXT/HTML are fastest. For PDFs, use text-based files (not scanned images).")

upl = st.file_uploader("Upload a document (PDF/TXT/HTML)", type=["pdf","txt","html","htm"])
c_main, c_side = st.columns([2,1])

if upl:
    tmp_path = f"/tmp/{upl.name}"
    with open(tmp_path, "wb") as f:
        f.write(upl.read())

    text = read_any(tmp_path)
    if len(text.strip()) < 200:
        st.error("This file seems empty, scanned, or too short. Use a text-based PDF or a .txt/.html file.")
        st.stop()

    stats = text_stats(text)
    st.info(f"Loaded {len(text)} characters")

    with st.spinner("Analyzing..."):
        bullets = summarize_text_bullets(text, target_sentences=TOP_SENTENCES)
        paragraph = bullets_to_paragraph(bullets)
        senti = sentiment_scores(text)
        senti_expl = sentiment_explanation(text, max_words=6, max_sents=2)
        keywords = extract_keywords(text, top_n=kwords)

    # Stats cards
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f'<div class="stat-card"><div class="stat-val">{stats["words"]:,}</div><div class="stat-lb">mots</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-card"><div class="stat-val">{stats["chars"]:,}</div><div class="stat-lb">caractères</div></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat-card"><div class="stat-val">{stats["paragraphs"]}</div><div class="stat-lb">paragraphes</div></div>', unsafe_allow_html=True)
    s4.markdown(f'<div class="stat-card"><div class="stat-val">{stats["reading_time"]}</div><div class="stat-lb">temps de lecture</div></div>', unsafe_allow_html=True)

    with c_main:
        st.subheader("Résumé (paragraphe)")
        if paragraph:
            st.markdown(f'<div class="summary-box">{paragraph}</div>', unsafe_allow_html=True)
        else:
            st.warning("No summary could be produced (document may be too short).")

        st.subheader("Top Keywords")
        st.pyplot(bar_keywords(keywords))

    with c_side:
        st.subheader("Overall Sentiment")
        st.pyplot(bar_sentiment(senti))
        # Label global
        label = "positif" if senti["compound"] >= 0.2 else "négatif" if senti["compound"] <= -0.2 else "neutre"
        st.markdown(f"**Label global : {label}**  (compound={senti['compound']:.3f})")

        with st.expander("Pourquoi ce score ?"):
            st.markdown("**Mots qui tirent le score vers le haut (positif)** : " + (", ".join(senti_expl["pos_words"]) if senti_expl["pos_words"] else "_aucun repéré_"))
            st.markdown("**Mots qui tirent le score vers le bas (négatif)** : " + (", ".join(senti_expl["neg_words"]) if senti_expl["neg_words"] else "_aucun repéré_"))
            if senti_expl["top_pos_sents"]:
                st.markdown("**Exemples de phrases positives :**")
                for s in senti_expl["top_pos_sents"]:
                    st.markdown(f"- {s}")
            if senti_expl["top_neg_sents"]:
                st.markdown("**Exemples de phrases négatives :**")
                for s in senti_expl["top_neg_sents"]:
                    st.markdown(f"- {s}")

        with st.expander("Scores bruts"):
            st.json(senti)

    # Exports
    payload = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "chars": len(text),
            "words": stats["words"],
            "reading_time": stats["reading_time"],
            "top_sentences": TOP_SENTENCES,
            "top_keywords": kwords,
        },
        "summary_paragraph": paragraph,
        "sentiment": senti,
        "sentiment_explanation": senti_expl,
        "keywords": keywords
    }
    jpath, mpath = save_json_md("outputs", payload, paragraph, keywords)
    st.success(f"Saved: {jpath}  and  {mpath}")

else:
    st.info("Drop a PDF / TXT / HTML file to start. For best results, prefer text-based PDFs (not scanned images).")
