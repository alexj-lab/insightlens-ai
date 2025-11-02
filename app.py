import os, re, json
from datetime import datetime

import streamlit as st
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────────────────────
# App constants & page config
# ─────────────────────────────────────────────────────────────
APP_NAME = "InsightLens – AI Document Analyzer"
MAX_CHARS = 80000
MIN_SENT_LEN = 22
TARGET_SENTENCES = 10         # candidates before rewrite
WPM = 200                     # reading speed words/min
TOP_KEYWORDS = 15

st.set_page_config(page_title="InsightLens", layout="wide", page_icon="🧾")

# Simple professional theme
st.markdown("""
<style>
:root { --brand:#0F62FE; --ink:#0f172a; --muted:#6b7280; --bg:#f7f8fa; --line:#e5e7eb; --card:#ffffff; }
html, body, [class*="block-container"] { background:var(--bg); color:var(--ink); }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 20px; box-shadow:0 1px 2px rgba(17,24,39,.05); }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; text-align:center; }
.kpi .v { font-size:1.2rem; font-weight:700; color:var(--brand); }
.kpi .l { font-size:.85rem; color:var(--muted); }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--line); background:#fff; font-size:.8rem; }
.summary { line-height:1.75; font-size:1.02rem; }
.small { color:var(--muted); font-size:.9rem; }
hr { border:0; border-top:1px solid var(--line); margin:8px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Stopwords (EN + FR basics, lightweight)
# ─────────────────────────────────────────────────────────────
STOP_WORDS = set("""
the and of to in for on with a an by is are was were be been being have has had do does did
will would should could may might must can this that these those i you he she it we they what which who when
where why how all each every both few more most other some such no nor not only own same so than too very
just but or as at from into through
le la les de des du un une et à au aux en dans sur pour par avec sans ce cet cette ces se ses son sa leurs
leur plus moins a ont est sont été être fait faire afin ainsi donc car ou où
""".split())

# ─────────────────────────────────────────────────────────────
# File IO utils
# ─────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    try:
        out = []
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                try: doc.authenticate("")
                except Exception: return ""
            for p in doc:
                t = p.get_text("text")
                if t and t.strip():
                    out.append(t)
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
        for t in soup(["script","style"]): t.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def clean_text(x: str) -> str:
    x = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', x)
    x = re.sub(r'\r\n?', '\n', x)
    x = re.sub(r'\n{3,}', '\n\n', x)
    x = re.sub(r'[ \t]+', ' ', x)
    return x.strip()

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        t = read_pdf(path)
    elif ext in (".html", ".htm"):
        t = read_html(path)
    else:
        t = read_txt(path)
    t = clean_text(t)
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS]
        st.toast(f"Document truncated to {MAX_CHARS:,} characters", icon="⚠️")
    return t

# ─────────────────────────────────────────────────────────────
# Sentence segmentation + heading filter
# ─────────────────────────────────────────────────────────────
BULLET = re.compile(r'^\s*[\-\•\–\*]\s+')
HEADING_LEAD = re.compile(r'^(part|section|phase|case|role|assessment|timeline|procedure|status|students|team|judgment|simulation|annex|appendix)\b', re.IGNORECASE)
ALL_CAPS_TOKEN = re.compile(r'^[A-Z0-9\W]+$')
QUOTE_LINE = re.compile(r'^([“"].+[”"])$')

def is_heading_like(line: str) -> bool:
    if not line: return False
    s = line.strip()
    if len(s) <= 8: return True
    if s.endswith(':'): return True
    if BULLET.match(s): return True
    if HEADING_LEAD.match(s): return True
    letters = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]', '', s)
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / max(len(letters),1)
        if upper_ratio > 0.65: return True
    toks = s.split()
    if len(toks) <= 6 and all(ALL_CAPS_TOKEN.match(t) for t in toks): return True
    if QUOTE_LINE.match(s): return True
    return False

def split_sentences(text: str):
    # Group lines, discard pure headings/labels
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts, buf = [], []
    for ln in lines:
        if is_heading_like(ln):
            if buf:
                parts.append(" ".join(buf)); buf = []
            continue
        buf.append(ln)
    if buf: parts.append(" ".join(buf))

    # Sentence splitting (keep only informative sentences)
    sents = []
    for p in parts:
        # split after ., !, ? followed by a capital (incl. accented caps)
        chunks = re.split(r'(?<=[\.\!\?])\s+(?=[A-ZÉÈÊÀÂÎÔÛÇ])', p)
        for c in chunks:
            c = c.strip()
            if not c: continue
            if len(c) >= MIN_SENT_LEN and not is_heading_like(c):
                sents.append(c)

    # Deduplicate (case-insensitive)
    seen, clean = set(), []
    for s in sents:
        k = s.lower()
        if k not in seen:
            seen.add(k); clean.append(s)
    return clean

# ─────────────────────────────────────────────────────────────
# Summarization: scoring + MMR + rewrite → single paragraph
# ─────────────────────────────────────────────────────────────
VERB_HINT = re.compile(r'\b(is|are|was|were|be|been|being|has|have|had|do|does|did|will|would|should|could|may|might|must|can|invest|argue|claim|decide|deliver|terminate|classify|apply|provide|purchase|sell|own|hire|open|approve|seek|grant|compensate|represent|notify|renew|calculate|measure|analyze|assess|consider)\b', re.IGNORECASE)

def sentence_quality(s: str) -> float:
    q = 0.0
    if VERB_HINT.search(s): q += 1.0
    if len(s) > 60: q += 0.4
    if not is_heading_like(s): q += 0.8
    if s.endswith(':'): q -= 0.5
    return q

def tfidf_matrix(sentences, max_features=12000):
    vec = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1,2), max_features=max_features)
    X = vec.fit_transform(sentences)  # sparse
    return vec, X

def position_bonus(n, i):
    return 1.0 - 0.35 * (i / max(n-1, 1))

def mmr_select(sim, relevance, k, diversity=0.7):
    """sim: dense NxN cosine_similarity matrix; relevance: dense (N,) array."""
    N = len(relevance)
    if N == 0: return []
    selected = [int(np.argmax(relevance))]
    candidates = set(range(N)) - set(selected)
    while len(selected) < min(k, N) and candidates:
        best, best_val = None, -1e9
        for i in candidates:
            redundancy = max(sim[i, selected]) if selected else 0.0
            val = (1 - diversity) * relevance[i] - diversity * redundancy
            if val > best_val:
                best_val, best = val, i
        selected.append(best)
        candidates.remove(best)
    return sorted(selected)

def rewrite_sentence(s: str) -> str:
    x = s.strip()
    x = re.sub(r'\s*\([^)]*\)', '', x)  # drop parentheticals
    x = re.sub(r'^(However|Moreover|Furthermore|Additionally|In addition|Thus|Therefore)[,:]\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'^(According to|Per|As noted)\s+[^:]+:\s+', '', x, flags=re.IGNORECASE)
    x = re.sub(r'\s*,\s*(which|that)\s+', ' ', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(due to|because of|owing to)\b', 'because of', x, flags=re.IGNORECASE)
    x = re.sub(r'\b(in order to)\b', 'to', x, flags=re.IGNORECASE)
    x = re.sub(r'\s{2,}', ' ', x).strip(' -–—;:,')
    if not re.search(r'[.!?]$', x): x += '.'
    return x

def summarize_paragraph(text: str, target_sentences=TARGET_SENTENCES) -> (str, list):
    sents_all = split_sentences(text)
    if not sents_all:
        return "This document discusses the main facts, issues, evidence and conclusions.", []

    # filter for informative sentences
    sents = [s for s in sents_all if sentence_quality(s) >= 1.2]
    if len(sents) < 5:
        sents = sents_all[:max(5, target_sentences)]

    N = len(sents)
    # TF-IDF
    vec, X = tfidf_matrix(sents, max_features=12000)

    # dense centrality proxy = row max TF-IDF
    centrality = X.max(axis=1).A.ravel()  # safe dense array
    # title similarity: use first non-heading line as "title"
    title = ""
    for ln in text.splitlines():
        t = ln.strip()
        if t and not is_heading_like(t):
            title = t
            break
    title_sim = np.zeros(N)
    if title:
        try:
            Xt = vec.transform([title])
            title_sim = cosine_similarity(X, Xt).ravel()
        except Exception:
            title_sim = np.zeros(N)

    pos_bonus = np.array([position_bonus(N, i) for i in range(N)], dtype=float)
    qual_bonus = np.array([sentence_quality(s) for s in sents], dtype=float) * 0.15

    # dense relevance (no sparse-scalar ops)
    relevance = 0.55 * centrality + 0.20 * pos_bonus + 0.10 * title_sim + 0.15 * qual_bonus

    # dense cosine sim over sentences
    sim = cosine_similarity(X)  # dense NxN

    idx = mmr_select(sim, relevance, k=target_sentences, diversity=0.7)
    chosen = [rewrite_sentence(sents[i]) for i in idx]

    # compact, dedupe, keep order
    seen, ordered = set(), []
    for s in chosen:
        k = s.lower()
        if k not in seen and len(s) >= 25:
            seen.add(k)
            ordered.append(s)
    if not ordered:
        ordered = [rewrite_sentence(s) for s in sents[:5]]

    # build single paragraph: “This document discusses …”
    cleaned = [re.sub(r'[ \t]*\.$', '', s).strip() for s in ordered]
    connectors = [" It then explains", " Additionally,", " Moreover,", " Finally,"]
    lead = "This document discusses "
    parts = [lead + cleaned[0].lstrip("-.• ")]
    for i, sent in enumerate(cleaned[1:], start=1):
        parts.append(f"{connectors[min(i-1, len(connectors)-1)]} {sent.lstrip('-.• ')}")
    paragraph = " ".join(parts).strip()
    if not paragraph.endswith("."): paragraph += "."
    return paragraph, ordered

# ─────────────────────────────────────────────────────────────
# Keywords, sentiment, stats
# ─────────────────────────────────────────────────────────────
def extract_keywords(text: str, top_n=TOP_KEYWORDS):
    # segment by paragraphs or fixed windows
    paras = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 60]
    if len(paras) < 2:
        chunk = 700
        paras = [text[i:i+chunk] for i in range(0, len(text), chunk)]
    if not paras: paras = [text]

    vec = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1,3), max_features=6000, min_df=1, max_df=0.85)
    X = vec.fit_transform(paras)
    if X.shape[1] == 0: return []

    arr = X.toarray()  # dense
    scores = arr.max(axis=0)
    terms = vec.get_feature_names_out()

    pairs = []
    for t, s in zip(terms, scores):
        t_stripped = t.strip()
        if len(t_stripped) < 3: continue
        if re.fullmatch(r'[\W_]+', t_stripped): continue
        if re.fullmatch(r'\d+(\.\d+)?', t_stripped): continue
        pairs.append((t_stripped, float(s)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    out, seen = [], set()
    for t, s in pairs:
        low = t.lower()
        if any(low in k or k in low for k in seen): continue
        seen.add(low); out.append((t, s))
        if len(out) >= top_n: break
    return out

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

def overall_sentiment(text: str):
    a = load_vader()
    s = a.polarity_scores(text[:6000])
    pos = s.get("pos", 0.0); neu = s.get("neu", 0.0); neg = s.get("neg", 0.0)
    tot = max(pos+neu+neg, 1e-9)
    return {
        "positive": pos/tot,
        "neutral": neu/tot,
        "negative": neg/tot,
        "compound": s.get("compound", 0.0)
    }

def text_stats(text: str):
    words = re.findall(r"\b\w+\b", text)
    wc = len(words); cc = len(text)
    paras = len([p for p in re.split(r"\n{2,}", text) if p.strip()])
    minutes = wc / WPM; mm = int(minutes); ss = int((minutes-mm)*60)
    rt = f"{mm}m{ss:02d}s" if mm else f"{ss}s"
    return {"words": wc, "chars": cc, "paragraphs": paras, "reading_time": rt}

# ─────────────────────────────────────────────────────────────
# Export helpers
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
        "", "## Summary (paragraph)", paragraph if paragraph else "(empty)", "",
        "## Top Keywords", ", ".join([t for t,_ in keywords]) if keywords else "(none)", ""
    ]
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return jpath, mpath

# ─────────────────────────────────────────────────────────────
# App w/ tabs
# ─────────────────────────────────────────────────────────────
st.title("🧾 " + APP_NAME)
st.caption("Upload a PDF / TXT / HTML. Get a clean executive paragraph, keywords, sentiment, and reading stats.")

with st.sidebar:
    st.header("Settings")
    kw_n = st.slider("Top keywords", 5, 30, TOP_KEYWORDS)
    st.caption("Tip: Prefer text-based PDFs (not scanned).")

upl = st.file_uploader("Upload document", type=["pdf","txt","html","htm"])

if not upl:
    st.info("Drop a PDF / TXT / HTML file to start.")
else:
    tmp = f"/tmp/{upl.name}"
    with open(tmp, "wb") as f:
        f.write(upl.read())

    text = read_any(tmp)
    if len(text.strip()) < 200:
        st.error("This file seems empty, scanned, or too short. Please use a text-based PDF or a .txt/.html file.")
        st.stop()

    stats = text_stats(text)

    with st.spinner("Analyzing document…"):
        paragraph, chosen_sentences = summarize_paragraph(text, target_sentences=TARGET_SENTENCES)
        senti = overall_sentiment(text)
        keywords = extract_keywords(text, top_n=kw_n)

    # Tabs
    t1, t2, t3, t4 = st.tabs(["Overview", "Summary", "Keywords", "Stats & Export"])

    with t1:
        st.subheader("What this tool does")
        st.markdown('<div class="card small">Upload a document to generate a single executive paragraph that starts with “This document discusses …”. Headings and list labels are ignored; only informative sentences are used.</div>', unsafe_allow_html=True)
        st.markdown("**How it works**")
        st.markdown('<div class="card small">We score sentences by TF-IDF importance, early position, title similarity, and linguistic quality (presence of verbs), then remove redundancies (MMR) and rewrite the selection into one cohesive paragraph.</div>', unsafe_allow_html=True)

    with t2:
        st.subheader("Executive Summary")
        st.markdown(f'<div class="card summary">{paragraph}</div>', unsafe_allow_html=True)
        st.caption("Optional: sanity-check the inputs used for the summary below.")
        st.markdown('<div class="card small">', unsafe_allow_html=True)
        for s in chosen_sentences:
            st.write("• " + s)
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.subheader("Top Keywords")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if keywords:
            terms = [t for t,_ in keywords[:15]]; vals = [v for _,v in keywords[:15]]
            fig = plt.figure(figsize=(6.5,3.6))
            plt.barh(terms[::-1], vals[::-1])
            plt.xlabel("TF-IDF (max across segments)")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("_No keywords detected._")
        st.markdown('</div>', unsafe_allow_html=True)

    with t4:
        st.subheader("Document Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi"><div class="v">{stats["words"]:,}</div><div class="l">words</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi"><div class="v">{stats["chars"]:,}</div><div class="l">characters</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi"><div class="v">{stats["paragraphs"]}</div><div class="l">paragraphs</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi"><div class="v">{stats["reading_time"]}</div><div class="l">reading time</div></div>', unsafe_allow_html=True)

        st.subheader("Export")
        payload = {
            "meta": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "chars": len(text),
                "words": stats["words"],
                "reading_time": stats["reading_time"],
                "target_sentences": TARGET_SENTENCES,
                "top_keywords": kw_n
            },
            "summary_paragraph": paragraph,
            "keywords": keywords,
            "sentiment": senti
        }
        jp, mp = save_json_md("outputs", payload, paragraph, keywords)
        st.markdown(f'<div class="card small">Saved reports: <span class="badge">JSON</span> {jp} &nbsp; · &nbsp; <span class="badge">Markdown</span> {mp}</div>', unsafe_allow_html=True)

        label = "Positive" if senti["compound"] >= 0.2 else "Negative" if senti["compound"] <= -0.2 else "Neutral"
        st.caption(f"Overall sentiment (VADER): {label} (compound={senti['compound']:.3f})")
