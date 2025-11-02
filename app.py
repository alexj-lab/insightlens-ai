import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

STOP_WORDS = set([
    "the", "and", "of", "to", "in", "for", "on", "with", "a", "an", "by", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "but", "or", "as", "at", "from", "into", "through"
])

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .summary-box {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 1px 2px rgba(60,64,67,0.3);
    }
    .summary-text {
        color: #202124;
        font-size: 1.05rem;
        line-height: 1.8;
        text-align: justify;
    }
    .stat-card {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    .info-box {
        background: #e8f0fe;
        border-left: 4px solid #1967d2;
        border-radius: 4px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== I/O ==========
def read_pdf(path):
    try:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        raise ValueError(f"PDF error: {str(e)}")

def read_txt(path):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except:
            continue
    raise ValueError("Cannot decode file")

def read_html(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        raise ValueError(f"HTML error: {str(e)}")

def clean_text(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = read_pdf(path)
    elif ext in [".html", ".htm"]:
        text = read_html(path)
    else:
        text = read_txt(path)
    
    text = clean_text(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        st.warning(f"⚠️ Document truncated to {MAX_CHARS:,} characters")
    
    return text

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def optimized_summary(text, target_sentences=8):
    """Résumé optimisé court et pertinent"""
    sentences = split_sentences(text)
    
    if len(sentences) <= target_sentences:
        return " ".join(sentences)
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            max_features=10000
        )
        
        X = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(X)
        
        # TextRank scores
        scores = np.ones(len(sentences))
        for _ in range(15):
            new_scores = np.zeros(len(sentences))
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j and similarity_matrix[i][j] > 0.1:
                        new_scores[i] += similarity_matrix[i][j] * scores[j]
            scores = 0.85 * new_scores + 0.15
        
        # Position bonus
        position_bonus = np.array([1.0 - (i / len(sentences)) * 0.4 for i in range(len(sentences))])
        scores = scores * position_bonus
        
        # Sélection diverse
        top_indices = scores.argsort()[-target_sentences:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        selected = [sentences[i] for i in top_indices_sorted]
        return " ".join(selected)
    
    except:
        return " ".join(sentences[:target_sentences])

def extract_keywords(text, top_n=15):
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 50]
    
    if len(paragraphs) < 2:
        chunk_size = 500
        paragraphs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    if len(paragraphs) < 2:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 3),
            max_features=5000,
            min_df=1,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(paragraphs)
        
        if isinstance(X, (coo_matrix, csr_matrix)):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)
        
        scores_array = X_dense.max(axis=0)
        if len(scores_array.shape) > 1:
            scores_array = scores_array.flatten()
        
        terms_array = vectorizer.get_feature_names_out()
        
        pairs = []
        for i, term in enumerate(terms_array):
            try:
                score = float(scores_array[i])
                if len(str(term)) > 2 and score > 0:
                    pairs.append((str(term), score))
            except:
                continue
        
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
    
    except:
        return []

def analyze_sentiment(text):
    try:
        analyzer = load_sentiment_analyzer()
        chunk_size = 5000
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 20000), chunk_size)]
        
        sentiments = [analyzer.polarity_scores(chunk) for chunk in chunks]
        
        return {
            'positive': np.mean([s['pos'] for s in sentiments]),
            'neutral': np.mean([s['neu'] for s in sentiments]),
            'negative': np.mean([s['neg'] for s in sentiments]),
            'compound': np.mean([s['compound'] for s in sentiments])
        }
    except:
        return {'positive': 0, 'neutral': 1, 'negative': 0, 'compound': 0}

def readability_metrics(text):
    sentences = split_sentences(text)
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return {}
    
    return {
        'avg_sentence_length': len(words) / len(sentences),
        'long_words_ratio': len([w for w in words if len(w) > 6]) / len(words),
        'lexical_diversity': len(set([w.lower() for w in words])) / len(words),
        'total_words': len(words),
        'unique_words': len(set([w.lower() for w in words])),
        'total_sentences': len(sentences)
    }

def plot_sentiment(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#34a853', '#5f6368', '#ea4335']
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.9)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=13, fontweight='700')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability', fontsize=12, fontweight='600')
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='700', color='#1a73e8', pad=20)
    ax.grid(axis='y', alpha=0.15)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_keywords(pairs):
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    bars = ax.barh(range(len(terms)), scores, color='#1967d2', height=0.65)
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=10, fontweight='600')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11)
    ax.set_xlabel('TF-IDF Score', fontsize=12, fontweight='600')
    ax.set_title('Top Keywords', fontsize=14, fontweight='700', color='#1a73e8', pad=20)
    ax.grid(axis='x', alpha=0.15)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

# ========== INTERFACE ==========
st.set_page_config(page_title="InsightLens AI Pro", page_icon="🔍", layout="wide")

st.markdown('<p class="main-header">InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis System</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_keywords = st.slider("Keywords", 8, 20, 12)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Models")
    st.caption("• TextRank")
    st.caption("• TF-IDF")
    st.caption("• VADER")

uploaded_file = st.file_uploader("📂 Upload Document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("Reading...")
        progress.progress(20)
        text = read_any(temp_path)
        
        words = text.split()
        if len(words) < 100:
            st.error("❌ Document too short")
            st.stop()
        
        num_sent = len(split_sentences(text))
        st.success(f"✅ **{uploaded_file.name}** • {len(words):,} words")
        
        status.text("Generating summary...")
        progress.progress(40)
        summary = optimized_summary(text, target_sentences=8)
        
        status.text("Analyzing...")
        progress.progress(70)
        sentiment_scores = analyze_sentiment(text)
        keywords = extract_keywords(text, top_n=num_keywords)
        readability = readability_metrics(text)
        
        progress.progress(100)
        status.empty()
        progress.empty()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💭 Sentiment", "🔑 Keywords", "📊 Statistics"])
        
        with tab1:
            st.markdown("## Executive Summary")
            
            st.markdown("""
            <div class="info-box">
                <strong>Algorithm:</strong> TextRank (Graph-based)<br>
                <strong>Quality:</strong> Extracts 8 most important sentences
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("## Sentiment Analysis")
            
            st.markdown("""
            <div class="info-box">
                <strong>Model:</strong> VADER<br>
                <strong>Scale:</strong> -1 (negative) to +1 (positive)
            </div>
            """, unsafe_allow_html=True)
            
            fig_sent = plot_sentiment(sentiment_scores)
            if fig_sent:
                st.pyplot(fig_sent)
                plt.close(fig_sent)
            
            st.markdown("---")
            
            compound = sentiment_scores['compound']
            
            # Utiliser st.markdown SANS unsafe_allow_html
            if compound >= 0.05:
                st.success("### Overall: POSITIVE ✅")
            elif compound <= -0.05:
                st.error("### Overall: NEGATIVE ❌")
            else:
                st.info("### Overall: NEUTRAL ➖")
            
            st.markdown(f"**Compound Score:** {compound:.3f}")
            
            st.markdown("#### Distribution")
            st.markdown(f"- **Positive:** {sentiment_scores['positive']:.1%} - Favorable language")
            st.markdown(f"- **Neutral:** {sentiment_scores['neutral']:.1%} - Factual statements")
            st.markdown(f"- **Negative:** {sentiment_scores['negative']:.1%} - Critical language")
            
            st.markdown("#### Interpretation")
            if compound >= 0.5:
                st.markdown("**Very positive** document expressing strong favorable sentiment.")
            elif compound >= 0.05:
                st.markdown("**Moderately positive** document with constructive tone.")
            elif compound <= -0.5:
                st.markdown("**Very negative** document with critical concerns.")
            elif compound <= -0.05:
                st.markdown("**Moderately negative** document.")
            else:
                st.markdown("**Neutral** document maintaining objective tone.")
        
        with tab3:
            st.markdown("## Keywords")
            
            st.markdown("""
            <div class="info-box">
                <strong>Algorithm:</strong> TF-IDF<br>
                <strong>N-grams:</strong> 1-3 words
            </div>
            """, unsafe_allow_html=True)
            
            if len(keywords) > 0:
                fig_kw = plot_keywords(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close(fig_kw)
                
                st.markdown("---")
                for i, (term, score) in enumerate(keywords, 1):
                    importance = "🔥 Critical" if score > 0.5 else "⭐ High" if score > 0.3 else "📌 Moderate"
                    with st.expander(f"**{i}. {term}** • {score:.4f} • {importance}"):
                        st.markdown(f"**Score:** {score:.4f}")
            else:
                st.info("No keywords")
        
        with tab4:
            st.markdown("## Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{len(text):,}</div><div class="stat-label">Characters</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{readability["total_words"]:,}</div><div class="stat-label">Words</div></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{readability["total_sentences"]}</div><div class="stat-label">Sentences</div></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{readability["unique_words"]:,}</div><div class="stat-label">Unique</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Avg Sentence", f"{readability['avg_sentence_length']:.1f} words")
            
            with col_r2:
                st.
