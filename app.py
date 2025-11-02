import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20

# Style matplotlib
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

# CSS Ultra-propre
st.markdown("""
<style>
    /* Reset et base */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Summary container */
    .summary-container {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    }
    .summary-point {
        color: #202124;
        font-size: 1rem;
        line-height: 1.8;
        margin: 1rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    .summary-point:before {
        content: "•";
        position: absolute;
        left: 0;
        color: #1a73e8;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Stats */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: box-shadow 0.2s;
    }
    .stat-card:hover {
        box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #5f6368;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f0fe;
        border-left: 4px solid #1967d2;
        border-radius: 4px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        color: #174ea6;
    }
    .info-box strong {
        color: #1967d2;
    }
    
    /* Sentiment explanation */
    .sentiment-box {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        line-height: 1.7;
        color: #202124;
    }
    .sentiment-box h4 {
        color: #1a73e8;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .badge-positive {
        background: #e6f4ea;
        color: #137333;
    }
    .badge-neutral {
        background: #f1f3f4;
        color: #5f6368;
    }
    .badge-negative {
        background: #fce8e6;
        color: #c5221f;
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

# ========== CACHE ==========
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# ========== ANALYSES ==========
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def advanced_extractive_summary(text, num_sentences=10):
    """
    Résumé extractif avancé avec TF-IDF + MMR (Maximal Marginal Relevance)
    pour diversité et qualité maximale
    """
    sentences = split_sentences(text)
    
    if len(sentences) <= num_sentences:
        return sentences
    
    try:
        # Vectorisation TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            max_features=10000,
            min_df=1,
            max_df=0.85
        )
        
        X = vectorizer.fit_transform(sentences)
        
        # Calculer scores TF-IDF moyens par phrase
        tfidf_scores = np.asarray(X.mean(axis=1)).flatten()
        
        # Bonus pour position (premières phrases souvent importantes)
        position_scores = np.array([1.0 - (i / len(sentences)) * 0.4 for i in range(len(sentences))])
        
        # Score combiné
        combined_scores = tfidf_scores * position_scores
        
        # Sélectionner top phrases avec MMR pour diversité
        selected_indices = []
        remaining_indices = list(range(len(sentences)))
        
        # Première phrase = meilleur score
        first_idx = combined_scores.argmax()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Sélection itérative avec MMR
        for _ in range(num_sentences - 1):
            if not remaining_indices:
                break
            
            best_score = -1
            best_idx = None
            
            for idx in remaining_indices:
                # Score de pertinence
                relevance = combined_scores[idx]
                
                # Pénalité de similarité avec phrases déjà sélectionnées
                similarities = []
                for sel_idx in selected_indices:
                    sim = np.dot(X[idx].toarray().flatten(), X[sel_idx].toarray().flatten())
                    similarities.append(sim)
                
                max_similarity = max(similarities) if similarities else 0
                
                # MMR score (balance pertinence et diversité)
                mmr_score = 0.7 * relevance - 0.3 * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Trier par ordre d'apparition
        selected_indices.sort()
        
        return [sentences[i] for i in selected_indices]
    
    except Exception as e:
        st.error(f"Summary error: {e}")
        # Fallback simple
        return sentences[:num_sentences]

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
        
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]
    
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

def generate_sentiment_explanation(scores):
    """Génère explication détaillée"""
    compound = scores['compound']
    pos = scores['positive']
    neu = scores['neutral']
    neg = scores['negative']
    
    # Déterminer badge
    if compound >= 0.05:
        badge = '<span class="badge badge-positive">POSITIVE</span>'
        tone = "positive"
    elif compound <= -0.05:
        badge = '<span class="badge badge-negative">NEGATIVE</span>'
        tone = "negative"
    else:
        badge = '<span class="badge badge-neutral">NEUTRAL</span>'
        tone = "neutral"
    
    explanation = f"""
<div class="sentiment-box">
    <h3>Overall Assessment {badge}</h3>
    <p><strong>Compound Score: {compound:.3f}</strong> (scale: -1 to +1)</p>
    
    <h4>📊 Distribution Breakdown</h4>
    
    <p><strong>Positive Content ({pos:.1%}):</strong><br>
    Language expressing favorable opinions, satisfaction, praise, or optimistic perspectives. 
    {'This is the <strong>dominant emotion</strong> in the document.' if pos == max(pos, neu, neg) else ''}</p>
    
    <p><strong>Neutral Content ({neu:.1%}):</strong><br>
    Factual statements, objective descriptions, and balanced language without strong emotional coloring.
    {'This is the <strong>dominant style</strong> in the document.' if neu == max(pos, neu, neg) else ''}</p>
    
    <p><strong>Negative Content ({neg:.1%}):</strong><br>
    Language expressing criticism, concerns, dissatisfaction, or problematic issues.
    {'This is the <strong>dominant emotion</strong> in the document.' if neg == max(pos, neu, neg) else ''}</p>
    
    <h4>💡 Interpretation</h4>
    <p>
"""
    
    if compound >= 0.5:
        explanation += "<strong>Very Positive Document:</strong> Expresses strong favorable sentiment, enthusiasm, or highly positive outcomes."
    elif compound >= 0.05:
        explanation += "<strong>Moderately Positive Document:</strong> Generally constructive with more favorable than unfavorable content."
    elif compound <= -0.5:
        explanation += "<strong>Very Negative Document:</strong> Expresses strong criticism, serious concerns, or negative outcomes."
    elif compound <= -0.05:
        explanation += "<strong>Moderately Negative Document:</strong> Contains criticism or concerns with more unfavorable than favorable content."
    else:
        explanation += "<strong>Neutral Document:</strong> Maintains objective, balanced tone. Typical of factual reports or technical documents."
    
    explanation += """
    </p>
    
    <h4>📖 Context</h4>
    <p>VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzes text by examining individual words, 
    punctuation patterns, capitalization, and contextual modifiers. It's particularly effective for business documents, 
    news articles, and social media content.</p>
</div>
"""
    
    return explanation

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

# ========== VISUALISATIONS ==========
def plot_sentiment(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#34a853', '#5f6368', '#ea4335']
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none', alpha=0.9)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=13, fontweight='700', color='#202124')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability', fontsize=12, fontweight='600', color='#5f6368')
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='700', color='#1a73e8', pad=20)
    ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#dadce0')
    
    ax.tick_params(colors='#5f6368')
    plt.tight_layout()
    return fig

def plot_keywords(pairs):
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    bars = ax.barh(range(len(terms)), scores, color='#1967d2', height=0.65, edgecolor='none')
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=10, fontweight='600', color='#5f6368')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11, color='#202124')
    ax.set_xlabel('TF-IDF Score', fontsize=12, fontweight='600', color='#5f6368')
    ax.set_title('Top Keywords & Phrases', fontsize=14, fontweight='700', color='#1a73e8', pad=20)
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.5)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#dadce0')
    
    ax.tick_params(colors='#5f6368')
    plt.tight_layout()
    return fig

# ========== INTERFACE ==========
st.set_page_config(page_title="InsightLens AI Pro", page_icon="🔍", layout="wide")

st.markdown('<p class="main-header">InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis & Intelligence System</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    num_summary = st.slider("Summary sentences", 5, 15, 10, help="Number of key sentences to extract")
    num_keywords = st.slider("Keywords", 8, 20, 12, help="Number of distinctive terms")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Models")
    st.caption("• Advanced TF-IDF + MMR")
    st.caption("• VADER Sentiment")
    st.caption("• Statistical Analysis")

uploaded_file = st.file_uploader("📂 Upload Document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("📖 Reading document...")
        progress.progress(20)
        text = read_any(temp_path)
        
        words = text.split()
        if len(words) < 100:
            st.error("❌ Document too short (minimum 100 words)")
            st.stop()
        
        num_sentences = len(split_sentences(text))
        st.success(f"✅ **{uploaded_file.name}** • {len(words):,} words • {num_sentences} sentences")
        
        status.text("🔍 Generating summary...")
        progress.progress(40)
        summary = advanced_extractive_summary(text, num_sentences=num_summary)
        
        status.text("💭 Analyzing sentiment...")
        progress.progress(60)
        sentiment_scores = analyze_sentiment(text)
        sentiment_explanation = generate_sentiment_explanation(sentiment_scores)
        
        status.text("🔑 Extracting keywords...")
        progress.progress(80)
        keywords = extract_keywords(text, top_n=num_keywords)
        
        status.text("📊 Computing metrics...")
        progress.progress(95)
        readability = readability_metrics(text)
        
        progress.progress(100)
        status.empty()
        progress.empty()
        
        # ========== TABS ==========
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💭 Sentiment", "🔑 Keywords", "📊 Statistics"])
        
        # TAB 1: SUMMARY
        with tab1:
            st.markdown("## Executive Summary")
            
            st.markdown("""
            <div class="info-box">
                <strong>Method:</strong> Advanced Extractive Summarization (TF-IDF + MMR)<br>
                <strong>Algorithm:</strong> Combines statistical importance with maximal marginal relevance for diversity<br>
                <strong>Quality:</strong> Selects most informative sentences while avoiding redundancy
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="summary-container">', unsafe_allow_html=True)
            for sentence in summary:
                st.markdown(f'<p class="summary-point">{sentence}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### How This Works")
            st.markdown("""
            This summary uses **Advanced Extractive Summarization** combining:
            
            1. **TF-IDF Scoring**: Identifies sentences with the most important terms
            2. **Position Weighting**: Gives priority to sentences at the beginning (often more important)
            3. **MMR (Maximal Marginal Relevance)**: Ensures diversity by avoiding similar sentences
            
            The result is a concise summary that captures key information without redundancy.
            """)
        
        # TAB 2: SENTIMENT
        with tab2:
            st.markdown("## Sentiment Analysis")
            
            st.markdown("""
            <div class="info-box">
                <strong>Model:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)<br>
                <strong>Method:</strong> Lexicon-based with contextual awareness<br>
                <strong>Scale:</strong> Compound score from -1 (very negative) to +1 (very positive)
            </div>
            """, unsafe_allow_html=True)
            
            fig_sent = plot_sentiment(sentiment_scores)
            if fig_sent:
                st.pyplot(fig_sent)
                plt.close(fig_sent)
            
            st.markdown("---")
            st.markdown(sentiment_explanation, unsafe_allow_html=True)
        
        # TAB 3: KEYWORDS
        with tab3:
            st.markdown("## Keywords & Phrases")
            
            st.markdown("""
            <div class="info-box">
                <strong>Algorithm:</strong> TF-IDF (Term Frequency-Inverse Document Frequency)<br>
                <strong>N-grams:</strong> 1-3 word combinations<br>
                <strong>Purpose:</strong> Identify terms most characteristic of this document
            </div>
            """, unsafe_allow_html=True)
            
            if len(keywords) > 0:
                fig_kw = plot_keywords(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close(fig_kw)
                
                st.markdown("---")
                st.markdown("### Keyword Details")
                
                for i, (term, score) in enumerate(keywords, 1):
                    if score > 0.5:
                        importance = "🔥 Critical"
                    elif score > 0.3:
                        importance = "⭐ High"
                    elif score > 0.15:
                        importance = "📌 Moderate"
                    else:
                        importance = "📎 Minor"
                    
                    with st.expander(f"**{i}. {term}** • Score: `{score:.4f}` • {importance}"):
                        st.markdown(f"**TF-IDF Score:** {score:.4f}")
                        st.markdown(f"**Importance:** {importance}")
                        st.markdown("This term is distinctive to this document compared to typical documents.")
            else:
                st.info("No keywords extracted")
        
        # TAB 4: STATISTICS
        with tab4:
            st.markdown("## Document Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(text):,}</div>
                    <div class="stat-label">Characters</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{readability['total_words']:,}</div>
                    <div class="stat-label">Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{readability['total_sentences']}</div>
                    <div class="stat-label">Sentences</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{readability['unique_words']:,}</div>
                    <div class="stat-label">Unique Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Readability Metrics")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            avg_len = readability['avg_sentence_length']
            
            with col_r1:
                st.markdown("#### Sentence Complexity")
                st.metric("Avg Length", f"{avg_len:.1f} words")
                if avg_len < 15:
                    st.caption("✅ Simple (Easy to read)")
                elif avg_len < 25:
                    st.caption("📖 Medium (Standard)")
                else:
                    st.caption("📚 Complex (Advanced)")
            
            with col_r2:
                st.markdown("#### Vocabulary")
                st.metric("Diversity", f"{readability['lexical_diversity']:.1%}")
                st.caption(f"{readability['unique_words']:,} unique words")
            
            with col_r3:
                st.markdown("#### Reading Time")
                reading_time = readability['total_words'] / 200
                st.metric("Estimated", f"{reading_time:.1f} min")
                st.caption("At 200 words/min")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Debug"):
            st.code(str(e))

else:
    st.info("👆 Upload a document to start analysis")
    
    st.markdown("""
    ### 🎯 Features
    
    - **Advanced Summarization**: TF-IDF + MMR for optimal sentence selection
    - **Sentiment Analysis**: VADER with detailed explanations
    - **Keyword Extraction**: Multi-gram TF-IDF analysis
    - **Document Statistics**: Comprehensive readability metrics
    
    ### 📄 Supported Formats
    
    - PDF (text-based)
    - TXT (plain text)
    - HTML (web pages)
    
    ### ✅ Best For
    
    Business reports • Research papers • Legal documents • News articles • Financial reports
    """)
