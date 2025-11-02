import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
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

# Style matplotlib professionnel
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

STOP_WORDS = set([
    "the", "and", "of", "to", "in", "for", "on", "with", "a", "an", "by", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "but", "or", "as", "at", "from", "into", "through",
    "les", "des", "et", "de", "la", "le", "pour", "dans", "sur", "avec", "par",
    "aux", "au", "une", "un", "du", "est", "sont", "ont", "a", "ce", "cette", "ces"
])

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .summary-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: white;
        border-left: 5px solid rgba(255,255,255,0.5);
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #2193b0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== FONCTIONS I/O ==========
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
        raise ValueError(f"PDF reading error: {str(e)}")

def read_txt(path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except:
            continue
    raise ValueError("Cannot decode text file")

def read_html(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        raise ValueError(f"HTML reading error: {str(e)}")

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

# ========== CACHE DES MODÈLES ==========
@st.cache_resource
def load_summarizer():
    """Charge le modèle BART une seule fois"""
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

@st.cache_resource
def load_sentiment_analyzer():
    """Charge VADER une seule fois"""
    return SentimentIntensityAnalyzer()

# ========== ANALYSES TEXTUELLES ==========
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def abstractive_summary(text, num_points=5):
    """Génère un vrai résumé avec BART (abstractif)"""
    try:
        summarizer = load_summarizer()
        
        # BART accepte max 1024 tokens (~4000 chars)
        input_text = text[:4000]
        
        # Générer résumé
        result = summarizer(
            input_text,
            max_length=150,
            min_length=50,
            do_sample=False,
            truncation=True
        )
        
        summary_text = result[0]['summary_text']
        
        # Découper en phrases
        sentences = re.split(r'(?<=[.!?])\s+', summary_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Retourner jusqu'à num_points phrases
        return sentences[:num_points] if len(sentences) > 0 else ["Summary generation successful but empty."]
    
    except Exception as e:
        st.error(f"Summary generation error: {str(e)}")
        # Fallback: résumé extractif simple
        sentences = split_sentences(text)
        return sentences[:num_points] if len(sentences) > 0 else ["Error generating summary."]

def extract_keywords(text, top_n=20):
    """Extraction mots-clés TF-IDF"""
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
    
    except Exception as e:
        return []

def analyze_sentiment(text):
    """Analyse sentiment VADER"""
    try:
        analyzer = load_sentiment_analyzer()
        chunk_size = 5000
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 20000), chunk_size)]
        
        sentiments = [analyzer.polarity_scores(chunk) for chunk in chunks]
        
        avg_scores = {
            'positive': np.mean([s['pos'] for s in sentiments]),
            'neutral': np.mean([s['neu'] for s in sentiments]),
            'negative': np.mean([s['neg'] for s in sentiments]),
            'compound': np.mean([s['compound'] for s in sentiments])
        }
        return avg_scores
    except:
        return {'positive': 0, 'neutral': 1, 'negative': 0, 'compound': 0}

def extract_entities(text):
    """Extraction entités"""
    entities = {}
    
    money_pattern = r'(?:€|USD|\$|EUR)\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?|\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\s*(?:€|EUR|USD|\$|million|billion|M|B|bn|mn)'
    money_matches = re.findall(money_pattern, text, re.I)
    valid_money = [m for m in money_matches if re.search(r'[€$]|million|billion|EUR|USD|[MB](?:n)?$', m, re.I)]
    if valid_money:
        entities['monetary_values'] = list(set(valid_money))[:12]
    
    percent_matches = re.findall(r'\b\d+(?:\.\d+)?%', text)
    if percent_matches:
        entities['percentages'] = list(set(percent_matches))[:12]
    
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
    date_matches = re.findall(date_pattern, text, re.I)
    if date_matches:
        entities['dates'] = list(set(date_matches))[:10]
    
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    name_matches = re.findall(name_pattern, text)
    valid_names = [n for n in name_matches if len(n.split()) <= 4]
    if valid_names:
        name_counter = Counter(valid_names)
        entities['proper_names'] = [name for name, count in name_counter.most_common(10)]
    
    return entities

def word_frequency_analysis(text, top_n=25):
    """Fréquence des mots"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in STOP_WORDS]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

def readability_metrics(text):
    """Métriques de lisibilité"""
    sentences = split_sentences(text)
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return {}
    
    avg_sentence_length = len(words) / len(sentences)
    long_words = len([w for w in words if len(w) > 6])
    unique_words = len(set([w.lower() for w in words]))
    lexical_diversity = unique_words / len(words) if len(words) > 0 else 0
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'long_words_ratio': long_words / len(words) if len(words) > 0 else 0,
        'lexical_diversity': lexical_diversity,
        'total_words': len(words),
        'unique_words': unique_words,
        'total_sentences': len(sentences)
    }

# ========== VISUALISATIONS ==========
def plot_keywords_professional(pairs):
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(terms)))
    bars = ax.barh(range(len(terms)), scores, color=colors, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11)
    ax.set_xlabel('TF-IDF Score', fontsize=11, fontweight='bold')
    ax.set_title('🔑 Top Keywords', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_sentiment_advanced(scores):
    labels = ['Positive\n😊', 'Neutral\n😐', 'Negative\n😟']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    bars = ax1.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax1.set_title('💭 Sentiment Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.pie(values, labels=['Positive', 'Neutral', 'Negative'], colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('📊 Breakdown', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_word_frequency_advanced(word_freq):
    if len(word_freq) == 0:
        return None
    
    words = [w for w, _ in word_freq[:20]]
    counts = [c for _, c in word_freq[:20]]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(words)))
    bars = ax.barh(range(len(words)), counts, color=colors, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words[::-1], fontsize=10)
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('📈 Most Frequent Words', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_readability_gauge(metrics):
    if not metrics:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    ax = axes[0]
    avg_len = metrics['avg_sentence_length']
    complexity = min(avg_len / 30, 1.0)
    
    ax.barh([0], [complexity], color='#3498db', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Simple', 'Medium', 'Complex'])
    ax.set_yticks([])
    ax.set_title(f'Sentence Length\n({avg_len:.1f} words)', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax = axes[1]
    long_ratio = metrics['long_words_ratio']
    
    ax.barh([0], [long_ratio], color='#e74c3c', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Few', 'Medium', 'Many'])
    ax.set_yticks([])
    ax.set_title(f'Long Words\n({long_ratio:.1%})', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax = axes[2]
    diversity = metrics['lexical_diversity']
    
    ax.barh([0], [diversity], color='#2ecc71', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_yticks([])
    ax.set_title(f'Vocabulary Diversity\n({diversity:.1%})', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ========== INTERFACE ==========
st.set_page_config(
    page_title="InsightLens AI Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<p class="main-title">🔍 InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis with AI-Generated Summaries</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    
    with st.expander("📋 Summary (BART AI)", expanded=True):
        st.markdown("**Abstractive summarization**: AI generates NEW text (not extraction)")
        num_summary = st.slider("Summary sentences", 3, 8, 5)
    
    with st.expander("🔑 Keywords (TF-IDF)", expanded=True):
        st.markdown("**Extraction**: Identifies distinctive terms")
        num_keywords = st.slider("Keywords count", 5, 25, 15)
    
    with st.expander("💭 Analysis Options", expanded=True):
        show_sentiment = st.checkbox("Sentiment Analysis", value=True)
        show_entities = st.checkbox("Entity Extraction", value=True)
        show_readability = st.checkbox("Readability Metrics", value=True)
        show_word_freq = st.checkbox("Word Frequency", value=False)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Models")
    st.caption("• BART (Facebook AI) - Summarization")
    st.caption("• VADER - Sentiment Analysis")
    st.caption("• TF-IDF - Keyword Extraction")

uploaded_file = st.file_uploader("📂 Upload Document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📖 Reading document...")
        progress_bar.progress(20)
        text = read_any(temp_path)
        
        words = text.split()
        if len(words) < 50:
            st.error("❌ Document too short (< 50 words)")
            st.stop()
        
        num_sentences = len(split_sentences(text))
        num_chars = len(text)
        
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Document loaded!</strong><br>
            📄 {uploaded_file.name} • {num_chars:,} chars • {len(words):,} words • {num_sentences} sentences
        </div>
        """, unsafe_allow_html=True)
        
        status_text.text("🤖 Generating AI summary with BART...")
        progress_bar.progress(40)
        
        summary = abstractive_summary(text, num_points=num_summary)
        
        progress_bar.progress(60)
        keywords = extract_keywords(text, top_n=num_keywords)
        
        progress_bar.progress(75)
        
        sentiment_scores = None
        if show_sentiment:
            sentiment_scores = analyze_sentiment(text)
        
        entities = None
        if show_entities:
            entities = extract_entities(text)
        
        readability = None
        if show_readability:
            readability = readability_metrics(text)
        
        word_freq = None
        if show_word_freq:
            word_freq = word_frequency_analysis(text)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        if len(summary) == 0:
            st.error("❌ Could not generate summary")
            st.stop()
        
        progress_bar.empty()
        status_text.empty()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📊 Analytics", "🏷️ Entities", "💾 Export"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("## 📋 AI-Generated Summary")
                st.markdown("""
                <div class="info-box">
                    <strong>Model:</strong> BART (facebook/bart-large-cnn)<br>
                    <strong>Type:</strong> Abstractive (AI writes new text, not extraction)<br>
                    <strong>Output:</strong> Fluent, concise summary in natural language
                </div>
                """, unsafe_allow_html=True)
                
                for i, sentence in enumerate(summary, 1):
                    st.markdown(f'<div class="summary-box"><strong>{i}.</strong> {sentence}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("## 🔑 Distinctive Keywords")
                st.markdown("""
                <div class="info-box">
                    <strong>Method:</strong> TF-IDF Multi-gram Extraction<br>
                    <strong>Purpose:</strong> Identify terms most characteristic of this document
                </div>
                """, unsafe_allow_html=True)
                
                if len(keywords) > 0:
                    fig_kw = plot_keywords_professional(keywords)
                    if fig_kw:
                        st.pyplot(fig_kw)
                        plt.close()
                    
                    with st.expander("📊 All keyword scores"):
                        for i, (term, score) in enumerate(keywords, 1):
                            importance = "🔥" if score > 0.5 else "⭐" if score > 0.3 else "📌"
                            st.markdown(f"**{i}. {term}** • `{score:.4f}` {importance}")
                else:
                    st.info("No keywords extracted")
            
            with col2:
                st.markdown("## 📊 Quick Stats")
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{num_chars:,}</p>
                    <p class="metric-label">Characters</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{len(words):,}</p>
                    <p class="metric-label">Words</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{num_sentences}</p>
                    <p class="metric-label">Sentences</p>
                </div>
                """, unsafe_allow_html=True)
                
                if readability:
                    st.markdown("---")
                    st.markdown("### 📖 Readability")
                    avg_len = readability['avg_sentence_length']
                    complexity = "✅ Simple" if avg_len < 15 else "📖 Medium" if avg_len < 25 else "📚 Complex"
                    st.metric("Avg Sentence", f"{avg_len:.1f} words", complexity)
                    st.metric("Vocabulary", f"{readability['lexical_diversity']:.1%} diversity")
        
        with tab2:
            st.markdown("## 📊 Analytics Dashboard")
            
            if sentiment_scores:
                st.markdown("### 💭 Sentiment Analysis")
                st.markdown("""
                <div class="info-box">
                    <strong>Model:</strong> VADER (Lexicon-based)<br>
                    <strong>Compound:</strong> -1 (negative) to +1 (positive)
                </div>
                """, unsafe_allow_html=True)
                
                fig_sent = plot_sentiment_advanced(sentiment_scores)
                if fig_sent:
                    st.pyplot(fig_sent)
                    plt.close()
                
                compound = sentiment_scores['compound']
                
                col1, col2 = st.columns(2)
                with col1:
                    if compound >= 0.05:
                        st.success("**POSITIVE Tone** 😊")
                    elif compound <= -0.05:
                        st.error("**NEGATIVE Tone** 😟")
                    else:
                        st.info("**NEUTRAL Tone** 😐")
                
                with col2:
                    st.metric("Compound Score", f"{compound:.3f}")
            
            st.markdown("---")
            
            if readability:
                st.markdown("### 📖 Readability Analysis")
                fig_read = plot_readability_gauge(readability)
                if fig_read:
                    st.pyplot(fig_read)
                    plt.close()
            
            if word_freq and show_word_freq:
                st.markdown("---")
                st.markdown("### 📈 Word Frequency")
                fig_freq = plot_word_frequency_advanced(word_freq)
                if fig_freq:
                    st.pyplot(fig_freq)
                    plt.close()
        
        with tab3:
            st.markdown("## 🏷️ Extracted Entities")
            
            if entities and len(entities) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'monetary_values' in entities and len(entities['monetary_values']) > 0:
                        st.markdown("### 💰 Money")
                        for val in sorted(set(entities['monetary_values']))[:8]:
                            st.markdown(f"• `{val}`")
                    
                    if 'dates' in entities and len(entities['dates']) > 0:
                        st.markdown("### 📅 Dates")
                        for date in entities['dates'][:6]:
                            st.markdown(f"• `{date}`")
                
                with col2:
                    if 'percentages' in entities and len(entities['percentages']) > 0:
                        st.markdown("### 📊 Percentages")
                        for perc in sorted(set(entities['percentages']), reverse=True)[:8]:
                            st.markdown(f"• `{perc}`")
                    
                    if 'proper_names' in entities and len(entities['proper_names']) > 0:
                        st.markdown("### 👤 Names")
                        for name in entities['proper_names'][:6]:
                            st.markdown(f"• `{name}`")
            else:
                st.info("No entities found")
        
        with tab4:
            st.markdown("## 💾 Export Results")
            
            export_data = {
                "metadata": {
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().isoformat(),
                    "chars": num_chars,
                    "words": len(words),
                    "sentences": num_sentences
                },
                "summary": {
                    "method": "BART abstractive",
                    "points": summary
                },
                "keywords": [{"term": t, "score": float(s)} for t, s in keywords],
                "sentiment": sentiment_scores if sentiment_scores else {},
                "entities": entities if entities else {}
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Debug"):
            st.code(str(e))

else:
    st.info("👆 Upload a document to start")
    st.markdown("""
    ### 🎯 What's New
    - ✨ **AI-Generated Summaries** (BART model)
    - 🔥 Real abstractive text generation
    - 🚀 Natural, fluent summaries
    
    ### 📚 How it works
    1. Upload PDF/TXT/HTML
    2. BART AI generates summary
    3. TF-IDF extracts keywords
    4. VADER analyzes sentiment
    5. Export results
    """)
