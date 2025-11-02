import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
MAX_SUMMARY_CHAR = 200

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

# ========== CSS PERSONNALISÉ ==========
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
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        st.warning(f"⚠️ Document truncated to {MAX_CHARS:,} characters for performance")
    
    return text

# ========== ANALYSES TEXTUELLES ==========
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def advanced_summarize(text, max_sent=5):
    """Résumé extractif avec TF-IDF + position weighting"""
    sentences = split_sentences(text)
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) <= max_sent:
        return [s[:MAX_SUMMARY_CHAR] + "..." if len(s) > MAX_SUMMARY_CHAR else s for s in sentences]
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 2),
            max_features=10000,
            min_df=1,
            max_df=0.85
        )
        
        X = vectorizer.fit_transform(sentences)
        tfidf_scores = np.asarray(X.mean(axis=1)).flatten()
        
        # Bonus pour position (début de document = plus important)
        position_scores = np.array([1.0 - (i / len(sentences)) * 0.3 for i in range(len(sentences))])
        
        combined_scores = tfidf_scores * position_scores
        
        top_indices = combined_scores.argsort()[-max_sent:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        return [s[:MAX_SUMMARY_CHAR] + "..." if len(s) > MAX_SUMMARY_CHAR else s for s in summary_sentences]
    
    except:
        return [s[:MAX_SUMMARY_CHAR] + "..." if len(s) > MAX_SUMMARY_CHAR else s for s in sentences[:max_sent]]

def extract_keywords(text, top_n=20):
    """Extraction mots-clés avec TF-IDF multi-gram"""
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
    """Analyse de sentiment VADER par chunks"""
    try:
        analyzer = SentimentIntensityAnalyzer()
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
    """Extraction d'entités : montants, %, dates, noms propres"""
    entities = {}
    
    # Montants monétaires
    money_pattern = r'(?:€|USD|\$|EUR)\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?|\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\s*(?:€|EUR|USD|\$|million|billion|M|B|bn|mn)'
    money_matches = re.findall(money_pattern, text, re.I)
    valid_money = [m for m in money_matches if re.search(r'[€$]|million|billion|EUR|USD|[MB](?:n)?$', m, re.I)]
    if valid_money:
        entities['monetary_values'] = list(set(valid_money))[:12]
    
    # Pourcentages
    percent_matches = re.findall(r'\b\d+(?:\.\d+)?%', text)
    if percent_matches:
        entities['percentages'] = list(set(percent_matches))[:12]
    
    # Dates complètes
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
    date_matches = re.findall(date_pattern, text, re.I)
    if date_matches:
        entities['dates'] = list(set(date_matches))[:10]
    
    # Noms propres (Majuscule suivie de minuscules, 2+ mots)
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    name_matches = re.findall(name_pattern, text)
    # Filtrer les faux positifs
    valid_names = [n for n in name_matches if n not in ['The Company', 'This Agreement'] and len(n.split()) <= 4]
    if valid_names:
        name_counter = Counter(valid_names)
        entities['proper_names'] = [name for name, count in name_counter.most_common(10)]
    
    return entities

def word_frequency_analysis(text, top_n=25):
    """Analyse fréquence des mots"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [w for w in words if w not in STOP_WORDS]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

def readability_metrics(text):
    """Calcul de métriques de lisibilité"""
    sentences = split_sentences(text)
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return {}
    
    avg_sentence_length = len(words) / len(sentences)
    
    # Nombre de mots longs (>6 lettres)
    long_words = len([w for w in words if len(w) > 6])
    
    # Complexité lexicale
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

# ========== VISUALISATIONS AVANCÉES ==========
def plot_keywords_professional(pairs):
    """Graphique keywords avec gradients"""
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(terms)))
    bars = ax.barh(range(len(terms)), scores, color=colors, height=0.75, edgecolor='black', linewidth=0.5)
    
    # Ajouter les valeurs
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11)
    ax.set_xlabel('TF-IDF Score (Importance)', fontsize=11, fontweight='bold')
    ax.set_title('🔑 Top Keywords & Phrases', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines('right').set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_sentiment_advanced(scores):
    """Graphique sentiment avec interprétation visuelle"""
    labels = ['Positive\n😊', 'Neutral\n😐', 'Negative\n😟']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
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
    
    # Pie chart
    ax2.pie(values, labels=['Positive', 'Neutral', 'Negative'], colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('📊 Sentiment Breakdown', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_word_frequency_advanced(word_freq):
    """Graphique fréquence avec nuage de mots stylisé"""
    if len(word_freq) == 0:
        return None
    
    words = [w for w, _ in word_freq[:20]]
    counts = [c for _, c in word_freq[:20]]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(words)))
    bars = ax.barh(range(len(words)), counts, color=colors, height=0.75, edgecolor='black', linewidth=0.5)
    
    # Ajouter les valeurs
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words[::-1], fontsize=10)
    ax.set_xlabel('Frequency (occurrences)', fontsize=11, fontweight='bold')
    ax.set_title('📈 Most Frequent Words', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_readability_gauge(metrics):
    """Jauge de lisibilité"""
    if not metrics:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    # Jauge 1: Avg sentence length
    ax = axes[0]
    avg_len = metrics['avg_sentence_length']
    complexity = min(avg_len / 30, 1.0)  # 30 mots = complexe
    
    ax.barh([0], [complexity], color='#3498db', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Simple', 'Medium', 'Complex'])
    ax.set_yticks([])
    ax.set_title(f'Sentence Length\n({avg_len:.1f} words/sentence)', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Jauge 2: Long words ratio
    ax = axes[1]
    long_ratio = metrics['long_words_ratio']
    
    ax.barh([0], [long_ratio], color='#e74c3c', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Few', 'Medium', 'Many'])
    ax.set_yticks([])
    ax.set_title(f'Long Words (>6 letters)\n({long_ratio:.1%})', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Jauge 3: Lexical diversity
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

# ========== INTERFACE STREAMLIT ==========
st.set_page_config(
    page_title="InsightLens AI Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header avec style
st.markdown('<p class="main-title">🔍 InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis & Intelligence System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar avec explications
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=InsightLens+AI", use_container_width=True)
    
    st.markdown("## ⚙️ Analysis Settings")
    
    with st.expander("📋 Summary Configuration", expanded=True):
        st.markdown("**How it works:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) to identify the most important sentences based on statistical word importance and document position.")
        num_summary = st.slider("Number of key points", 3, 10, 5, help="Extract N most important sentences")
    
    with st.expander("🔑 Keywords Configuration", expanded=True):
        st.markdown("**How it works:** Extracts distinctive words and phrases (1-3 word combinations) that best characterize the document using TF-IDF scoring.")
        num_keywords = st.slider("Number of keywords", 5, 25, 15, help="Top N distinctive terms")
    
    with st.expander("💭 Analysis Options", expanded=True):
        show_sentiment = st.checkbox("Sentiment Analysis", value=True, help="VADER sentiment analyzer for emotional tone")
        show_entities = st.checkbox("Entity Extraction", value=True, help="Regex-based detection of money, dates, names")
        show_readability = st.checkbox("Readability Metrics", value=True, help="Document complexity analysis")
        show_word_freq = st.checkbox("Word Frequency", value=False, help="Most common words visualization")
    
    st.markdown("---")
    st.markdown("### 📚 About the Technology")
    st.markdown("""
    **Machine Learning Models:**
    - TF-IDF Vectorization (scikit-learn)
    - VADER Sentiment Analysis
    - N-gram extraction (1-3 words)
    
    **Processing Pipeline:**
    1. Document ingestion & cleaning
    2. Sentence segmentation
    3. Feature extraction (TF-IDF)
    4. Statistical analysis
    5. Visualization & export
    """)
    
    st.caption("v2.0 Pro | Powered by scikit-learn & VADER")

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload Your Document",
    type=["pdf", "txt", "html", "htm"],
    help="Supported formats: Text-based PDF, TXT, HTML"
)

if uploaded_file:
    try:
        # Sauvegarde fichier
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Lecture avec progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📖 Reading document...")
        progress_bar.progress(20)
        text = read_any(temp_path)
        
        # Validation
        words = text.split()
        if len(words) < 50:
            st.error("❌ Document too short (< 50 words). Please upload a longer document for meaningful analysis.")
            st.stop()
        
        status_text.text("🔍 Analyzing text structure...")
        progress_bar.progress(40)
        
        # Métriques de base
        num_sentences = len(split_sentences(text))
        num_chars = len(text)
        
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Document loaded successfully!</strong><br>
            📄 <strong>File:</strong> {uploaded_file.name}<br>
            📊 <strong>Size:</strong> {num_chars:,} characters • {len(words):,} words • {num_sentences} sentences
        </div>
        """, unsafe_allow_html=True)
        
        # Analyse principale
        status_text.text("🤖 Running AI analysis...")
        progress_bar.progress(60)
        
        summary = advanced_summarize(text, max_sent=num_summary)
        keywords = extract_keywords(text, top_n=num_keywords)
        
        progress_bar.progress(70)
        
        sentiment_scores = None
        if show_sentiment:
            sentiment_scores = analyze_sentiment(text)
        
        progress_bar.progress(80)
        
        entities = None
        if show_entities:
            entities = extract_entities(text)
        
        progress_bar.progress(90)
        
        readability = None
        if show_readability:
            readability = readability_metrics(text)
        
        word_freq = None
        if show_word_freq:
            word_freq = word_frequency_analysis(text)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Validation résumé
        if len(summary) == 0:
            st.error("❌ Could not generate summary. Document may be too short or poorly formatted.")
            st.stop()
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # ========== TABS POUR ORGANISATION ==========
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary & Keywords", "📊 Analytics Dashboard", "🏷️ Entities & Details", "💾 Export & Reports"])
        
        # ========== TAB 1: SUMMARY & KEYWORDS ==========
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("## 📋 Executive Summary")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Method:</strong> TF-IDF Extractive Summarization<br>
                    <strong>Algorithm:</strong> Scores each sentence based on statistical word importance + position weighting (early sentences get bonus)<br>
                    <strong>Output:</strong> Top {len(summary)} most informative sentences (max {MAX_SUMMARY_CHAR} characters each)
                </div>
                """, unsafe_allow_html=True)
                
                for i, sentence in enumerate(summary, 1):
                    st.markdown(f'<div class="summary-box"><strong>{i}.</strong> {sentence}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("## 🔑 Distinctive Keywords & Phrases")
                st.markdown("""
                <div class="info-box">
                    <strong>Method:</strong> TF-IDF Multi-gram Extraction (1-3 word phrases)<br>
                    <strong>Algorithm:</strong> Identifies terms that are frequent in this document but rare across typical documents<br>
                    <strong>Score:</strong> Higher = more distinctive and important to this document
                </div>
                """, unsafe_allow_html=True)
                
                if len(keywords) > 0:
                    fig_kw = plot_keywords_professional(keywords)
                    if fig_kw:
                        st.pyplot(fig_kw)
                        plt.close()
                    
                    with st.expander("📊 View all keyword scores & explanations"):
                        st.markdown("**TF-IDF Score Interpretation:**")
                        st.markdown("- **0.1-0.3:** Minor term, appears occasionally")
                        st.markdown("- **0.3-0.5:** Moderate importance, recurring theme")
                        st.markdown("- **0.5-0.7:** High importance, key concept")
                        st.markdown("- **>0.7:** Critical term, central to document")
                        st.markdown("---")
                        
                        for i, (term, score) in enumerate(keywords, 1):
                            if score > 0.5:
                                importance = "🔥 Critical"
                            elif score > 0.3:
                                importance = "⭐ High"
                            elif score > 0.1:
                                importance = "📌 Moderate"
                            else:
                                importance = "📎 Minor"
                            
                            st.markdown(f"**{i}. {term}** • Score: `{score:.4f}` • {importance}")
                else:
                    st.info("ℹ️ No distinctive keywords extracted. Document may be too short or repetitive.")
            
            with col2:
                st.markdown("## 📊 Quick Stats")
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{num_chars:,}</p>
                    <p class="metric-label">Total Characters</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{len(words):,}</p>
                    <p class="metric-label">Total Words</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-value">{num_sentences}</p>
                    <p class="metric-label">Sentences</p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(keywords) > 0:
                    st.markdown(f"""
                    <div class="metric-box">
                        <p class="metric-value" style="font-size: 1.5rem;">{keywords[0][0]}</p>
                        <p class="metric-label">Top Keyword</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if readability:
                    st.markdown("---")
                    st.markdown("### 📖 Readability")
                    
                    avg_len = readability['avg_sentence_length']
                    if avg_len < 15:
                        complexity_text = "✅ Simple (easy to read)"
                    elif avg_len < 25:
                        complexity_text = "📖 Medium (moderate)"
                    else:
                        complexity_text = "📚 Complex (advanced)"
                    
                    st.metric("Avg Sentence Length", f"{avg_len:.1f} words", complexity_text)
                    st.metric("Unique Words", f"{readability['unique_words']:,}")
                    st.metric("Vocabulary Richness", f"{readability['lexical_diversity']:.1%}")
        
        # ========== TAB 2: ANALYTICS DASHBOARD ==========
        with tab2:
            st.markdown("## 📊 Comprehensive Analytics Dashboard")
            
            if sentiment_scores:
                st.markdown("### 💭 Sentiment Analysis")
                st.markdown("""
                <div class="info-box">
                    <strong>Model:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)<br>
                    <strong>Method:</strong> Lexicon-based sentiment analysis with rules for punctuation, capitalization, degree modifiers<br>
                    <strong>Compound Score:</strong> Normalized weighted composite score ranging from -1 (most negative) to +1 (most positive)
                </div>
                """, unsafe_allow_html=True)
                
                fig_sent = plot_sentiment_advanced(sentiment_scores)
                if fig_sent:
                    st.pyplot(fig_sent)
                    plt.close()
                
                compound = sentiment_scores['compound']
                
                col_sent1, col_sent2, col_sent3 = st.columns(3)
                
                with col_sent1:
                    if compound >= 0.05:
                        st.success("**Overall Tone: POSITIVE** 😊")
                        st.markdown("Document expresses favorable, optimistic, or agreeable sentiment.")
                    elif compound <= -0.05:
                        st.error("**Overall Tone: NEGATIVE** 😟")
                        st.markdown("Document expresses unfavorable, critical, or disagreeable sentiment.")
                    else:
                        st.info("**Overall Tone: NEUTRAL** 😐")
                        st.markdown("Document maintains objective, balanced, or factual tone.")
                
                with col_sent2:
                    st.metric("Compound Score", f"{compound:.3f}", 
                             "Positive" if compound > 0 else "Negative" if compound < 0 else "Neutral")
                
                with col_sent3:
                    dominant = max(sentiment_scores, key=lambda k: sentiment_scores[k] if k != 'compound' else 0)
                    st.metric("Dominant Emotion", dominant.capitalize(), f"{sentiment_scores[dominant]:.1%}")
            
            st.markdown("---")
            
            if readability:
                st.markdown("### 📖 Readability & Complexity Analysis")
                st.markdown("""
                <div class="info-box">
                    <strong>Metrics Explained:</strong><br>
                    • <strong>Sentence Length:</strong> Average words per sentence (shorter = easier)<br>
                    • <strong>Long Words:</strong> Percentage of words >6 letters (fewer = simpler)<br>
                    • <strong>Vocabulary Diversity:</strong> Unique words / total words (higher = richer vocabulary)
                </div>
                """, unsafe_allow_html=True)
                
                fig_read = plot_readability_gauge(readability)
                if fig_read:
                    st.pyplot(fig_read)
                    plt.close()
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                
                with col_r1:
                    st.metric("Avg Sentence", f"{readability['avg_sentence_length']:.1f} words")
                
                with col_r2:
                    st.metric("Long Words", f"{readability['long_words_ratio']:.1%}")
                
                with col_r3:
                    st.metric("Lexical Diversity", f"{readability['lexical_diversity']:.1%}")
                
                with col_r4:
                    reading_time = readability['total_words'] / 200  # 200 mots/min
                    st.metric("Est. Reading Time", f"{reading_time:.1f} min")
            
            st.markdown("---")
            
            if word_freq and show_word_freq:
                st.markdown("### 📈 Word Frequency Analysis")
                st.markdown("""
                <div class="info-box">
                    <strong>Method:</strong> Simple word counting after removing stop words<br>
                    <strong>Purpose:</strong> Identifies most recurring themes and topics in raw frequency (not adjusted for importance like TF-IDF)
                </div>
                """, unsafe_allow_html=True)
                
                fig_freq = plot_word_frequency_advanced(word_freq)
                if fig_freq:
                    st.pyplot(fig_freq)
                    plt.close()
        
        # ========== TAB 3: ENTITIES & DETAILS ==========
        with tab3:
            st.markdown("## 🏷️ Extracted Entities & Named Elements")
            
            if entities and len(entities) > 0:
                st.markdown("""
                <div class="info-box">
                    <strong>Extraction Method:</strong> Regex pattern matching + filtering<br>
                    <strong>Entity Types:</strong> Monetary values, Percentages, Dates, Proper names<br>
                    <strong>Use Case:</strong> Quick reference for key numbers, dates, and people mentioned in document
                </div>
                """, unsafe_allow_html=True)
                
                col_e1, col_e2 = st.columns(2)
                
                with col_e1:
                    if 'monetary_values' in entities and len(entities['monetary_values']) > 0:
                        st.markdown("### 💰 Monetary Values")
                        st.markdown("*Financial amounts and currency mentions*")
                        
                        for i, val in enumerate(sorted(set(entities['monetary_values']))[:10], 1):
                            st.markdown(f"**{i}.** `{val}`")
                    
                    if 'dates' in entities and len(entities['dates']) > 0:
                        st.markdown("### 📅 Important Dates")
                        st.markdown("*Timeline and temporal references*")
                        
                        for i, date in enumerate(entities['dates'][:8], 1):
                            st.markdown(f"**{i}.** `{date}`")
                
                with col_e2:
                    if 'percentages' in entities and len(entities['percentages']) > 0:
                        st.markdown("### 📊 Percentages & Ratios")
                        st.markdown("*Statistical figures and proportions*")
                        
                        # Trier par valeur numérique
                        percent_values = [(p, float(p.strip('%'))) for p in entities['percentages']]
                        percent_sorted = sorted(percent_values, key=lambda x: x[1], reverse=True)
                        
                        for i, (perc, _) in enumerate(percent_sorted[:10], 1):
                            st.markdown(f"**{i}.** `{perc}`")
                    
                    if 'proper_names' in entities and len(entities['proper_names']) > 0:
                        st.markdown("### 👤 Proper Names")
                        st.markdown("*People, organizations, locations*")
                        
                        for i, name in enumerate(entities['proper_names'][:8], 1):
                            st.markdown(f"**{i}.** `{name}`")
            else:
                st.info("ℹ️ No entities extracted. Document may not contain numerical data, dates, or proper names.")
            
            st.markdown("---")
            
            # Section debug et métadonnées
            with st.expander("🔍 Document Metadata & Debug Info"):
                st.json({
                    "filename": uploaded_file.name,
                    "file_size_bytes": uploaded_file.size,
                    "processed_chars": len(text),
                    "processed_words": len(words),
                    "sentences_detected": num_sentences,
                    "summary_points_extracted": len(summary),
                    "keywords_extracted": len(keywords),
                    "entities_found": {k: len(v) for k, v in entities.items()} if entities else {},
                    "sentiment_analysis": "completed" if sentiment_scores else "skipped",
                    "readability_analysis": "completed" if readability else "skipped"
                })
        
        # ========== TAB 4: EXPORT & REPORTS ==========
        with tab4:
            st.markdown("## 💾 Export Results & Generate Reports")
            
            st.markdown("""
            <div class="info-box">
                <strong>Available Formats:</strong><br>
                • <strong>JSON:</strong> Machine-readable format for integration with other tools, databases, or APIs<br>
                • <strong>Markdown:</strong> Human-readable report for documentation, sharing, or further editing<br>
                • <strong>Python Dict:</strong> Direct data structure for programmatic use
            </div>
            """, unsafe_allow_html=True)
            
            # Préparer les données d'export
            export_data = {
                "metadata": {
                    "filename": uploaded_file.name,
                    "file_size_bytes": uploaded_file.size,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "insightlens_version": "2.0-pro",
                    "statistics": {
                        "characters": num_chars,
                        "words": len(words),
                        "sentences": num_sentences,
                        "unique_words": readability['unique_words'] if readability else None
                    }
                },
                "summary": {
                    "method": "TF-IDF extractive summarization",
                    "points": summary
                },
                "keywords": {
                    "method": "TF-IDF multi-gram extraction",
                    "terms": [{"term": t, "score": float(s), "rank": i+1} for i, (t, s) in enumerate(keywords)]
                },
                "sentiment": {
                    "method": "VADER sentiment analysis",
                    "scores": sentiment_scores if sentiment_scores else {}
                } if sentiment_scores else None,
                "entities": entities if entities else {},
                "readability": readability if readability else None
            }
            
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                st.markdown("### 📥 JSON Export")
                st.markdown("*Structured data for integration*")
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json_str,
                    file_name=f"insightlens_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                st.caption(f"Size: {len(json_str):,} bytes")
            
            with col_export2:
                st.markdown("### 📄 Markdown Report")
                st.markdown("*Formatted document report*")
                
                # Générer rapport Markdown
                markdown_report = f"""# 🔍 InsightLens AI Analysis Report

**Document:** {uploaded_file.name}  
**Analyzed:** {datetime.now().strftime('%B %d, %Y at %H:%M')}  
**Version:** InsightLens AI Pro 2.0

---

## 📊 Document Statistics

- **Characters:** {num_chars:,}
- **Words:** {len(words):,}
- **Sentences:** {num_sentences}
- **Unique Words:** {readability['unique_words']:,} ({readability['lexical_diversity']:.1%} diversity)
- **Avg Sentence Length:** {readability['avg_sentence_length']:.1f} words
- **Estimated Reading Time:** {readability['total_words'] / 200:.1f} minutes

---

## 📋 Executive Summary

**Method:** TF-IDF Extractive Summarization  
**Key Points Extracted:** {len(summary)}

{chr(10).join([f"{i}. {s}" for i, s in enumerate(summary, 1)])}

---

## 🔑 Distinctive Keywords

**Method:** TF-IDF Multi-gram Analysis  
**Top {len(keywords)} Terms:**

{chr(10).join([f"- **{t}** (score: {s:.4f})" for t, s in keywords[:15]])}

---

## 💭 Sentiment Analysis

**Model:** VADER Sentiment Analyzer  
**Overall Tone:** {"Positive 😊" if sentiment_scores['compound'] >= 0.05 else "Negative 😟" if sentiment_scores['compound'] <= -0.05 else "Neutral 😐"}

- **Positive:** {sentiment_scores['positive']:.1%}
- **Neutral:** {sentiment_scores['neutral']:.1%}
- **Negative:** {sentiment_scores['negative']:.1%}
- **Compound Score:** {sentiment_scores['compound']:.3f}

---

## 🏷️ Extracted Entities

### 💰 Monetary Values
{chr(10).join([f"- {v}" for v in (entities.get('monetary_values', [])[:8])]) if entities and 'monetary_values' in entities else "*None detected*"}

### 📊 Percentages
{chr(10).join([f"- {v}" for v in (entities.get('percentages', [])[:8])]) if entities and 'percentages' in entities else "*None detected*"}

### 📅 Dates
{chr(10).join([f"- {v}" for v in (entities.get('dates', [])[:6])]) if entities and 'dates' in entities else "*None detected*"}

---

## 📖 Readability Metrics

- **Average Sentence Length:** {readability['avg_sentence_length']:.1f} words
- **Long Words Ratio:** {readability['long_words_ratio']:.1%}
- **Lexical Diversity:** {readability['lexical_diversity']:.1%}
- **Complexity Assessment:** {"Simple (easy to read)" if readability['avg_sentence_length'] < 15 else "Medium (moderate complexity)" if readability['avg_sentence_length'] < 25 else "Complex (advanced reading level)"}

---

*Report generated by InsightLens AI Pro 2.0*  
*Technology: TF-IDF, VADER, Scikit-learn, Python 3.10+*
"""
                
                st.download_button(
                    label="💾 Download Markdown",
                    data=markdown_report,
                    file_name=f"insightlens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
                st.caption(f"Size: {len(markdown_report):,} bytes")
            
            with col_export3:
                st.markdown("### 🐍 Python Dict")
                st.markdown("*Copy-paste ready data*")
                
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.code(json_str, language="python")
                    st.success("✅ Code displayed! Use your browser's copy function")
                
                st.caption("Python dictionary format")
            
            st.markdown("---")
            
            # Preview du rapport
            with st.expander("👁️ Preview Markdown Report"):
                st.markdown(markdown_report)
    
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.markdown("""
        <div class="info-box">
            <strong>💡 Troubleshooting Tips:</strong><br>
            • Ensure PDF is text-based (not scanned image)<br>
            • Try converting to TXT format<br>
            • Check file is not corrupted or password-protected<br>
            • Verify file size is under 10 MB
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Technical Error Details"):
            st.code(str(e))

else:
    # Welcome screen élégante
    st.markdown("## 👋 Welcome to InsightLens AI Pro")
    
    st.markdown("""
    <div class="info-box" style="font-size: 1.1rem;">
        <strong>Transform lengthy documents into actionable insights in seconds.</strong><br>
        Upload any PDF, TXT, or HTML file to begin your intelligent analysis.
    </div>
    """, unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 🎯 Core Capabilities
        
        #### 📋 Intelligent Summarization
        Automatically extracts the most important sentences using TF-IDF statistical analysis combined with position weighting. Perfect for quickly understanding long reports.
        
        #### 🔑 Keyword Extraction
        Identifies distinctive terms and phrases (1-3 word combinations) that best characterize your document. Uses advanced multi-gram TF-IDF scoring.
        
        #### 💭 Sentiment Analysis
        Analyzes emotional tone using VADER (Valence Aware Dictionary and sEntiment Reasoner). Detects positive, neutral, and negative sentiment with high accuracy.
        
        #### 🏷️ Entity Recognition
        Automatically detects and extracts monetary values, percentages, dates, and proper names using sophisticated regex patterns.
        
        #### 📖 Readability Metrics
        Evaluates document complexity through sentence length analysis, vocabulary diversity, and lexical richness measurements.
        """)
    
    with col_info2:
        st.markdown("""
        ### ✅ Ideal Use Cases
        
        - **Business Reports & Contracts**  
          Quickly understand key terms, obligations, and financial figures
        
        - **Research Papers & Articles**  
          Extract main findings, methodology, and conclusions
        
        - **Financial Documents**  
          Identify monetary values, percentages, trends, and sentiment
        
        - **Legal Texts & Case Studies**  
          Find key dates, parties, and critical clauses
        
        - **News & Media Analysis**  
          Gauge sentiment, extract key facts, identify main topics
        
        ### 🚀 How to Get Started
        
        1. **Upload** your document (PDF, TXT, or HTML)
        2. **Configure** analysis settings in the sidebar
        3. **Wait** 10-30 seconds for AI processing
        4. **Explore** results across 4 organized tabs
        5. **Export** insights in JSON or Markdown format
        
        ### 💡 Pro Tips for Best Results
        
        - **Document Length:** 500-10,000 words optimal
        - **Format:** Well-structured text with clear paragraphs
        - **Language:** Optimized for English (works with French)
        - **Quality:** Text-based PDFs only (no scanned images)
        """)
    
    st.markdown("---")
    
    st.markdown("### 🔬 Technology Stack")
    
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("""
        **Machine Learning**
        - TF-IDF Vectorization
        - VADER Sentiment
        - N-gram Analysis
        - Feature Extraction
        """)
    
    with col_tech2:
        st.markdown("""
        **Data Processing**
        - PyMuPDF (PDF parsing)
        - BeautifulSoup (HTML)
        - Regex (Entity extraction)
        - NumPy (Numerical ops)
        """)
    
    with col_tech3:
        st.markdown("""
        **Visualization**
        - Matplotlib charts
        - Seaborn styling
        - Custom gradients
        - Interactive plots
        """)
