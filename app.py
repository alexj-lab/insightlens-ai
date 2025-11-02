import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
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

# Style matplotlib professionnel
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

# CSS Professionnel minimaliste
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #202124;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .summary-block {
        background: #f8f9fa;
        border-left: 3px solid #1967d2;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 4px;
        line-height: 1.8;
        font-size: 1.05rem;
        color: #202124;
    }
    .stat-box {
        background: #ffffff;
        border: 1px solid #dadce0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1967d2;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #5f6368;
        margin-top: 0.3rem;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 3px solid #34a853;
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
        raise ValueError(f"PDF error: {str(e)}")

def read_txt(path):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
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

# ========== CACHE MODÈLES ==========
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# ========== ANALYSES ==========
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def generate_comprehensive_summary(text):
    """Génère un résumé complet de 10-20 lignes"""
    try:
        summarizer = load_summarizer()
        
        # Utiliser plus de texte pour un résumé plus complet
        input_text = text[:6000]  # ~1500 mots
        
        # Paramètres pour résumé long et détaillé
        result = summarizer(
            input_text,
            max_length=300,  # Plus long pour 10-20 lignes
            min_length=150,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            truncation=True
        )
        
        summary_text = result[0]['summary_text']
        
        # Formater en paragraphe propre
        return summary_text.strip()
    
    except Exception as e:
        st.error(f"Summary error: {str(e)}")
        # Fallback extractif
        sentences = split_sentences(text)
        return " ".join(sentences[:10])

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
        
        avg_scores = {
            'positive': np.mean([s['pos'] for s in sentiments]),
            'neutral': np.mean([s['neu'] for s in sentiments]),
            'negative': np.mean([s['neg'] for s in sentiments]),
            'compound': np.mean([s['compound'] for s in sentiments])
        }
        return avg_scores
    except:
        return {'positive': 0, 'neutral': 1, 'negative': 0, 'compound': 0}

def readability_metrics(text):
    sentences = split_sentences(text)
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return {}
    
    avg_sentence_length = len(words) / len(sentences)
    long_words = len([w for w in words if len(w) > 6])
    unique_words = len(set([w.lower() for w in words]))
    lexical_diversity = unique_words / len(words)
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'long_words_ratio': long_words / len(words),
        'lexical_diversity': lexical_diversity,
        'total_words': len(words),
        'unique_words': unique_words,
        'total_sentences': len(sentences)
    }

# ========== VISUALISATIONS PROFESSIONNELLES ==========
def plot_sentiment_professional(scores):
    """Graphique sentiment professionnel"""
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#34a853', '#5f6368', '#ea4335']
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none', alpha=0.9)
    
    # Valeurs sur les barres
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=13, fontweight='700', color='#202124')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=12, fontweight='600', color='#5f6368')
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='700', color='#202124', pad=20)
    ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dadce0')
    ax.spines['bottom'].set_color('#dadce0')
    ax.tick_params(colors='#5f6368')
    
    plt.tight_layout()
    return fig

def plot_keywords_professional(pairs):
    """Graphique keywords professionnel"""
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Couleur unique professionnelle
    bars = ax.barh(range(len(terms)), scores, color='#1967d2', height=0.65, edgecolor='none')
    
    # Valeurs à droite des barres
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.008, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=10, fontweight='600', color='#5f6368')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11, color='#202124')
    ax.set_xlabel('TF-IDF Score (Importance)', fontsize=12, fontweight='600', color='#5f6368')
    ax.set_title('Top Keywords & Phrases', fontsize=14, fontweight='700', color='#202124', pad=20)
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dadce0')
    ax.spines['bottom'].set_color('#dadce0')
    ax.tick_params(colors='#5f6368')
    
    plt.tight_layout()
    return fig

# ========== INTERFACE ==========
st.set_page_config(
    page_title="InsightLens AI Pro",
    page_icon="🔍",
    layout="wide"
)

st.markdown('<p class="main-header">InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar minimaliste
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    num_keywords = st.slider("Number of keywords", 8, 20, 12)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Models")
    st.caption("• BART (Summarization)")
    st.caption("• VADER (Sentiment)")
    st.caption("• TF-IDF (Keywords)")

uploaded_file = st.file_uploader("📂 Upload Document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("Reading document...")
        progress.progress(25)
        text = read_any(temp_path)
        
        words = text.split()
        if len(words) < 100:
            st.error("❌ Document too short (minimum 100 words required)")
            st.stop()
        
        num_sentences = len(split_sentences(text))
        
        st.success(f"✅ Document loaded: **{uploaded_file.name}** • {len(words):,} words • {num_sentences} sentences")
        
        status.text("Generating AI summary...")
        progress.progress(50)
        summary = generate_comprehensive_summary(text)
        
        status.text("Analyzing sentiment...")
        progress.progress(70)
        sentiment_scores = analyze_sentiment(text)
        
        status.text("Extracting keywords...")
        progress.progress(85)
        keywords = extract_keywords(text, top_n=num_keywords)
        
        status.text("Computing readability metrics...")
        progress.progress(95)
        readability = readability_metrics(text)
        
        progress.progress(100)
        status.empty()
        progress.empty()
        
        # ========== 4 CATÉGORIES ==========
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💭 Sentiment", "🔑 Keywords", "📊 Statistics"])
        
        # ========== TAB 1: SUMMARY ==========
        with tab1:
            st.markdown("## Executive Summary")
            
            st.markdown("""
            <div class="info-card">
                <strong>Model:</strong> BART (facebook/bart-large-cnn)<br>
                <strong>Method:</strong> Abstractive summarization (AI-generated text)<br>
                <strong>Length:</strong> Comprehensive overview (10-20 lines)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="summary-block">
                {summary}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### How it works")
            st.markdown("""
            This summary is generated by BART, a state-of-the-art AI model from Facebook Research. 
            Unlike extractive methods that simply copy sentences from the original document, 
            BART **generates new text** that captures the essential meaning in a natural, fluent way.
            
            The model has been trained on millions of documents and can:
            - Identify key themes and arguments
            - Synthesize information from multiple paragraphs
            - Produce coherent, readable summaries
            - Maintain factual accuracy while being concise
            """)
        
        # ========== TAB 2: SENTIMENT ==========
        with tab2:
            st.markdown("## General Tone Analysis")
            
            st.markdown("""
            <div class="info-card">
                <strong>Model:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)<br>
                <strong>Method:</strong> Lexicon-based sentiment analysis<br>
                <strong>Compound Score:</strong> Ranges from -1 (most negative) to +1 (most positive)
            </div>
            """, unsafe_allow_html=True)
            
            # Graphique
            fig_sent = plot_sentiment_professional(sentiment_scores)
            if fig_sent:
                st.pyplot(fig_sent)
                plt.close(fig_sent)
            
            st.markdown("---")
            
            # Interprétation
            col1, col2, col3 = st.columns(3)
            
            compound = sentiment_scores['compound']
            
            with col1:
                st.markdown("### Overall Tone")
                if compound >= 0.05:
                    st.success("**POSITIVE** ✅")
                    tone_desc = "The document expresses favorable, optimistic, or agreeable sentiment."
                elif compound <= -0.05:
                    st.error("**NEGATIVE** ❌")
                    tone_desc = "The document expresses unfavorable, critical, or disagreeable sentiment."
                else:
                    st.info("**NEUTRAL** ➖")
                    tone_desc = "The document maintains an objective, balanced, or factual tone."
                
                st.caption(tone_desc)
            
            with col2:
                st.markdown("### Compound Score")
                st.metric("", f"{compound:.3f}", delta=None)
                st.caption("Normalized weighted composite score")
            
            with col3:
                st.markdown("### Dominant Emotion")
                dominant = max(['positive', 'neutral', 'negative'], key=lambda k: sentiment_scores[k])
                st.metric("", dominant.title(), f"{sentiment_scores[dominant]:.1%}")
                st.caption("Highest probability emotion")
            
            st.markdown("---")
            
            st.markdown("### Understanding Sentiment Scores")
            st.markdown("""
            **Positive Score:** Indicates favorable language, praise, optimism, or positive emotions.
            
            **Neutral Score:** Represents factual, objective, or balanced language without strong emotional content.
            
            **Negative Score:** Indicates criticism, concerns, problems, or negative emotions.
            
            **Compound Score Interpretation:**
            - **>= 0.05**: Overall positive document
            - **-0.05 to 0.05**: Neutral document
            - **<= -0.05**: Overall negative document
            
            VADER is particularly effective for business documents, social media, and news articles. 
            It accounts for punctuation, capitalization, degree modifiers (very, extremely), 
            and contextual valence shifters (but, however).
            """)
        
        # ========== TAB 3: KEYWORDS ==========
        with tab3:
            st.markdown("## Distinctive Keywords & Phrases")
            
            st.markdown("""
            <div class="info-card">
                <strong>Method:</strong> TF-IDF (Term Frequency-Inverse Document Frequency)<br>
                <strong>N-grams:</strong> 1-3 word combinations<br>
                <strong>Purpose:</strong> Identify terms most characteristic of this document
            </div>
            """, unsafe_allow_html=True)
            
            if len(keywords) > 0:
                # Graphique
                fig_kw = plot_keywords_professional(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close(fig_kw)
                
                st.markdown("---")
                
                # Table détaillée
                st.markdown("### Detailed Keyword Analysis")
                
                for i, (term, score) in enumerate(keywords, 1):
                    if score > 0.5:
                        importance = "🔥 Critical"
                        importance_desc = "Core concept central to document"
                    elif score > 0.3:
                        importance = "⭐ High"
                        importance_desc = "Major theme or key topic"
                    elif score > 0.15:
                        importance = "📌 Moderate"
                        importance_desc = "Supporting concept or recurring term"
                    else:
                        importance = "📎 Minor"
                        importance_desc = "Contextual or supplementary term"
                    
                    with st.expander(f"**{i}. {term}** • Score: `{score:.4f}` • {importance}"):
                        st.markdown(f"**Importance Level:** {importance_desc}")
                        st.markdown(f"**TF-IDF Score:** {score:.4f}")
                        st.markdown(f"**Interpretation:** This term appears frequently in this document but is relatively rare in typical documents, making it distinctive.")
                
                st.markdown("---")
                
                st.markdown("### How TF-IDF Works")
                st.markdown("""
                TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates 
                how important a word or phrase is to a document.
                
                **Components:**
                - **Term Frequency (TF):** How often the term appears in this document
                - **Inverse Document Frequency (IDF):** How rare the term is across all documents
                
                **Score Calculation:**
                - High score = Term appears frequently HERE but rarely in general documents
                - Low score = Common term that appears everywhere (like "the", "and", "is")
                
                **Why It Matters:**
                Terms with high TF-IDF scores are the most distinctive and informative for understanding 
                what makes this document unique. They represent the core topics, specialized vocabulary, 
                and key concepts specific to this text.
                """)
            else:
                st.info("No distinctive keywords extracted. Document may be too short or repetitive.")
        
        # ========== TAB 4: STATISTICS ==========
        with tab4:
            st.markdown("## Document Statistics & Readability")
            
            # Quick Stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{len(text):,}</div>
                    <div class="stat-label">Characters</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{readability['total_words']:,}</div>
                    <div class="stat-label">Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{readability['total_sentences']}</div>
                    <div class="stat-label">Sentences</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{readability['unique_words']:,}</div>
                    <div class="stat-label">Unique Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Readability Metrics
            st.markdown("### Readability Analysis")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            avg_len = readability['avg_sentence_length']
            
            with col_r1:
                st.markdown("#### Sentence Complexity")
                st.metric("Average Sentence Length", f"{avg_len:.1f} words")
                
                if avg_len < 15:
                    complexity = "✅ Simple (Easy to read)"
                    complexity_desc = "Short sentences, accessible language"
                elif avg_len < 25:
                    complexity = "📖 Medium (Moderate complexity)"
                    complexity_desc = "Standard business/academic writing"
                else:
                    complexity = "📚 Complex (Advanced reading)"
                    complexity_desc = "Long sentences, requires focus"
                
                st.caption(complexity)
                st.markdown(complexity_desc)
            
            with col_r2:
                st.markdown("#### Vocabulary Richness")
                st.metric("Lexical Diversity", f"{readability['lexical_diversity']:.1%}")
                
                diversity = readability['lexical_diversity']
                if diversity > 0.5:
                    diversity_desc = "High vocabulary variety"
                elif diversity > 0.3:
                    diversity_desc = "Moderate vocabulary variety"
                else:
                    diversity_desc = "Limited vocabulary variety"
                
                st.caption(diversity_desc)
                st.markdown(f"Unique words: {readability['unique_words']:,} out of {readability['total_words']:,} total")
            
            with col_r3:
                st.markdown("#### Reading Time")
                reading_time = readability['total_words'] / 200  # 200 words/min average
                st.metric("Estimated Time", f"{reading_time:.1f} minutes")
                st.caption("Based on 200 words/min")
                st.markdown(f"At 250 wpm (fast): {readability['total_words'] / 250:.1f} min")
            
            st.markdown("---")
            
            st.markdown("### Additional Metrics")
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.markdown("#### Word Length Distribution")
                long_words_ratio = readability['long_words_ratio']
                st.metric("Long Words (>6 letters)", f"{long_words_ratio:.1%}")
                st.caption("Higher ratio indicates more complex vocabulary")
            
            with col_a2:
                st.markdown("#### Document Type Assessment")
                if avg_len < 15 and long_words_ratio < 0.3:
                    doc_type = "Conversational / Simple"
                elif avg_len < 20 and long_words_ratio < 0.4:
                    doc_type = "Standard Business Writing"
                elif avg_len < 25 and long_words_ratio < 0.5:
                    doc_type = "Professional / Academic"
                else:
                    doc_type = "Technical / Legal"
                
                st.info(f"**Document Type:** {doc_type}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Debug Info"):
            st.code(str(e))

else:
    st.info("👆 Upload a document to start analysis")
    
    st.markdown("""
    ### Supported Formats
    - PDF (text-based, not scanned)
    - TXT (plain text)
    - HTML (web pages)
    
    ### Analysis Features
    1. **AI-Generated Summary** - Comprehensive 10-20 line overview
    2. **Sentiment Analysis** - Overall tone with detailed breakdown
    3. **Keyword Extraction** - Most distinctive terms and phrases
    4. **Readability Statistics** - Document complexity and metrics
    
    ### Best For
    - Business reports & contracts
    - Research papers & articles
    - Financial documents
    - Legal texts & case studies
    """)
