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

# CSS Clean
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
    .summary-container {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 8px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(60,64,67,0.1);
    }
    .summary-text {
        color: #202124;
        font-size: 1.05rem;
        line-height: 1.8;
        text-align: justify;
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
    .sentiment-explanation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        line-height: 1.7;
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
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    except Exception as e:
        st.error(f"Failed to load BART model: {e}")
        return None

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# ========== ANALYSES ==========
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def generate_comprehensive_summary(text):
    """Génère résumé propre et lisible"""
    try:
        summarizer = load_summarizer()
        
        if summarizer is None:
            # Fallback extractif
            sentences = split_sentences(text)
            summary_sentences = sentences[:8]
            return " ".join(summary_sentences)
        
        # Utiliser chunk de texte optimal
        input_text = text[:5000]
        
        # Générer avec paramètres robustes
        result = summarizer(
            input_text,
            max_length=250,
            min_length=120,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            truncation=True,
            length_penalty=1.0
        )
        
        if result and len(result) > 0 and 'summary_text' in result[0]:
            summary_text = result[0]['summary_text']
            
            # Nettoyer et formater
            summary_text = summary_text.strip()
            
            # Remplacer les points par des sauts de ligne pour meilleure lisibilité
            summary_text = re.sub(r'\.\s+', '.\n\n', summary_text)
            
            return summary_text
        else:
            # Fallback
            sentences = split_sentences(text)
            return " ".join(sentences[:8])
    
    except Exception as e:
        st.warning(f"Summary generation issue: {str(e)}. Using extractive fallback.")
        sentences = split_sentences(text)
        return " ".join(sentences[:8])

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

def generate_sentiment_explanation(scores):
    """Génère explication textuelle du sentiment"""
    compound = scores['compound']
    pos = scores['positive']
    neu = scores['neutral']
    neg = scores['negative']
    
    # Déterminer le ton général
    if compound >= 0.05:
        overall = "positive"
        overall_desc = "favorable, optimistic, or agreeable"
    elif compound <= -0.05:
        overall = "negative"
        overall_desc = "critical, unfavorable, or disagreeable"
    else:
        overall = "neutral"
        overall_desc = "objective, balanced, or factual"
    
    # Analyser la distribution
    dominant = max(['positive', 'neutral', 'negative'], key=lambda k: scores[k])
    
    explanation = f"""
**Overall Assessment:** This document has a **{overall}** tone (compound score: {compound:.3f}). 
The language used is predominantly {overall_desc}.

**Sentiment Breakdown:**

• **Positive content ({pos:.1%})**: This represents the proportion of language expressing favorable opinions, 
praise, satisfaction, or optimistic perspectives. {"This is the dominant emotion in the document." if dominant == 'positive' else ""}

• **Neutral content ({neu:.1%})**: This represents factual statements, objective descriptions, 
and balanced language without strong emotional coloring. {"This is the dominant style in the document." if dominant == 'neutral' else ""}

• **Negative content ({neg:.1%})**: This represents the proportion of language expressing criticism, 
concerns, dissatisfaction, or problematic issues. {"This is the dominant emotion in the document." if dominant == 'negative' else ""}

**Interpretation:** 
"""
    
    if compound >= 0.5:
        explanation += "The document is **very positive**, expressing strong favorable sentiment. This suggests enthusiasm, approval, or highly positive outcomes."
    elif compound >= 0.05:
        explanation += "The document is **moderately positive**, with more favorable than unfavorable content. This suggests a generally constructive or optimistic perspective."
    elif compound <= -0.5:
        explanation += "The document is **very negative**, expressing strong critical or unfavorable sentiment. This suggests serious concerns, criticisms, or negative outcomes."
    elif compound <= -0.05:
        explanation += "The document is **moderately negative**, with more unfavorable than favorable content. This suggests criticism, concerns, or problematic aspects being discussed."
    else:
        explanation += "The document is **neutral**, maintaining an objective and balanced tone. This is typical of factual reports, technical documents, or balanced analyses."
    
    return explanation

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

# ========== VISUALISATIONS ==========
def plot_sentiment_professional(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#34a853', '#5f6368', '#ea4335']
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none', alpha=0.9)
    
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
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    bars = ax.barh(range(len(terms)), scores, color='#1967d2', height=0.65, edgecolor='none')
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.008, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=10, fontweight='600', color='#5f6368')
    
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=11, color='#202124')
    ax.set_xlabel('TF-IDF Score', fontsize=12, fontweight='600', color='#5f6368')
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
st.set_page_config(page_title="InsightLens AI Pro", page_icon="🔍", layout="wide")

st.markdown('<p class="main-header">InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional Document Analysis System</p>', unsafe_allow_html=True)
st.markdown("---")

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
            st.error("❌ Document too short (minimum 100 words)")
            st.stop()
        
        num_sentences = len(split_sentences(text))
        
        st.success(f"✅ **{uploaded_file.name}** • {len(words):,} words • {num_sentences} sentences")
        
        status.text("Generating AI summary...")
        progress.progress(50)
        summary = generate_comprehensive_summary(text)
        
        status.text("Analyzing sentiment...")
        progress.progress(70)
        sentiment_scores = analyze_sentiment(text)
        sentiment_explanation = generate_sentiment_explanation(sentiment_scores)
        
        status.text("Extracting keywords...")
        progress.progress(85)
        keywords = extract_keywords(text, top_n=num_keywords)
        
        status.text("Computing statistics...")
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
                <strong>AI Model:</strong> BART (Facebook Research)<br>
                <strong>Type:</strong> Abstractive summarization (generates new text)<br>
                <strong>Quality:</strong> Natural, fluent, comprehensive overview
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="summary-container">
                <div class="summary-text">
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### About This Summary")
            st.markdown("""
            This summary was generated using **BART** (Bidirectional and Auto-Regressive Transformers), 
            a state-of-the-art AI model developed by Facebook Research.
            
            **Key Features:**
            - **Abstractive approach**: Creates new sentences rather than copying from the original
            - **Contextual understanding**: Comprehends relationships between different parts of the document
            - **Information synthesis**: Combines multiple ideas into coherent statements
            - **Natural language**: Produces fluent, readable text
            
            The model was trained on millions of documents and can identify core themes, 
            synthesize complex information, and produce human-quality summaries.
            """)
        
        # ========== TAB 2: SENTIMENT ==========
        with tab2:
            st.markdown("## General Tone Analysis")
            
            st.markdown("""
            <div class="info-card">
                <strong>Model:</strong> VADER (Valence Aware Dictionary)<br>
                <strong>Method:</strong> Lexicon-based sentiment analysis<br>
                <strong>Scale:</strong> Compound score from -1 (negative) to +1 (positive)
            </div>
            """, unsafe_allow_html=True)
            
            # Graphique
            fig_sent = plot_sentiment_professional(sentiment_scores)
            if fig_sent:
                st.pyplot(fig_sent)
                plt.close(fig_sent)
            
            st.markdown("---")
            
            # Explication détaillée
            st.markdown("### Detailed Analysis")
            
            st.markdown(f"""
            <div class="sentiment-explanation">
                {sentiment_explanation}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### Understanding VADER Sentiment Scores")
            st.markdown("""
            VADER is specifically designed for analyzing sentiment in business, news, and social media text.
            
            **How it works:**
            - Analyzes each word's emotional valence (positive/negative intensity)
            - Accounts for punctuation (!!!) and capitalization (VERY)
            - Recognizes intensifiers (extremely, barely) and negations (not, never)
            - Handles contextual shifts (but, however, although)
            
            **Score interpretation:**
            - **Positive score**: Percentage of positive language
            - **Neutral score**: Percentage of factual/objective language
            - **Negative score**: Percentage of negative language
            - **Compound score**: Overall normalized score (-1 to +1)
            """)
        
        # ========== TAB 3: KEYWORDS ==========
        with tab3:
            st.markdown("## Distinctive Keywords & Phrases")
            
            st.markdown("""
            <div class="info-card">
                <strong>Algorithm:</strong> TF-IDF (Term Frequency-Inverse Document Frequency)<br>
                <strong>N-grams:</strong> 1-3 word combinations<br>
                <strong>Purpose:</strong> Identify most characteristic terms
            </div>
            """, unsafe_allow_html=True)
            
            if len(keywords) > 0:
                fig_kw = plot_keywords_professional(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close(fig_kw)
                
                st.markdown("---")
                
                st.markdown("### Keyword Details")
                
                for i, (term, score) in enumerate(keywords, 1):
                    if score > 0.5:
                        badge = "🔥 Critical"
                        desc = "Core concept central to document"
                    elif score > 0.3:
                        badge = "⭐ High"
                        desc = "Major theme or key topic"
                    elif score > 0.15:
                        badge = "📌 Moderate"
                        desc = "Supporting concept"
                    else:
                        badge = "📎 Minor"
                        desc = "Contextual term"
                    
                    with st.expander(f"**{i}. {term}** • `{score:.4f}` • {badge}"):
                        st.markdown(f"**Importance:** {desc}")
                        st.markdown(f"**Score:** {score:.4f}")
            else:
                st.info("No keywords extracted")
        
        # ========== TAB 4: STATISTICS ==========
        with tab4:
            st.markdown("## Document Statistics")
            
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
            
            st.markdown("### Readability Metrics")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            avg_len = readability['avg_sentence_length']
            
            with col_r1:
                st.markdown("#### Sentence Complexity")
                st.metric("Avg Sentence Length", f"{avg_len:.1f} words")
                
                if avg_len < 15:
                    st.caption("✅ Simple (Easy)")
                elif avg_len < 25:
                    st.caption("📖 Medium")
                else:
                    st.caption("📚 Complex")
            
            with col_r2:
                st.markdown("#### Vocabulary Richness")
                st.metric("Lexical Diversity", f"{readability['lexical_diversity']:.1%}")
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
    st.info("👆 Upload a document to start")
    st.markdown("""
    ### Features
    - AI-generated comprehensive summary
    - Sentiment analysis with detailed explanations
    - Keyword extraction with importance scores
    - Readability and document statistics
    """)
