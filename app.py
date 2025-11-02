import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
from datetime import datetime

# Configuration
MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .summary-point {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== INGESTION ==========
def read_pdf(path):
    """Extrait le texte d'un PDF avec gestion d'erreurs robuste"""
    try:
        text = []
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text.append(page_text)
        return "\n\n".join(text)
    except Exception as e:
        raise ValueError(f"PDF reading error: {str(e)}")

def read_txt(path):
    """Lit un fichier texte avec plusieurs encodages"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file")

def read_html(path):
    """Extrait le texte d'un HTML"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            # Supprimer scripts et styles
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        raise ValueError(f"HTML reading error: {str(e)}")

def clean_text(text):
    """Nettoyage avancé du texte"""
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    # Normaliser les sauts de ligne
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def read_any(path):
    """Lit n'importe quel format avec gestion robuste"""
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

# ========== ANALYSE TEXTUELLE ==========
STOP_WORDS = set([
    "the", "and", "of", "to", "in", "for", "on", "with", "a", "an", "by", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "but", "or", "as", "at", "from", "into", "through",
    # Français
    "les", "des", "et", "de", "la", "le", "pour", "dans", "sur", "avec", "par",
    "aux", "au", "une", "un", "du", "est", "sont", "ont", "a", "ce", "cette", "ces",
    "qui", "que", "dont", "où", "nous", "vous", "ils", "elles", "son", "sa", "ses",
    "leur", "leurs", "tout", "tous", "toute", "toutes", "aussi", "très", "plus"
])

def split_sentences(text):
    """Découpage intelligent en phrases"""
    # Pattern pour détecter les fins de phrases
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filtrer les phrases trop courtes
    sentences = [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]
    return sentences

def advanced_summarize(text, max_sent=5):
    """Résumé extractif avancé avec TF-IDF + position"""
    sentences = split_sentences(text)
    
    if len(sentences) <= max_sent:
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
        
        # Score TF-IDF moyen par phrase
        tfidf_scores = np.asarray(X.mean(axis=1)).flatten()
        
        # Bonus pour les phrases en début de document (souvent plus importantes)
        position_scores = np.array([1.0 - (i / len(sentences)) * 0.3 for i in range(len(sentences))])
        
        # Score combiné
        combined_scores = tfidf_scores * position_scores
        
        # Sélectionner les top phrases
        top_indices = combined_scores.argsort()[-max_sent:][::-1]
        
        # Trier par ordre d'apparition original
        top_indices_sorted = sorted(top_indices)
        
        return [sentences[i] for i in top_indices_sorted]
    
    except Exception as e:
        st.warning(f"Advanced summarization failed: {e}. Using simple method.")
        return sentences[:max_sent]

def extract_keywords(text, top_n=15):
    """Extraction de mots-clés avec TF-IDF sur paragraphes"""
    # Découper en paragraphes
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if len(p.strip()) > 50]
    
    if len(paragraphs) < 2:
        # Fallback : découper en chunks de 500 caractères
        chunk_size = 500
        paragraphs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    if len(paragraphs) < 2:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 3),  # Uni, bi et tri-grammes
            max_features=5000,
            min_df=1,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(paragraphs)
        
        # Score maximum de chaque terme
        scores_matrix = X.max(axis=0)
        scores = np.array(scores_matrix).flatten()
        
        terms = vectorizer.get_feature_names_out()
        
        # Filtrer les termes trop courts
        filtered_pairs = [(term, score) for term, score in zip(terms, scores) if len(term) > 2]
        
        # Trier et retourner top N
        sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
        
        return sorted_pairs[:top_n]
    
    except Exception as e:
        st.warning(f"Keyword extraction error: {e}")
        return []

def analyze_sentiment(text):
    """Analyse de sentiment avec VADER"""
    try:
        analyzer = SentimentIntensityAnalyzer()
        
        # Analyser par chunks pour plus de précision
        chunk_size = 5000
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 20000), chunk_size)]
        
        sentiments = []
        for chunk in chunks:
            scores = analyzer.polarity_scores(chunk)
            sentiments.append(scores)
        
        # Moyenne des scores
        avg_scores = {
            'positive': np.mean([s['pos'] for s in sentiments]),
            'neutral': np.mean([s['neu'] for s in sentiments]),
            'negative': np.mean([s['neg'] for s in sentiments]),
            'compound': np.mean([s['compound'] for s in sentiments])
        }
        
        return avg_scores
    
    except Exception as e:
        st.warning(f"Sentiment analysis error: {e}")
        return {'positive': 0, 'neutral': 1, 'negative': 0, 'compound': 0}

def extract_entities(text):
    """Extraction d'entités simples (nombres, montants, pourcentages)"""
    entities = {}
    
    # Montants monétaires
    money_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|bn|mn))?\b'
    money_matches = re.findall(money_pattern, text, re.IGNORECASE)
    if money_matches:
        entities['monetary_values'] = money_matches[:10]
    
    # Pourcentages
    percent_pattern = r'\d+(?:\.\d+)?%'
    percent_matches = re.findall(percent_pattern, text)
    if percent_matches:
        entities['percentages'] = percent_matches[:10]
    
    # Dates (années)
    year_pattern = r'\b(19|20)\d{2}\b'
    year_matches = re.findall(year_pattern, text)
    if year_matches:
        entities['years'] = list(set(year_matches))[:10]
    
    return entities

def word_frequency_analysis(text, top_n=20):
    """Analyse de fréquence des mots"""
    # Nettoyer et tokeniser
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filtrer les stop words
    filtered_words = [w for w in words if w not in STOP_WORDS]
    
    # Compter les fréquences
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_n)

# ========== VISUALISATIONS ==========
def plot_keywords(pairs):
    """Graphique des mots-clés"""
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:12]]
    scores = [s for _, s in pairs[:12]]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(terms)))
    
    ax.barh(range(len(terms)), scores, color=colors)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1])
    ax.set_xlabel('TF-IDF Score', fontsize=11, fontweight='bold')
    ax.set_title('Top Keywords (TF-IDF)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_sentiment(scores):
    """Graphique du sentiment"""
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax.set_title('Sentiment Analysis (VADER)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_word_frequency(word_freq):
    """Graphique de fréquence des mots"""
    if len(word_freq) == 0:
        return None
    
    words = [w for w, _ in word_freq[:15]]
    counts = [c for _, c in word_freq[:15]]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(words)), counts, color='#3498db', alpha=0.8)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words[::-1])
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Most Frequent Words', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ========== INTERFACE STREAMLIT ==========
st.set_page_config(
    page_title="InsightLens AI Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown('<p class="main-header">🔍 InsightLens AI Pro</p>', unsafe_allow_html=True)
st.markdown("**Advanced Document Analysis & Summarization System**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    num_summary_points = st.slider(
        "Summary sentences",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of key sentences to extract"
    )
    
    num_keywords = st.slider(
        "Keywords to extract",
        min_value=5,
        max_value=25,
        value=15,
        help="Number of distinctive keywords"
    )
    
    show_sentiment = st.checkbox("Sentiment Analysis", value=True)
    show_entities = st.checkbox("Entity Extraction", value=True)
    show_word_freq = st.checkbox("Word Frequency", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    - ✅ Advanced TF-IDF summarization
    - ✅ Multi-gram keyword extraction
    - ✅ VADER sentiment analysis
    - ✅ Entity recognition (numbers, dates)
    - ✅ JSON export for integration
    """)
    
    st.markdown("---")
    st.caption("Powered by scikit-learn & VADER")

# File uploader
uploaded_file = st.file_uploader(
    "📄 Upload your document",
    type=["pdf", "txt", "html", "htm"],
    help="Supported: PDF (text-based), TXT, HTML"
)

if uploaded_file:
    try:
        # Save file
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Read and process
        with st.spinner("📖 Reading document..."):
            text = read_any(temp_path)
        
        if len(text.strip()) < 100:
            st.error("❌ Document too short or empty. Please upload a text-based file.")
            st.stop()
        
        st.success(f"✅ Document loaded: **{len(text):,}** characters")
        
        # Analysis
        with st.spinner("🤖 Analyzing document... This may take 10-30 seconds"):
            summary = advanced_summarize(text, max_sent=num_summary_points)
            keywords = extract_keywords(text, top_n=num_keywords)
            
            sentiment_scores = None
            if show_sentiment:
                sentiment_scores = analyze_sentiment(text)
            
            entities = None
            if show_entities:
                entities = extract_entities(text)
            
            word_freq = None
            if show_word_freq:
                word_freq = word_frequency_analysis(text)
        
        st.success("✅ Analysis complete!")
        
        # Layout: 2 columns
        col1, col2 = st.columns([2, 1])
        
        # === COLUMN 1: Main Results ===
        with col1:
            # Summary
            st.subheader("📋 Executive Summary")
            st.markdown("*Key sentences extracted using advanced TF-IDF + position weighting*")
            
            for i, sentence in enumerate(summary, 1):
                st.markdown(f'<div class="summary-point"><strong>{i}.</strong> {sentence}</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Keywords
            st.subheader("🔑 Top Keywords & Phrases")
            if len(keywords) > 0:
                fig_kw = plot_keywords(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close()
                
                with st.expander("📊 View all keyword scores"):
                    for term, score in keywords:
                        st.text(f"{term}: {score:.4f}")
            else:
                st.info("No keywords extracted")
            
            # Entities
            if entities and len(entities) > 0:
                st.markdown("---")
                st.subheader("🏷️ Extracted Entities")
                
                ent_col1, ent_col2, ent_col3 = st.columns(3)
                
                with ent_col1:
                    if 'monetary_values' in entities:
                        st.markdown("**💰 Monetary Values**")
                        for val in entities['monetary_values'][:5]:
                            st.text(f"• {val}")
                
                with ent_col2:
                    if 'percentages' in entities:
                        st.markdown("**📊 Percentages**")
                        for val in entities['percentages'][:5]:
                            st.text(f"• {val}")
                
                with ent_col3:
                    if 'years' in entities:
                        st.markdown("**📅 Years**")
                        for val in entities['years'][:5]:
                            st.text(f"• {val}")
        
        # === COLUMN 2: Stats & Sentiment ===
        with col2:
            # Document stats
            st.subheader("📊 Document Statistics")
            
            num_words = len(text.split())
            num_sentences = len(split_sentences(text))
            avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
            
            st.metric("Characters", f"{len(text):,}")
            st.metric("Words", f"{num_words:,}")
            st.metric("Sentences", num_sentences)
            st.metric("Avg sentence length", f"{avg_sentence_length:.1f} words")
            
            if len(keywords) > 0:
                st.metric("Top keyword", keywords[0][0])
            
            st.markdown("---")
            
            # Sentiment
            if sentiment_scores:
                st.subheader("💭 Sentiment Analysis")
                
                fig_sent = plot_sentiment(sentiment_scores)
                if fig_sent:
                    st.pyplot(fig_sent)
                    plt.close()
                
                # Interpretation
                compound = sentiment_scores['compound']
                if compound >= 0.05:
                    tone = "**Positive** 😊"
                    color = "green"
                elif compound <= -0.05:
                    tone = "**Negative** 😟"
                    color = "red"
                else:
                    tone = "**Neutral** 😐"
                    color = "gray"
                
                st.markdown(f"**Overall tone:** :{color}[{tone}]")
                st.caption(f"Compound score: {compound:.3f}")
        
        # Word frequency (full width)
        if word_freq and len(word_freq) > 0:
            st.markdown("---")
            st.subheader("📈 Word Frequency Analysis")
            fig_freq = plot_word_frequency(word_freq)
            if fig_freq:
                st.pyplot(fig_freq)
                plt.close()
        
        # Export section
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        export_data = {
            "metadata": {
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "char_count": len(text),
                "word_count": num_words,
                "sentence_count": num_sentences
            },
            "summary": summary,
            "keywords": [{"term": t, "score": float(s)} for t, s in keywords],
            "sentiment": sentiment_scores if sentiment_scores else {},
            "entities": entities if entities else {}
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_export2:
            # Markdown report
            md_report = f"""# Document Analysis Report

**File:** {uploaded_file.name}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{chr(10).join([f"{i}. {s}" for i, s in enumerate(summary, 1)])}

## Top Keywords

{chr(10).join([f"- **{t}**: {s:.4f}" for t, s in keywords[:10]])}

## Statistics

- Characters: {len(text):,}
- Words: {num_words:,}
- Sentences: {num_sentences}

---
*Generated by InsightLens AI Pro*
"""
            
            st.download_button(
                label="📥 Download Markdown",
                data=md_report,
                file_name=f"report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.info("💡 **Troubleshooting:**\n- Ensure PDF is text-based (not scanned)\n- Try a different file format\n- Check file is not corrupted")
        
        with st.expander("🔍 Technical details"):
            st.code(str(e))

else:
    # Welcome screen
    col_intro1, col_intro2 = st.columns([1, 1])
    
    with col_intro1:
        st.info("👈 **Upload a document to start**")
        
        st.markdown("""
        ### 🎯 What is InsightLens AI Pro?
        
        A professional-grade document analysis tool that uses advanced NLP techniques to:
        
        - **Summarize** long documents into key insights
        - **Extract** the most important keywords and phrases
        - **Analyze** sentiment and tone
        - **Identify** entities (numbers, dates, amounts)
        - **Export** results in JSON/Markdown format
        
        ### ✅ Best suited for:
        
        - 📄 Research papers & academic articles
        - 💼 Business reports & white papers
        - 📰 News articles & blog posts
        - 💰 Financial documents & earnings reports
        - 📊 Market research & surveys
        - 📋 Legal documents & contracts
        """)
    
    with col_intro2:
        st.markdown("""
        ### 🚀 How to use
        
        1. **Upload** your document (PDF, TXT, or HTML)
        2. **Configure** analysis settings in the sidebar
        3. **Wait** 10-30 seconds for processing
        4. **Review** the comprehensive analysis
        5. **Export** results in JSON or Markdown
        
        ### ⚙️ Technical features
        
        - **Advanced TF-IDF** with position weighting
        - **Multi-gram extraction** (1-3 word phrases)
        - **VADER sentiment** with chunk averaging
        - **Regex-based entity** recognition
        - **Robust error handling** for any file type
        
        ### ⚠️ Limitations
        
        - Max **50,000 characters** per document
        - **Text-based PDFs only** (no OCR for scanned images)
        - Optimized for **English** (works with French but less precise)
        - Processing time: **10-30 seconds** depending on document size
        
        ### 💡 Tips for best results
        
        - Use well-formatted documents with clear paragraphs
        - Longer documents (1000+ words) give better keyword extraction
        - Technical/financial documents benefit most from entity extraction
        """)



