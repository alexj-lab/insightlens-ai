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

MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20

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
    raise ValueError("Cannot decode text file")

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
        st.warning(f"⚠️ Document truncated to {MAX_CHARS:,} chars")
    
    return text

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENTENCE_LENGTH]

def advanced_summarize(text, max_sent=5):
    sentences = split_sentences(text)
    
    if len(sentences) <= max_sent:
        return sentences
    
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
        position_scores = np.array([1.0 - (i / len(sentences)) * 0.3 for i in range(len(sentences))])
        combined_scores = tfidf_scores * position_scores
        
        top_indices = combined_scores.argsort()[-max_sent:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        return [sentences[i] for i in top_indices_sorted]
    except:
        return sentences[:max_sent]

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
        scores_matrix = X.max(axis=0)
        scores_array = np.array(scores_matrix).flatten()
        terms_array = vectorizer.get_feature_names_out()
        
        # Créer liste explicite
        pairs = []
        for term, score in zip(terms_array, scores_array):
            if len(str(term)) > 2:
                pairs.append((str(term), float(score)))
        
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]
    
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}")
        return []

def analyze_sentiment(text):
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
    entities = {}
    
    money_matches = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|M|B|bn))?\b', text, re.I)
    if money_matches:
        entities['monetary_values'] = money_matches[:10]
    
    percent_matches = re.findall(r'\d+(?:\.\d+)?%', text)
    if percent_matches:
        entities['percentages'] = percent_matches[:10]
    
    year_matches = re.findall(r'\b(19|20)\d{2}\b', text)
    if year_matches:
        entities['years'] = list(set(year_matches))[:10]
    
    return entities

def plot_keywords(pairs):
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
    ax.set_title('Top Keywords', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_sentiment(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax.set_title('Sentiment Analysis', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# ========== UI ==========
st.set_page_config(page_title="InsightLens AI Pro", page_icon="🔍", layout="wide")

st.title("🔍 InsightLens AI Pro")
st.markdown("**Advanced Document Analysis & Summarization**")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    num_summary = st.slider("Summary sentences", 3, 10, 5)
    num_keywords = st.slider("Keywords", 5, 25, 15)
    show_sentiment = st.checkbox("Sentiment Analysis", value=True)
    show_entities = st.checkbox("Entity Extraction", value=True)
    
    st.markdown("---")
    st.caption("Powered by TF-IDF & VADER")

uploaded_file = st.file_uploader("📄 Upload document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("📖 Reading..."):
            text = read_any(temp_path)
        
        # FIX : Validation plus permissive
        words = text.split()
        if len(words) < 20:
            st.error("❌ Document too short (< 20 words)")
            st.stop()
        
        if len(text.strip()) < 50:
            st.warning("⚠️ Very short document. Results may be limited.")
        
        st.success(f"✅ Loaded: {len(text):,} characters")
        
        with st.spinner("🤖 Analyzing..."):
            summary = advanced_summarize(text, max_sent=num_summary)
            keywords = extract_keywords(text, top_n=num_keywords)
            
            sentiment_scores = None
            if show_sentiment:
                sentiment_scores = analyze_sentiment(text)
            
            entities = None
            if show_entities:
                entities = extract_entities(text)
        
        st.success("✅ Analysis complete!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Executive Summary")
            for i, sentence in enumerate(summary, 1):
                st.markdown(f"**{i}.** {sentence}")
            
            st.markdown("---")
            
            st.subheader("🔑 Top Keywords")
            if len(keywords) > 0:
                fig_kw = plot_keywords(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close()
                
                with st.expander("📊 All scores"):
                    for term, score in keywords:
                        st.text(f"{term}: {score:.4f}")
            else:
                st.info("No keywords extracted")
            
            if entities and len(entities) > 0:
                st.markdown("---")
                st.subheader("🏷️ Entities")
                
                e1, e2, e3 = st.columns(3)
                with e1:
                    if 'monetary_values' in entities:
                        st.markdown("**💰 Money**")
                        for v in entities['monetary_values'][:5]:
                            st.text(f"• {v}")
                with e2:
                    if 'percentages' in entities:
                        st.markdown("**📊 Percentages**")
                        for v in entities['percentages'][:5]:
                            st.text(f"• {v}")
                with e3:
                    if 'years' in entities:
                        st.markdown("**📅 Years**")
                        for v in entities['years'][:5]:
                            st.text(f"• {v}")
        
        with col2:
            st.subheader("📊 Stats")
            st.metric("Characters", f"{len(text):,}")
            st.metric("Words", f"{len(words):,}")
            st.metric("Sentences", len(split_sentences(text)))
            if len(keywords) > 0:
                st.metric("Top keyword", keywords[0][0])
            
            st.markdown("---")
            
            if sentiment_scores:
                st.subheader("💭 Sentiment")
                fig_sent = plot_sentiment(sentiment_scores)
                if fig_sent:
                    st.pyplot(fig_sent)
                    plt.close()
                
                compound = sentiment_scores['compound']
                if compound >= 0.05:
                    st.success("**Positive tone** 😊")
                elif compound <= -0.05:
                    st.error("**Negative tone** 😟")
                else:
                    st.info("**Neutral tone** 😐")
        
        st.markdown("---")
        st.subheader("💾 Export")
        
        export_data = {
            "filename": uploaded_file.name,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "keywords": [{"term": t, "score": s} for t, s in keywords],
            "sentiment": sentiment_scores if sentiment_scores else {},
            "entities": entities if entities else {}
        }
        
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Try: text-based PDF (not scanned), or different file")

else:
    st.info("👆 Upload a document to start")
    st.markdown("""
    ### 🎯 Features
    - Advanced TF-IDF summarization
    - Multi-gram keyword extraction
    - VADER sentiment analysis
    - Entity extraction (money, %, dates)
    - JSON export
    
    ### ✅ Best for
    Reports • Papers • Articles • Financial docs
    
    ### ⚠️ Limitations
    - Max 50K characters
    - Text PDFs only (no scans)
    - Optimized for English
    """)
