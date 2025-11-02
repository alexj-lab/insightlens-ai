import streamlit as st
import os
import re
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import json
from datetime import datetime

MAX_CHARS = 50000
MIN_SENTENCE_LENGTH = 20
MAX_SUMMARY_CHAR = 150  # Limite par bullet point

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
    """Résumé extractif avec limitation de longueur"""
    sentences = split_sentences(text)
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) <= max_sent:
        # Tronquer les phrases longues
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
        position_scores = np.array([1.0 - (i / len(sentences)) * 0.3 for i in range(len(sentences))])
        combined_scores = tfidf_scores * position_scores
        
        top_indices = combined_scores.argsort()[-max_sent:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        # Récupérer et tronquer les phrases
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        
        # Tronquer si trop long
        return [s[:MAX_SUMMARY_CHAR] + "..." if len(s) > MAX_SUMMARY_CHAR else s for s in summary_sentences]
    
    except Exception as e:
        return [s[:MAX_SUMMARY_CHAR] + "..." if len(s) > MAX_SUMMARY_CHAR else s for s in sentences[:max_sent]]

def extract_keywords(text, top_n=15):
    """FIX CRITIQUE : Gestion correcte des sparse matrices"""
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
        
        # FIX : Convertir sparse matrix → dense array proprement
        if isinstance(X, (coo_matrix, csr_matrix)):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)
        
        # Calculer max par colonne (terme)
        scores_array = X_dense.max(axis=0)
        
        # S'assurer que c'est un array 1D
        if len(scores_array.shape) > 1:
            scores_array = scores_array.flatten()
        
        terms_array = vectorizer.get_feature_names_out()
        
        # Créer liste de tuples avec conversion explicite
        pairs = []
        for i, term in enumerate(terms_array):
            try:
                score = float(scores_array[i])
                if len(str(term)) > 2 and score > 0:
                    pairs.append((str(term), score))
            except (ValueError, TypeError, IndexError):
                continue
        
        # Trier
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]
    
    except Exception as e:
        st.warning(f"Keywords extraction failed: {str(e)}")
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
    """FIX : Regex améliorés pour meilleurs résultats"""
    entities = {}
    
    # Montants : améliorer la détection
    money_pattern = r'(?:€|USD|\$|EUR)\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?|\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\s*(?:€|EUR|USD|\$|million|billion|M|B|bn|mn)'
    money_matches = re.findall(money_pattern, text, re.I)
    
    # Filtrer les faux positifs (pas juste "51" ou "1")
    valid_money = [m for m in money_matches if re.search(r'[€$]|million|billion|EUR|USD|[MB](?:n)?$', m, re.I)]
    
    if valid_money:
        entities['monetary_values'] = list(set(valid_money))[:10]
    
    # Pourcentages
    percent_matches = re.findall(r'\b\d+(?:\.\d+)?%', text)
    if percent_matches:
        entities['percentages'] = list(set(percent_matches))[:10]
    
    # Dates complètes (plus utile que juste années)
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
    date_matches = re.findall(date_pattern, text, re.I)
    if date_matches:
        entities['dates'] = list(set(date_matches))[:10]
    
    return entities

def plot_keywords(pairs):
    if len(pairs) == 0:
        return None
    
    terms = [t for t, _ in pairs[:10]]  # Limiter à 10 pour clarté
    scores = [s for _, s in pairs[:10]]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(terms)))
    ax.barh(range(len(terms)), scores, color=colors, height=0.7)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=10)
    ax.set_xlabel('TF-IDF Score', fontsize=10, fontweight='bold')
    ax.set_title('Top Keywords', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig

def plot_sentiment(scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['positive'], scores['neutral'], scores['negative']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
    ax.set_title('Sentiment Analysis (VADER)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig

# ========== UI ==========
st.set_page_config(page_title="InsightLens AI Pro", page_icon="🔍", layout="wide")

st.title("🔍 InsightLens AI Pro")
st.markdown("**Advanced Document Analysis & Summarization**")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    num_summary = st.slider("Summary points", 3, 8, 5, help="Number of key sentences")
    num_keywords = st.slider("Keywords", 5, 20, 12, help="Top distinctive terms")
    show_sentiment = st.checkbox("Sentiment Analysis", value=True)
    show_entities = st.checkbox("Entity Extraction", value=True)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.caption("TF-IDF extractive summarization")
    st.caption("VADER sentiment analysis")
    st.caption("Regex entity recognition")

uploaded_file = st.file_uploader("📄 Upload document", type=["pdf", "txt", "html", "htm"])

if uploaded_file:
    try:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("📖 Reading document..."):
            text = read_any(temp_path)
        
        # Validation
        words = text.split()
        if len(words) < 30:
            st.error("❌ Document too short (< 30 words)")
            st.info("💡 Upload a document with at least 50 words for meaningful analysis")
            st.stop()
        
        st.success(f"✅ Loaded: **{len(text):,}** characters • **{len(words):,}** words")
        
        # Analysis
        with st.spinner("🤖 Analyzing document..."):
            summary = advanced_summarize(text, max_sent=num_summary)
            keywords = extract_keywords(text, top_n=num_keywords)
            
            sentiment_scores = None
            if show_sentiment:
                sentiment_scores = analyze_sentiment(text)
            
            entities = None
            if show_entities:
                entities = extract_entities(text)
        
        # Check summary
        if len(summary) == 0:
            st.error("❌ Could not generate summary")
            st.info("💡 Document may be too short or poorly formatted")
            st.stop()
        
        st.success("✅ Analysis complete!")
        
        # Layout
        col1, col2 = st.columns([2.5, 1])
        
        with col1:
            st.subheader("📋 Key Points Summary")
            st.caption(f"*Top {len(summary)} sentences (max {MAX_SUMMARY_CHAR} chars each)*")
            
            for i, sentence in enumerate(summary, 1):
                st.markdown(f"**{i}.** {sentence}")
            
            st.markdown("---")
            
            st.subheader("🔑 Top Keywords & Phrases")
            if len(keywords) > 0:
                fig_kw = plot_keywords(keywords)
                if fig_kw:
                    st.pyplot(fig_kw)
                    plt.close()
                
                with st.expander("📊 View all keyword scores"):
                    for term, score in keywords:
                        st.text(f"• {term}: {score:.4f}")
            else:
                st.info("ℹ️ No keywords extracted")
            
            # Entities
            if entities and len(entities) > 0:
                st.markdown("---")
                st.subheader("🏷️ Extracted Entities")
                
                cols = st.columns(3)
                
                if 'monetary_values' in entities and len(entities['monetary_values']) > 0:
                    with cols[0]:
                        st.markdown("**💰 Money**")
                        for v in sorted(set(entities['monetary_values']))[:6]:
                            st.text(f"• {v}")
                
                if 'percentages' in entities and len(entities['percentages']) > 0:
                    with cols[1]:
                        st.markdown("**📊 Percentages**")
                        for v in sorted(set(entities['percentages']), reverse=True)[:6]:
                            st.text(f"• {v}")
                
                if 'dates' in entities and len(entities['dates']) > 0:
                    with cols[2]:
                        st.markdown("**📅 Dates**")
                        for v in entities['dates'][:6]:
                            st.text(f"• {v}")
        
        with col2:
            st.subheader("📊 Document Stats")
            st.metric("Characters", f"{len(text):,}")
            st.metric("Words", f"{len(words):,}")
            
            num_sentences = len(split_sentences(text))
            st.metric("Sentences", num_sentences)
            
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
                
                st.caption(f"Compound: {compound:.3f}")
        
        # Export
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        export_data = {
            "metadata": {
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "characters": len(text),
                "words": len(words),
                "sentences": num_sentences
            },
            "summary": summary,
            "keywords": [{"term": t, "score": float(s)} for t, s in keywords],
            "sentiment": sentiment_scores if sentiment_scores else {},
            "entities": entities if entities else {}
        }
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.download_button(
                label="📥 JSON Export",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_exp2:
            markdown_report = f"""# Document Analysis Report

**File:** {uploaded_file.name}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 📋 Key Points

{chr(10).join([f"{i}. {s}" for i, s in enumerate(summary, 1)])}

## 🔑 Top Keywords

{chr(10).join([f"- **{t}**: {s:.4f}" for t, s in keywords[:8]])}

## 📊 Statistics

- **Characters:** {len(text):,}
- **Words:** {len(words):,}
- **Sentences:** {num_sentences}

---
*Generated by InsightLens AI Pro*
"""
            
            st.download_button(
                label="📥 Markdown Report",
                data=markdown_report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Try a different file or format")
        
        with st.expander("🔍 Debug info"):
            st.code(str(e))

else:
    st.info("👆 **Upload a document to start**")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        ### 🎯 What it does
        
        - **Summarizes** docs into concise bullet points
        - **Extracts** key terms and phrases
        - **Analyzes** sentiment (positive/neutral/negative)
        - **Identifies** entities (money, %, dates)
        - **Exports** results in JSON/Markdown
        
        ### ✅ Best for
        
        - Business reports & contracts
        - Research papers & articles
        - Financial documents
        - Legal texts & case studies
        """)
    
    with col_right:
        st.markdown("""
        ### 🚀 Quick start
        
        1. Upload PDF, TXT, or HTML
        2. Adjust settings in sidebar
        3. Wait 10-20 seconds
        4. Review & export results
        
        ### ⚠️ Limitations
        
        - Max 50,000 characters
        - Text-based PDFs only (no scans)
        - Minimum 30 words required
        - Optimized for English
        
        ### 💡 Pro tips
        
        - Well-formatted docs work best
        - 500+ word docs give better results
        - Clear paragraphs improve accuracy
        """)
