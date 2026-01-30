"""
Topic modeling and word analysis functions for Presidential Speech Dashboard.
Provides word clouds, n-grams, keyword trends, and LDA topic modeling.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from utils.constants import (
    MAX_TEXT_LENGTH,
    DEFAULT_N_TOPICS,
    WORDCLOUD_MAX_WORDS,
    WORDCLOUD_WIDTH,
    WORDCLOUD_HEIGHT,
    DEFAULT_NGRAM_TOP_K,
)


def get_stopwords() -> set:
    """Get English stopwords from NLTK."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception:
        # Fallback basic stopwords
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'we', 'they', 'what', 'which', 'who', 'whom', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
            'once', 'upon', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'if',
            'because', 'until', 'while', 'about', 'against', 'any', 'its', 'his',
            'her', 'their', 'our', 'your', 'my', 'me', 'him', 'them', 'us',
            'one', 'two', 'first', 'new', 'even', 'much', 'many', 'get', 'well',
            'make', 'made', 'say', 'said', 'like', 'come', 'came', 'go', 'went',
            'take', 'took', 'know', 'knew', 'see', 'saw', 'think', 'thought',
            'give', 'gave', 'tell', 'told', 'put', 'set', 'let', 'yet', 'still',
            'mr', 'mrs', 'ms', 'sir', 'dear', 'speaker', 'members', 'fellow',
            'citizens', 'congress', 'gentleman', 'gentlemen', 'lady', 'ladies',
            'house', 'senate', 'representatives', 'president', 'honorable',
        }


def clean_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    if not text or not isinstance(text, str):
        return ""
    if MAX_TEXT_LENGTH is not None and len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.lower()


def get_word_frequencies(text: str, remove_stopwords: bool = True) -> Counter:
    """Get word frequency counts from text."""
    text = clean_text(text)
    if not text:
        return Counter()

    words = re.findall(r'\b[a-z]{3,}\b', text)

    if remove_stopwords:
        stopwords = get_stopwords()
        words = [w for w in words if w not in stopwords]

    return Counter(words)


def generate_wordcloud(texts: List[str], max_words: int = WORDCLOUD_MAX_WORDS):
    """
    Generate a word cloud from a list of texts.

    Returns a WordCloud object or None if generation fails.
    """
    try:
        from wordcloud import WordCloud

        # Combine all texts
        combined = ' '.join(clean_text(t) for t in texts if t)
        if not combined:
            return None

        # Get word frequencies
        freq = get_word_frequencies(combined, remove_stopwords=True)
        if not freq:
            return None

        # Generate word cloud
        wc = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            max_words=max_words,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=0.7,
        )

        wc.generate_from_frequencies(freq)
        return wc

    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None


def extract_ngrams(texts: List[str], n: int = 2, top_k: int = DEFAULT_NGRAM_TOP_K) -> List[Tuple[str, int]]:
    """
    Extract top n-grams from a list of texts.

    Args:
        texts: List of text strings
        n: Size of n-grams (2 for bigrams, 3 for trigrams)
        top_k: Number of top n-grams to return

    Returns:
        List of (ngram_string, count) tuples
    """
    try:
        from nltk import ngrams as nltk_ngrams
        from nltk.tokenize import word_tokenize
    except ImportError:
        return _extract_ngrams_fallback(texts, n, top_k)

    stopwords = get_stopwords()
    all_ngrams = []

    for text in texts:
        text = clean_text(text)
        if not text:
            continue

        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # Filter tokens
        tokens = [t for t in tokens if t.isalpha() and len(t) >= 3 and t not in stopwords]

        # Extract n-grams
        text_ngrams = list(nltk_ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)

    # Count and return top k
    ngram_counts = Counter(all_ngrams)
    top_ngrams = ngram_counts.most_common(top_k)

    return [(' '.join(ng), count) for ng, count in top_ngrams]


def _extract_ngrams_fallback(texts: List[str], n: int, top_k: int) -> List[Tuple[str, int]]:
    """Fallback n-gram extraction without NLTK."""
    stopwords = get_stopwords()
    all_ngrams = []

    for text in texts:
        text = clean_text(text)
        if not text:
            continue

        tokens = [t for t in text.split() if t.isalpha() and len(t) >= 3 and t not in stopwords]

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            all_ngrams.append(ngram)

    ngram_counts = Counter(all_ngrams)
    top_ngrams = ngram_counts.most_common(top_k)

    return [(' '.join(ng), count) for ng, count in top_ngrams]


def keyword_frequency_over_time(
    df: pd.DataFrame,
    keywords: List[str],
    text_col: str = 'transcript',
    time_col: str = 'year'
) -> pd.DataFrame:
    """
    Track keyword frequencies over time.

    Args:
        df: DataFrame with text and time columns
        keywords: List of keywords to track
        text_col: Name of text column
        time_col: Name of time column

    Returns:
        DataFrame with keyword counts per time period
    """
    if df.empty or text_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    results = []

    for _, row in df.iterrows():
        text = clean_text(str(row.get(text_col, '')))
        time_val = row.get(time_col)

        if not text or pd.isna(time_val):
            continue

        words = text.split()
        word_counts = Counter(words)

        row_data = {time_col: time_val}
        for keyword in keywords:
            row_data[keyword] = word_counts.get(keyword.lower(), 0)

        results.append(row_data)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # Aggregate by time period
    return result_df.groupby(time_col).sum().reset_index()


def build_topic_model(texts: List[str], n_topics: int = DEFAULT_N_TOPICS) -> Dict:
    """
    Build an LDA topic model from texts.

    Args:
        texts: List of text strings
        n_topics: Number of topics to extract

    Returns:
        Dictionary with 'topics' (list of top words per topic),
        'doc_topics' (dominant topic per document),
        and 'topic_distributions' (full distribution per document)
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError:
        return {"topics": [], "doc_topics": [], "topic_distributions": []}

    # Clean texts
    cleaned_texts = [clean_text(t) for t in texts if t]
    if len(cleaned_texts) < n_topics:
        return {"topics": [], "doc_topics": [], "topic_distributions": []}

    stopwords = list(get_stopwords())

    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words=stopwords,
    )

    try:
        dtm = vectorizer.fit_transform(cleaned_texts)
    except ValueError:
        return {"topics": [], "doc_topics": [], "topic_distributions": []}

    feature_names = vectorizer.get_feature_names_out()

    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_method='online',
    )

    doc_topic_dist = lda.fit_transform(dtm)

    # Extract top words per topic
    topics = []
    n_top_words = 10
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics.append({
            "topic_id": topic_idx,
            "words": top_words,
            "weights": [float(topic[i]) for i in top_word_indices],
        })

    # Get dominant topic per document
    doc_topics = [int(np.argmax(dist)) for dist in doc_topic_dist]

    return {
        "topics": topics,
        "doc_topics": doc_topics,
        "topic_distributions": doc_topic_dist.tolist(),
    }


def get_top_words(texts: List[str], top_k: int = 50) -> List[Tuple[str, int]]:
    """Get top k most frequent words from texts."""
    combined_freq = Counter()

    for text in texts:
        freq = get_word_frequencies(text, remove_stopwords=True)
        combined_freq.update(freq)

    return combined_freq.most_common(top_k)
