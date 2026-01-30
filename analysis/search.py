"""
Search and similarity functions for Presidential Speech Dashboard.
Provides BM25 full-text search and TF-IDF similarity.
"""

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# No constants needed currently


def clean_for_search(text: str) -> str:
    """Clean text for search indexing."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    return ' '.join(text.lower().split())


class SpeechSearchEngine:
    """BM25-based full-text search engine for speeches."""

    def __init__(self, df: pd.DataFrame, text_col: str = 'transcript'):
        """
        Initialize search engine with a DataFrame of speeches.

        Args:
            df: DataFrame with speech data
            text_col: Name of column containing text to search
        """
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self._index = None
        self._corpus = None

    def _build_index(self):
        """Build the BM25 index."""
        if self._index is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._index = False
            return

        # Tokenize corpus
        self._corpus = []
        for text in self.df[self.text_col].fillna(''):
            cleaned = clean_for_search(str(text))
            tokens = cleaned.split()
            self._corpus.append(tokens)

        self._index = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for speeches matching a query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of dicts with speech info and relevance scores
        """
        self._build_index()

        if self._index is False or self._index is None:
            return self._fallback_search(query, top_k)

        query_tokens = clean_for_search(query).split()
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                row = self.df.iloc[idx]
                results.append({
                    'index': int(idx),
                    'score': float(scores[idx]),
                    'name': row.get('name', 'Unknown'),
                    'title': row.get('title', 'Unknown'),
                    'year': row.get('year', 'Unknown'),
                    'date': row.get('date', 'Unknown'),
                    'excerpt': self._get_excerpt(str(row.get(self.text_col, '')), query),
                })

        return results

    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword search fallback."""
        query_terms = clean_for_search(query).split()
        if not query_terms:
            return []

        scores = []
        for idx, row in self.df.iterrows():
            text = clean_for_search(str(row.get(self.text_col, '')))
            score = sum(text.count(term) for term in query_terms)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                row = self.df.iloc[idx]
                results.append({
                    'index': int(idx),
                    'score': float(score),
                    'name': row.get('name', 'Unknown'),
                    'title': row.get('title', 'Unknown'),
                    'year': row.get('year', 'Unknown'),
                    'date': row.get('date', 'Unknown'),
                    'excerpt': self._get_excerpt(str(row.get(self.text_col, '')), query),
                })

        return results

    def _get_excerpt(self, text: str, query: str, context_words: int = 30) -> str:
        """Extract excerpt around query match."""
        text_lower = text.lower()
        query_terms = clean_for_search(query).split()

        # Find first occurrence of any query term
        best_pos = len(text)
        for term in query_terms:
            pos = text_lower.find(term)
            if 0 <= pos < best_pos:
                best_pos = pos

        if best_pos == len(text):
            # No match found, return beginning
            words = text.split()[:context_words * 2]
            return ' '.join(words) + '...'

        # Get surrounding context
        words = text.split()
        word_positions = []
        current_pos = 0
        for i, word in enumerate(words):
            word_positions.append(current_pos)
            current_pos += len(word) + 1

        # Find word index closest to match position
        center_word = 0
        for i, pos in enumerate(word_positions):
            if pos >= best_pos:
                center_word = max(0, i - 1)
                break
        else:
            center_word = len(words) - 1

        start = max(0, center_word - context_words)
        end = min(len(words), center_word + context_words)

        excerpt = ' '.join(words[start:end])
        if start > 0:
            excerpt = '...' + excerpt
        if end < len(words):
            excerpt = excerpt + '...'

        return excerpt


def compute_tfidf_matrix(texts: List[str]):
    """
    Compute TF-IDF matrix for texts.

    Returns vectorizer and matrix, or (None, None) on failure.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None, None

    cleaned_texts = [clean_for_search(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(cleaned_texts)
        return vectorizer, matrix
    except ValueError:
        return None, None


def find_similar_speeches(
    df: pd.DataFrame,
    speech_idx: int,
    text_col: str = 'transcript',
    top_k: int = 5
) -> List[Dict]:
    """
    Find speeches similar to a given speech using TF-IDF cosine similarity.

    Args:
        df: DataFrame with speech data
        speech_idx: Index of the reference speech
        text_col: Name of text column
        top_k: Number of similar speeches to return

    Returns:
        List of dicts with similar speech info and similarity scores
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return []

    texts = df[text_col].fillna('').tolist()

    vectorizer, tfidf_matrix = compute_tfidf_matrix(texts)
    if vectorizer is None:
        return []

    # Compute similarities to the target speech
    target_vector = tfidf_matrix[speech_idx]
    similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()

    # Get top k most similar (excluding self)
    similar_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in similar_indices:
        if idx != speech_idx and len(results) < top_k:
            row = df.iloc[idx]
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'name': row.get('name', 'Unknown'),
                'title': row.get('title', 'Unknown'),
                'year': row.get('year', 'Unknown'),
                'date': row.get('date', 'Unknown'),
            })

    return results


def export_to_csv(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """
    Export DataFrame to CSV string.

    Args:
        df: DataFrame to export
        columns: Columns to include (None = all)

    Returns:
        CSV string
    """
    if columns:
        available_cols = [c for c in columns if c in df.columns]
        export_df = df[available_cols]
    else:
        # Exclude transcript by default (too large)
        export_df = df.drop(columns=['transcript'], errors='ignore')

    return export_df.to_csv(index=False)


def highlight_entities_in_text(text: str, max_length: int = 2000) -> str:
    """
    Return text with markdown highlighting around recognized entities.

    Uses NLTK for named entity recognition.
    """
    if not text:
        return ""

    text = text[:max_length] if len(text) > max_length else text

    try:
        import nltk
        from nltk import pos_tag, word_tokenize, ne_chunk
        from nltk.tree import Tree

        # Tokenize and tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged)

        # Extract entities with their positions
        entities = []
        current_pos = 0

        for item in chunked:
            if isinstance(item, Tree):
                entity_text = ' '.join(word for word, tag in item.leaves())
                entity_label = item.label()
                # Find position in original text
                start = text.find(entity_text, current_pos)
                if start >= 0:
                    entities.append((start, start + len(entity_text), entity_text, entity_label))
                    current_pos = start + len(entity_text)

        if not entities:
            return text

        # Build highlighted text
        result = []
        last_end = 0

        for start, end, ent_text, label in sorted(entities):
            if start >= last_end:
                result.append(text[last_end:start])
                result.append(f"**{ent_text}** ({label})")
                last_end = end

        result.append(text[last_end:])

        return ''.join(result)

    except Exception:
        return text
