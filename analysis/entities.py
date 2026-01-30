"""
Named Entity Recognition functions for Presidential Speech Dashboard.
Uses NLTK for entity extraction, counting, and trend analysis.
"""

import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from utils.constants import MAX_TEXT_LENGTH, ENTITY_TYPES


def extract_entities(text: str, types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract named entities from text using NLTK.

    Args:
        text: Text to analyze
        types: List of entity types to extract (e.g., ['PERSON', 'GPE', 'ORGANIZATION'])
               If None, extracts all available types

    Returns:
        Dictionary mapping entity types to lists of entity names
    """
    if not text or not isinstance(text, str):
        return {}

    if MAX_TEXT_LENGTH is not None and len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    if types is None:
        types = list(ENTITY_TYPES.keys())

    try:
        import nltk
        from nltk import pos_tag, word_tokenize, ne_chunk
        from nltk.tree import Tree

        # Tokenize and POS tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # Named entity chunking
        chunked = ne_chunk(tagged)

        entities = {t: [] for t in types}

        # NLTK uses different labels than spaCy
        # Map NLTK labels to our standardized types
        label_map = {
            'PERSON': 'PERSON',
            'GPE': 'GPE',  # Geopolitical entity (countries, cities, states)
            'ORGANIZATION': 'ORG',
            'FACILITY': 'LOC',
            'GSP': 'GPE',  # Geo-socio-political
            'LOCATION': 'LOC',
        }

        for subtree in chunked:
            if isinstance(subtree, Tree):
                entity_label = subtree.label()
                entity_text = ' '.join(word for word, tag in subtree.leaves())

                # Map to our label system
                mapped_label = label_map.get(entity_label, entity_label)

                if mapped_label in types and entity_text and len(entity_text) > 1:
                    entities[mapped_label].append(entity_text)

        return entities

    except Exception as e:
        print(f"Error extracting entities: {e}")
        return _extract_entities_fallback(text, types)


def _extract_entities_fallback(text: str, types: List[str]) -> Dict[str, List[str]]:
    """
    Fallback entity extraction using simple pattern matching.
    Looks for capitalized phrases as potential entities.
    """
    entities = {t: [] for t in types}

    # Find capitalized words/phrases (potential proper nouns)
    # Pattern for capitalized words that aren't at sentence start
    capitalized = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|(?<=\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)

    # Common titles that indicate a person
    person_titles = {'Mr', 'Mrs', 'Ms', 'Dr', 'President', 'Senator', 'General', 'Admiral', 'Secretary', 'Governor', 'Mayor'}

    # Common country/place indicators
    place_words = {'America', 'United', 'States', 'Congress', 'Senate', 'House', 'Nation', 'Republic', 'Kingdom', 'Empire'}

    for phrase in capitalized:
        words = phrase.split()
        if not words:
            continue

        first_word = words[0]

        # Simple heuristic classification
        if first_word in person_titles or (len(words) >= 2 and words[0][0].isupper() and words[-1][0].isupper()):
            if 'PERSON' in types:
                entities['PERSON'].append(phrase)
        elif any(w in place_words for w in words):
            if 'GPE' in types:
                entities['GPE'].append(phrase)
        elif 'ORG' in types and len(words) >= 2:
            entities['ORG'].append(phrase)

    return entities


def extract_entities_with_counts(text: str, types: Optional[List[str]] = None) -> Dict[str, Counter]:
    """
    Extract named entities with frequency counts.

    Args:
        text: Text to analyze
        types: List of entity types to extract

    Returns:
        Dictionary mapping entity types to Counter objects
    """
    entities = extract_entities(text, types)
    return {t: Counter(ents) for t, ents in entities.items()}


def get_top_entities(
    texts: List[str],
    entity_type: str = 'PERSON',
    top_k: int = 20
) -> List[tuple]:
    """
    Get top entities from a list of texts.

    Args:
        texts: List of text strings
        entity_type: Type of entity to extract
        top_k: Number of top entities to return

    Returns:
        List of (entity_name, count) tuples
    """
    all_entities = Counter()

    for text in texts:
        if not text:
            continue
        entities = extract_entities_with_counts(text, [entity_type])
        if entity_type in entities:
            all_entities.update(entities[entity_type])

    return all_entities.most_common(top_k)


def top_entities_by_period(
    df: pd.DataFrame,
    time_col: str = 'year',
    entity_type: str = 'PERSON',
    text_col: str = 'transcript',
    top_k: int = 10
) -> pd.DataFrame:
    """
    Get top entities grouped by time period.

    Args:
        df: DataFrame with text and time columns
        time_col: Name of time column
        entity_type: Type of entity to extract
        text_col: Name of text column
        top_k: Number of top entities per period

    Returns:
        DataFrame with period, entity, and count columns
    """
    if df.empty or text_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    results = []

    # Group by time period
    for period, group in df.groupby(time_col):
        texts = group[text_col].dropna().tolist()
        top_ents = get_top_entities(texts, entity_type, top_k)

        for entity, count in top_ents:
            results.append({
                time_col: period,
                'entity': entity,
                'count': count,
                'entity_type': entity_type,
            })

    return pd.DataFrame(results)


def entity_mentions_over_time(
    df: pd.DataFrame,
    entity_name: str,
    text_col: str = 'transcript',
    time_col: str = 'year'
) -> pd.DataFrame:
    """
    Track mentions of a specific entity over time.

    Args:
        df: DataFrame with text and time columns
        entity_name: Name of entity to track
        text_col: Name of text column
        time_col: Name of time column

    Returns:
        DataFrame with time period and mention counts
    """
    if df.empty or text_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    results = []
    entity_lower = entity_name.lower()

    for _, row in df.iterrows():
        text = str(row.get(text_col, '')).lower()
        time_val = row.get(time_col)

        if pd.isna(time_val):
            continue

        # Simple string matching for the entity
        count = len(re.findall(r'\b' + re.escape(entity_lower) + r'\b', text))

        results.append({
            time_col: time_val,
            'mentions': count,
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    return result_df.groupby(time_col)['mentions'].sum().reset_index()


def entity_cooccurrence(
    texts: List[str],
    entity_types: List[str] = ['PERSON', 'ORG'],
    min_count: int = 2
) -> pd.DataFrame:
    """
    Find entities that frequently appear together.

    Args:
        texts: List of text strings
        entity_types: Types of entities to consider
        min_count: Minimum co-occurrence count to include

    Returns:
        DataFrame with entity pairs and co-occurrence counts
    """
    from itertools import combinations

    cooccurrences = Counter()

    for text in texts:
        if not text:
            continue

        # Get all entities from this text
        entities = extract_entities(text, entity_types)
        all_ents = set()
        for ent_list in entities.values():
            all_ents.update(ent_list)

        # Count pairs
        for pair in combinations(sorted(all_ents), 2):
            cooccurrences[pair] += 1

    # Filter and convert to DataFrame
    results = [
        {'entity1': pair[0], 'entity2': pair[1], 'count': count}
        for pair, count in cooccurrences.items()
        if count >= min_count
    ]

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('count', ascending=False).reset_index(drop=True)

    return df


def get_entity_summary(texts: List[str]) -> Dict:
    """
    Get summary of all entity types found in texts.

    Returns dict with counts per entity type.
    """
    summary = {}

    for entity_type, label in ENTITY_TYPES.items():
        top_ents = get_top_entities(texts, entity_type, top_k=10)
        summary[entity_type] = {
            'label': label,
            'top_entities': top_ents,
            'total_unique': len(set(e for e, _ in top_ents)),
        }

    return summary
