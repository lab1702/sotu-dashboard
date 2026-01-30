"""
Emotion analysis functions for Presidential Speech Dashboard.
Uses a lexicon-based approach for multi-emotion detection and tone analysis.
"""

import re
from typing import Dict, List, Optional
from collections import Counter

import pandas as pd
import numpy as np

from utils.constants import MAX_TEXT_LENGTH, EMOTION_CATEGORIES

# Emotion lexicons - curated word lists for each emotion category
# Based on common emotion lexicon patterns (simplified NRC-style)
EMOTION_LEXICON = {
    "fear": {
        "fear", "afraid", "scared", "terror", "terrified", "horror", "dread",
        "panic", "anxiety", "anxious", "worry", "worried", "threat", "danger",
        "dangerous", "risk", "alarming", "frightening", "terrifying", "nightmare",
        "peril", "menace", "intimidate", "coward", "trembling", "apprehension",
        "distress", "uneasy", "nervous", "phobia", "scare", "fright", "alarm",
        "shock", "startled", "threatened", "vulnerable", "insecure", "uncertain"
    },
    "anger": {
        "anger", "angry", "rage", "furious", "fury", "outrage", "outraged",
        "hate", "hatred", "hostile", "hostility", "aggression", "aggressive",
        "violent", "violence", "resentment", "bitter", "bitterness", "irritate",
        "irritated", "annoy", "annoyed", "frustrated", "frustration", "mad",
        "enraged", "infuriated", "livid", "indignant", "wrath", "vengeful",
        "revenge", "spite", "malice", "contempt", "scorn", "disgust", "loathe"
    },
    "anticipation": {
        "anticipation", "anticipate", "expect", "expected", "expectation", "hope",
        "hopeful", "hoping", "await", "awaiting", "eager", "eagerly", "look forward",
        "optimism", "optimistic", "future", "prospect", "potential", "possibility",
        "promise", "promising", "aspire", "aspiration", "dream", "vision", "plan",
        "planning", "prepare", "preparation", "ready", "excitement", "excited"
    },
    "trust": {
        "trust", "trusting", "faith", "faithful", "believe", "belief", "confidence",
        "confident", "rely", "reliable", "dependable", "honest", "honesty", "loyal",
        "loyalty", "sincere", "sincerity", "integrity", "credible", "authentic",
        "genuine", "truthful", "devoted", "devotion", "committed", "commitment",
        "assured", "assurance", "secure", "security", "safe", "safety", "protect",
        "protection", "support", "supportive", "ally", "alliance", "friend", "friendship"
    },
    "surprise": {
        "surprise", "surprised", "surprising", "amazed", "amazing", "astonished",
        "astonishing", "astounded", "shocked", "shocking", "stunned", "stunning",
        "unexpected", "unforeseen", "sudden", "suddenly", "remarkable", "incredible",
        "unbelievable", "extraordinary", "wonder", "wonderful", "miracle", "miraculous",
        "startling", "bewildered", "dumbfounded", "speechless", "awe", "awesome"
    },
    "sadness": {
        "sad", "sadness", "sorrow", "sorrowful", "grief", "grieving", "mourn",
        "mourning", "depressed", "depression", "melancholy", "despair", "despairing",
        "hopeless", "hopelessness", "misery", "miserable", "unhappy", "unhappiness",
        "heartbreak", "heartbroken", "tragedy", "tragic", "loss", "lost", "suffer",
        "suffering", "pain", "painful", "anguish", "agony", "tears", "cry", "crying",
        "weep", "weeping", "lament", "regret", "remorse", "lonely", "loneliness"
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "revolting", "repulsive", "repugnant",
        "abhorrent", "loathsome", "nauseating", "sickening", "vile", "foul", "gross",
        "offensive", "objectionable", "distasteful", "unpleasant", "nasty", "horrible",
        "horrid", "awful", "terrible", "dreadful", "appalling", "shameful", "despicable",
        "contemptible", "detestable", "odious", "abominable", "corrupt", "corruption"
    },
    "joy": {
        "joy", "joyful", "joyous", "happy", "happiness", "delight", "delighted",
        "delightful", "pleased", "pleasure", "glad", "gladness", "cheerful", "cheer",
        "merry", "elated", "elation", "ecstatic", "ecstasy", "euphoric", "euphoria",
        "thrilled", "thrilling", "excited", "exciting", "celebrate", "celebration",
        "triumph", "triumphant", "victory", "victorious", "success", "successful",
        "wonderful", "fantastic", "excellent", "great", "love", "loving", "blessed"
    },
    "positive": {
        "good", "great", "excellent", "wonderful", "amazing", "fantastic", "terrific",
        "outstanding", "superb", "brilliant", "magnificent", "splendid", "marvelous",
        "fabulous", "glorious", "beautiful", "perfect", "ideal", "best", "better",
        "success", "successful", "win", "winner", "victory", "achieve", "achievement",
        "accomplish", "accomplishment", "progress", "improve", "improvement", "benefit",
        "beneficial", "advantage", "gain", "prosper", "prosperity", "thrive", "flourish",
        "peace", "peaceful", "harmony", "harmonious", "freedom", "free", "liberty",
        "opportunity", "hope", "hopeful", "optimism", "optimistic", "bright", "promising"
    },
    "negative": {
        "bad", "worse", "worst", "terrible", "horrible", "awful", "dreadful", "poor",
        "fail", "failure", "failing", "defeat", "defeated", "loss", "lose", "losing",
        "problem", "trouble", "crisis", "disaster", "catastrophe", "tragedy", "tragic",
        "wrong", "error", "mistake", "fault", "blame", "guilty", "crime", "criminal",
        "corrupt", "corruption", "evil", "wicked", "sinful", "immoral", "unjust",
        "unfair", "injustice", "cruel", "cruelty", "violent", "violence", "war",
        "conflict", "struggle", "suffer", "suffering", "pain", "painful", "hurt",
        "harm", "damage", "destroy", "destruction", "death", "dead", "die", "dying"
    }
}


def _tokenize(text: str) -> List[str]:
    """Simple tokenization for emotion analysis."""
    text = text.lower()
    # Remove punctuation and split
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def analyze_emotions(text: str) -> Dict[str, float]:
    """
    Analyze emotions in text using lexicon-based approach.

    Returns dictionary with scores for each emotion category:
    fear, anger, anticipation, trust, surprise, positive, negative,
    sadness, disgust, joy.
    """
    if not text or not isinstance(text, str):
        return {emotion: 0.0 for emotion in EMOTION_CATEGORIES}

    if MAX_TEXT_LENGTH is not None and len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    words = _tokenize(text)
    word_count = len(words)

    if word_count == 0:
        return {emotion: 0.0 for emotion in EMOTION_CATEGORIES}

    word_set = set(words)
    word_counter = Counter(words)

    scores = {}
    for emotion in EMOTION_CATEGORIES:
        lexicon = EMOTION_LEXICON.get(emotion, set())
        # Count matches (with frequency)
        matches = sum(word_counter[word] for word in word_set & lexicon)
        # Normalize by word count and scale
        scores[emotion] = round(matches / word_count * 100, 3)

    return scores


def get_emotional_intensity(text: str) -> float:
    """
    Calculate overall emotional intensity score.

    Returns a single value representing total emotional content.
    """
    emotions = analyze_emotions(text)
    # Sum all emotion scores (excluding positive/negative which are meta-categories)
    core_emotions = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']
    return round(sum(emotions.get(e, 0) for e in core_emotions), 3)


def get_dominant_emotion(text: str) -> str:
    """Get the dominant emotion in a text."""
    emotions = analyze_emotions(text)
    # Exclude meta-categories
    core_emotions = {k: v for k, v in emotions.items() if k not in ['positive', 'negative']}
    if not core_emotions or all(v == 0 for v in core_emotions.values()):
        return "neutral"
    return max(core_emotions, key=core_emotions.get)


def analyze_tone_shifts(text: str, segment_size: int = 500) -> pd.DataFrame:
    """
    Analyze how emotional tone shifts within a single speech.

    Splits text into segments and analyzes each.

    Args:
        text: Full text to analyze
        segment_size: Approximate words per segment

    Returns:
        DataFrame with segment emotions
    """
    if not text or not isinstance(text, str):
        return pd.DataFrame()

    # No text truncation - analyze full speech

    words = text.split()
    if len(words) < segment_size:
        # Single segment
        emotions = analyze_emotions(text)
        emotions['segment'] = 1
        emotions['segment_start'] = 0
        emotions['segment_end'] = len(words)
        return pd.DataFrame([emotions])

    segments = []
    num_segments = max(1, len(words) // segment_size)

    for i in range(num_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(words))
        segment_text = ' '.join(words[start:end])

        emotions = analyze_emotions(segment_text)
        emotions['segment'] = i + 1
        emotions['segment_start'] = start
        emotions['segment_end'] = end
        segments.append(emotions)

    return pd.DataFrame(segments)


def emotion_trends_over_time(
    df: pd.DataFrame,
    text_col: str = 'transcript',
    time_col: str = 'year',
    emotions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Track how emotional content changes over time.

    Args:
        df: DataFrame with text and time columns
        text_col: Name of text column
        time_col: Name of time column
        emotions: List of emotions to track (default: all)

    Returns:
        DataFrame with average emotion scores per time period
    """
    if df.empty or text_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    if emotions is None:
        emotions = EMOTION_CATEGORIES

    results = []

    for _, row in df.iterrows():
        text = row.get(text_col)
        time_val = row.get(time_col)

        if pd.isna(text) or pd.isna(time_val):
            continue

        emotion_scores = analyze_emotions(str(text))

        row_data = {time_col: time_val}
        for emotion in emotions:
            row_data[emotion] = emotion_scores.get(emotion, 0)

        results.append(row_data)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # Average by time period
    return result_df.groupby(time_col).mean().reset_index()


def compare_emotions_by_group(
    df: pd.DataFrame,
    group_col: str,
    text_col: str = 'transcript'
) -> pd.DataFrame:
    """
    Compare average emotional content across groups.

    Args:
        df: DataFrame with text and group columns
        group_col: Column to group by (e.g., 'party', 'name')
        text_col: Name of text column

    Returns:
        DataFrame with average emotion scores per group
    """
    if df.empty or text_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    results = []

    for group, group_df in df.groupby(group_col):
        group_emotions = {emotion: [] for emotion in EMOTION_CATEGORIES}
        group_emotions['intensity'] = []

        for _, row in group_df.iterrows():
            text = row.get(text_col)
            if pd.isna(text):
                continue

            scores = analyze_emotions(str(text))
            intensity = get_emotional_intensity(str(text))

            for emotion in EMOTION_CATEGORIES:
                group_emotions[emotion].append(scores.get(emotion, 0))
            group_emotions['intensity'].append(intensity)

        # Calculate averages
        row_data = {group_col: group, 'speech_count': len(group_df)}
        for emotion in EMOTION_CATEGORIES + ['intensity']:
            vals = group_emotions[emotion]
            row_data[f'{emotion}_avg'] = round(np.mean(vals), 3) if vals else 0

        results.append(row_data)

    return pd.DataFrame(results)


def get_emotion_summary(texts: List[str]) -> Dict:
    """
    Get summary statistics for emotions across texts.
    """
    all_scores = {emotion: [] for emotion in EMOTION_CATEGORIES}
    intensities = []

    for text in texts:
        if not text:
            continue
        scores = analyze_emotions(str(text))
        intensity = get_emotional_intensity(str(text))

        for emotion in EMOTION_CATEGORIES:
            all_scores[emotion].append(scores.get(emotion, 0))
        intensities.append(intensity)

    summary = {}
    for emotion in EMOTION_CATEGORIES:
        vals = all_scores[emotion]
        if vals:
            summary[emotion] = {
                'mean': round(np.mean(vals), 3),
                'std': round(np.std(vals), 3),
                'max': round(max(vals), 3),
                'min': round(min(vals), 3),
            }
        else:
            summary[emotion] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

    summary['intensity'] = {
        'mean': round(np.mean(intensities), 3) if intensities else 0,
        'std': round(np.std(intensities), 3) if intensities else 0,
    }

    return summary
