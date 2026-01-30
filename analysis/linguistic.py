"""
Linguistic analysis functions for Presidential Speech Dashboard.
Provides pronoun analysis, sentence length, question ratios, and tense analysis.
"""

import re
from collections import Counter
from typing import Dict, List

from utils.constants import (
    FIRST_PERSON_SINGULAR,
    FIRST_PERSON_PLURAL,
    SECOND_PERSON,
    THIRD_PERSON,
    MAX_TEXT_LENGTH,
)


def validate_text(text: str) -> str:
    """Validate and clean text input."""
    if not text or not isinstance(text, str):
        return ""
    if MAX_TEXT_LENGTH is not None and len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH]
    return text


def analyze_pronouns(text: str) -> Dict:
    """
    Analyze pronoun usage in text.

    Returns counts and rates for first person singular (I/me),
    first person plural (we/us), second person, and third person.
    """
    text = validate_text(text)
    if not text:
        return {
            "first_singular_count": 0,
            "first_plural_count": 0,
            "second_count": 0,
            "third_count": 0,
            "total_pronouns": 0,
            "first_singular_rate": 0.0,
            "first_plural_rate": 0.0,
            "i_we_ratio": 0.0,
        }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_counts = Counter(words)

    first_singular = sum(word_counts.get(p, 0) for p in FIRST_PERSON_SINGULAR)
    first_plural = sum(word_counts.get(p, 0) for p in FIRST_PERSON_PLURAL)
    second = sum(word_counts.get(p, 0) for p in SECOND_PERSON)
    third = sum(word_counts.get(p, 0) for p in THIRD_PERSON)

    total = first_singular + first_plural + second + third
    total_words = len(words)

    return {
        "first_singular_count": first_singular,
        "first_plural_count": first_plural,
        "second_count": second,
        "third_count": third,
        "total_pronouns": total,
        "first_singular_rate": round(first_singular / total_words * 100, 2) if total_words > 0 else 0.0,
        "first_plural_rate": round(first_plural / total_words * 100, 2) if total_words > 0 else 0.0,
        "i_we_ratio": round(first_singular / first_plural, 2) if first_plural > 0 else float(first_singular),
    }


def analyze_sentence_lengths(text: str) -> Dict:
    """
    Analyze sentence length distribution in text.

    Returns mean, median, std deviation, and distribution data.
    """
    text = validate_text(text)
    if not text:
        return {
            "mean_length": 0.0,
            "median_length": 0.0,
            "std_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "sentence_count": 0,
            "lengths": [],
        }

    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {
            "mean_length": 0.0,
            "median_length": 0.0,
            "std_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "sentence_count": 0,
            "lengths": [],
        }

    # Count words per sentence
    lengths = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]

    if not lengths:
        return {
            "mean_length": 0.0,
            "median_length": 0.0,
            "std_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "sentence_count": 0,
            "lengths": [],
        }

    import numpy as np
    lengths_arr = np.array(lengths)

    return {
        "mean_length": round(float(np.mean(lengths_arr)), 1),
        "median_length": round(float(np.median(lengths_arr)), 1),
        "std_length": round(float(np.std(lengths_arr)), 1),
        "min_length": int(np.min(lengths_arr)),
        "max_length": int(np.max(lengths_arr)),
        "sentence_count": len(lengths),
        "lengths": lengths,
    }


def analyze_question_ratio(text: str) -> Dict:
    """
    Analyze the ratio of questions to statements.

    Returns question count, statement count, and ratio.
    """
    text = validate_text(text)
    if not text:
        return {
            "question_count": 0,
            "statement_count": 0,
            "exclamation_count": 0,
            "question_ratio": 0.0,
        }

    questions = len(re.findall(r'\?', text))
    statements = len(re.findall(r'\.', text))
    exclamations = len(re.findall(r'!', text))

    total = questions + statements + exclamations

    return {
        "question_count": questions,
        "statement_count": statements,
        "exclamation_count": exclamations,
        "question_ratio": round(questions / total * 100, 2) if total > 0 else 0.0,
    }


def analyze_tense(text: str) -> Dict:
    """
    Analyze verb tense distribution using NLTK POS tagging.

    Returns past, present, and future tense counts and ratios.
    """
    text = validate_text(text)
    if not text:
        return {
            "past_count": 0,
            "present_count": 0,
            "future_count": 0,
            "past_ratio": 0.0,
            "present_ratio": 0.0,
            "future_ratio": 0.0,
        }

    try:
        import nltk
        from nltk import pos_tag, word_tokenize

        # Tokenize and POS tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        past_count = 0
        present_count = 0
        future_count = 0

        # Track if we see "will" or "shall" for future tense
        prev_word = ""
        for word, tag in tagged:
            word_lower = word.lower()

            # Check for future tense (will/shall + verb)
            if prev_word in ("will", "shall", "going") and tag.startswith("VB"):
                future_count += 1
            elif tag in ("VBD", "VBN"):  # Past tense or past participle
                past_count += 1
            elif tag in ("VB", "VBG", "VBP", "VBZ"):  # Present forms
                present_count += 1

            prev_word = word_lower

        total = past_count + present_count + future_count

        return {
            "past_count": past_count,
            "present_count": present_count,
            "future_count": future_count,
            "past_ratio": round(past_count / total * 100, 2) if total > 0 else 0.0,
            "present_ratio": round(present_count / total * 100, 2) if total > 0 else 0.0,
            "future_ratio": round(future_count / total * 100, 2) if total > 0 else 0.0,
        }
    except Exception:
        return _analyze_tense_fallback(text)


def _analyze_tense_fallback(text: str) -> Dict:
    """Fallback tense analysis using regex patterns."""
    # Simple pattern matching for common tense indicators
    words = text.lower().split()

    # Past tense indicators (words ending in -ed)
    past_patterns = len(re.findall(r'\b\w+ed\b', text.lower()))

    # Future tense indicators
    future_patterns = len(re.findall(r'\b(will|shall|going to)\b', text.lower()))

    # Estimate present as remainder
    total_verbs = max(1, past_patterns + future_patterns + 10)  # Rough estimate
    present_patterns = total_verbs - past_patterns - future_patterns

    total = past_patterns + present_patterns + future_patterns

    return {
        "past_count": past_patterns,
        "present_count": present_patterns,
        "future_count": future_patterns,
        "past_ratio": round(past_patterns / total * 100, 2) if total > 0 else 0.0,
        "present_ratio": round(present_patterns / total * 100, 2) if total > 0 else 0.0,
        "future_ratio": round(future_patterns / total * 100, 2) if total > 0 else 0.0,
    }


def get_linguistic_summary(text: str) -> Dict:
    """Get complete linguistic analysis summary."""
    return {
        "pronouns": analyze_pronouns(text),
        "sentences": analyze_sentence_lengths(text),
        "questions": analyze_question_ratio(text),
        "tense": analyze_tense(text),
    }
