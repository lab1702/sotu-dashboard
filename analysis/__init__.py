"""
Analysis modules for Presidential Speech Dashboard.
Provides NLP analysis functions for linguistic patterns, topics, entities, emotions, and search.
"""

import nltk
import warnings

# Suppress warnings during initialization
warnings.filterwarnings("ignore", category=UserWarning)

# Download required NLTK data on import
def _ensure_nltk_data():
    """Download required NLTK resources if not present."""
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
        ('corpora/words', 'words'),
    ]
    for path, resource in required:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

_ensure_nltk_data()
