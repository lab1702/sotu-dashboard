"""
Shared constants for the Presidential Speech Dashboard.
"""

# Text processing limits
# Set to None to analyze full text without truncation
MAX_TEXT_LENGTH = None
MTLD_THRESHOLD = 0.72
MATTR_WINDOW_SIZE = 100
SENTIMENT_POSITIVE_THRESHOLD = 0.1
SENTIMENT_NEGATIVE_THRESHOLD = -0.1
MTLD_MIN_LENGTH = 10

# Colorblind-safe palette (Wong, 2011 - Nature Methods)
COLORBLIND_PALETTE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky Blue
    "#D55E00",  # Vermillion
    "#000000",  # Black
]

# Party colors
PARTY_COLORS = {
    "Democratic": "#0072B2",      # Blue
    "Republican": "#D55E00",      # Red/Vermillion
    "Democratic-Republican": "#009E73",  # Green
    "Whig": "#E69F00",            # Orange
    "Federalist": "#CC79A7",      # Pink
    "Unaffiliated": "#56B4E9",    # Sky Blue
    "National Union": "#F0E442",  # Yellow
}

# War periods for comparative analysis
WAR_PERIODS = {
    "War of 1812": (1812, 1815),
    "Mexican-American War": (1846, 1848),
    "Civil War": (1861, 1865),
    "Spanish-American War": (1898, 1898),
    "World War I": (1917, 1918),
    "World War II": (1941, 1945),
    "Korean War": (1950, 1953),
    "Vietnam War": (1964, 1975),
    "Gulf War": (1990, 1991),
    "War on Terror": (2001, 2021),
}

# Historical eras for grouping
HISTORICAL_ERAS = {
    "Early Republic": (1789, 1828),
    "Jacksonian Era": (1829, 1848),
    "Antebellum": (1849, 1860),
    "Civil War Era": (1861, 1865),
    "Reconstruction": (1866, 1877),
    "Gilded Age": (1878, 1900),
    "Progressive Era": (1901, 1920),
    "Roaring Twenties": (1921, 1929),
    "Great Depression & WWII": (1930, 1945),
    "Post-War Era": (1946, 1963),
    "Civil Rights Era": (1964, 1980),
    "Modern Era": (1981, 2000),
    "21st Century": (2001, 2030),
}

# Emotion categories from NRCLex
EMOTION_CATEGORIES = [
    "fear",
    "anger",
    "anticipation",
    "trust",
    "surprise",
    "positive",
    "negative",
    "sadness",
    "disgust",
    "joy",
]

# Named entity types to track
ENTITY_TYPES = {
    "PERSON": "People",
    "GPE": "Countries/Cities",
    "ORG": "Organizations",
    "NORP": "Nationalities/Groups",
    "EVENT": "Events",
    "LAW": "Laws/Documents",
    "DATE": "Dates",
    "LOC": "Locations",
}

# Keywords for tracking trends
DEFAULT_KEYWORDS = [
    "economy",
    "war",
    "peace",
    "freedom",
    "liberty",
    "democracy",
    "security",
    "health",
    "education",
    "justice",
    "america",
    "congress",
    "nation",
    "people",
    "government",
]

# Pronoun categories for linguistic analysis
FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs"}

# Readability category order
READABILITY_ORDER = [
    "Very Easy",
    "Easy",
    "Fairly Easy",
    "Standard",
    "Fairly Difficult",
    "Difficult",
    "Very Difficult",
]

# Sentiment category order
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

# Lexical diversity category order
LEXICAL_ORDER = ["Very High", "High", "Moderate", "Low", "Very Low"]

# Default number of topics for LDA
DEFAULT_N_TOPICS = 10

# Default word cloud settings
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400

# N-gram settings
DEFAULT_NGRAM_TOP_K = 20
