import glob
import os
import warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import textstat
from taipy import Config
from taipy.gui import Gui, Markdown
from textblob import TextBlob

# Suppress ResourceWarning from cmudict
warnings.filterwarnings("ignore", category=ResourceWarning, module="cmudict")

# Constants
MAX_TEXT_LENGTH = 5000
MTLD_THRESHOLD = 0.72
MATTR_WINDOW_SIZE = 100
SENTIMENT_POSITIVE_THRESHOLD = 0.1
SENTIMENT_NEGATIVE_THRESHOLD = -0.1
MTLD_MIN_LENGTH = 10


def validate_text_input(text):
    """Validate and clean text input for analysis."""
    if pd.isna(text) or text == "":
        return ""
    return (
        str(text)[:MAX_TEXT_LENGTH] if len(str(text)) > MAX_TEXT_LENGTH else str(text)
    )


def validate_dataframe_column(df, column_name):
    """Check if a column exists in the dataframe."""
    return column_name in df.columns and not df[column_name].empty


# Data loading and querying functions
def calculate_sentiment(text):
    """Calculate sentiment polarity and subjectivity for a text.

    Args:
        text: Input text string to analyze

    Returns:
        tuple: (polarity: float, subjectivity: float)
    """
    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0

        blob = TextBlob(text_sample)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except (AttributeError, TypeError, ValueError) as e:
        return 0.0, 0.0


def calculate_readability(text):
    """Calculate readability scores for a text.

    Args:
        text: Input text string to analyze

    Returns:
        tuple: (flesch_reading_ease, flesch_kincaid_grade, gunning_fog, coleman_liau)
    """
    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0, 0.0, 0.0

        # Calculate various readability scores
        flesch_reading_ease = textstat.flesch_reading_ease(text_sample)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text_sample)
        gunning_fog = textstat.gunning_fog(text_sample)
        coleman_liau = textstat.coleman_liau_index(text_sample)

        return flesch_reading_ease, flesch_kincaid_grade, gunning_fog, coleman_liau
    except (AttributeError, TypeError, ValueError) as e:
        return 0.0, 0.0, 0.0, 0.0


def calculate_lexical_diversity(text):
    """Calculate lexical diversity scores for a text.

    Args:
        text: Input text string to analyze

    Returns:
        tuple: (ttr, mattr, mtld) - Type-Token Ratio, Moving Average TTR, Measure of Textual Lexical Diversity
    """
    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0, 0.0

        # Clean and tokenize text
        import re

        # Remove punctuation and convert to lowercase
        words = re.findall(r"\b[a-zA-Z]+\b", text_sample.lower())

        if len(words) == 0:
            return 0.0, 0.0, 0.0

        # Calculate metrics
        total_words = len(words)
        unique_words = len(set(words))

        # Type-Token Ratio (TTR) - basic lexical diversity
        ttr = unique_words / total_words if total_words > 0 else 0.0

        # Moving Average Type-Token Ratio (MATTR) - more stable for longer texts
        # Calculate TTR for moving windows
        window_size = min(MATTR_WINDOW_SIZE, total_words)
        if total_words >= window_size:
            ttrs = []
            for i in range(total_words - window_size + 1):
                window_words = words[i : i + window_size]
                window_unique = len(set(window_words))
                ttrs.append(window_unique / window_size)
            mattr = sum(ttrs) / len(ttrs) if ttrs else 0.0
        else:
            mattr = ttr

        # Measure of Textual Lexical Diversity (MTLD)
        # Calculate forward and backward MTLD and take average
        def calculate_mtld_direction(word_list):
            if len(word_list) < MTLD_MIN_LENGTH:
                return len(word_list)

            factors = 0
            start = 0

            while start < len(word_list):
                unique_in_segment = set()
                for i in range(start, len(word_list)):
                    unique_in_segment.add(word_list[i])
                    current_ttr = len(unique_in_segment) / (i - start + 1)

                    if current_ttr <= MTLD_THRESHOLD:
                        factors += 1
                        start = i + 1
                        break
                else:
                    # Reached end without hitting threshold
                    remaining_length = len(word_list) - start
                    if remaining_length > 0:
                        remaining_ttr = len(set(word_list[start:])) / remaining_length
                        factors += remaining_ttr / MTLD_THRESHOLD
                    break

            return len(word_list) / factors if factors > 0 else len(word_list)

        mtld_forward = calculate_mtld_direction(words)
        mtld_backward = calculate_mtld_direction(words[::-1])
        mtld = (mtld_forward + mtld_backward) / 2

        return round(ttr, 4), round(mattr, 4), round(mtld, 2)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as e:
        return 0.0, 0.0, 0.0


def load_speech_data():
    """Load presidential speech data using DuckDB.

    Returns:
        pd.DataFrame: Processed speech data with analysis columns
    """
    conn = duckdb.connect(":memory:")

    # Use the exact query pattern from DATA.md
    query = "SELECT unnest(COLUMNS(*)) FROM 'presidential_speeches/[0-9]*.json'"
    try:
        result = conn.execute(query)
        df = result.df()

        # Convert date field to just date (remove time component)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            # Replace NaT values with empty string
            df["date"] = df["date"].fillna("Unknown")

        # Convert nullable integers to regular integers
        for col in df.columns:
            if df[col].dtype == "Int64":
                df[col] = df[col].fillna(0).astype(int)

        # Convert boolean columns to strings for better Taipy compatibility
        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(str)

        # Add sentiment analysis
        print("Calculating sentiment scores...")
        if "transcript" in df.columns:
            sentiment_data = df["transcript"].apply(calculate_sentiment)
            df["sentiment_polarity"] = sentiment_data.apply(lambda x: round(x[0], 3))
            df["sentiment_subjectivity"] = sentiment_data.apply(
                lambda x: round(x[1], 3)
            )

            # Add sentiment category
            df["sentiment_category"] = df["sentiment_polarity"].apply(
                lambda x: (
                    "Positive"
                    if x > SENTIMENT_POSITIVE_THRESHOLD
                    else "Negative" if x < SENTIMENT_NEGATIVE_THRESHOLD else "Neutral"
                )
            )

        # Add readability analysis
        print("Calculating readability scores...")
        if "transcript" in df.columns:
            readability_data = df["transcript"].apply(calculate_readability)
            df["flesch_reading_ease"] = readability_data.apply(lambda x: round(x[0], 1))
            df["flesch_kincaid_grade"] = readability_data.apply(
                lambda x: round(x[1], 1)
            )
            df["gunning_fog"] = readability_data.apply(lambda x: round(x[2], 1))
            df["coleman_liau"] = readability_data.apply(lambda x: round(x[3], 1))

            # Add readability category based on Flesch Reading Ease
            df["readability_category"] = df["flesch_reading_ease"].apply(
                lambda x: (
                    "Very Easy"
                    if x >= 90
                    else (
                        "Easy"
                        if x >= 80
                        else (
                            "Fairly Easy"
                            if x >= 70
                            else (
                                "Standard"
                                if x >= 60
                                else (
                                    "Fairly Difficult"
                                    if x >= 50
                                    else "Difficult" if x >= 30 else "Very Difficult"
                                )
                            )
                        )
                    )
                )
            )

        # Add lexical diversity analysis
        print("Calculating lexical diversity scores...")
        if "transcript" in df.columns:
            lexical_data = df["transcript"].apply(calculate_lexical_diversity)
            df["ttr"] = lexical_data.apply(lambda x: x[0])  # Type-Token Ratio
            df["mattr"] = lexical_data.apply(lambda x: x[1])  # Moving Average TTR
            df["mtld"] = lexical_data.apply(
                lambda x: x[2]
            )  # Measure of Textual Lexical Diversity

            # Add lexical diversity category based on TTR
            df["lexical_diversity_category"] = df["ttr"].apply(
                lambda x: (
                    "Very High"
                    if x >= 0.8
                    else (
                        "High"
                        if x >= 0.6
                        else (
                            "Moderate"
                            if x >= 0.4
                            else "Low" if x >= 0.3 else "Very Low"
                        )
                    )
                )
            )

        # Sort by date chronologically for proper time series visualization
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        conn.close()
        return pd.DataFrame()


def filter_by_president(df, president_name=None):
    """Filter speeches by president.

    Args:
        df: DataFrame to filter
        president_name: Name of president to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not president_name or president_name == "All":
        return df
    return (
        df[df["name"] == president_name]
        if validate_dataframe_column(df, "name")
        else df
    )


def filter_by_decade(df, decade=None):
    """Filter speeches by decade.

    Args:
        df: DataFrame to filter
        decade: Decade string to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not decade or decade == "All":
        return df
    return df[df["decade"] == decade] if "decade" in df.columns else df


def filter_by_speech_type(df, speech_type=None):
    """Filter speeches by speech type.

    Args:
        df: DataFrame to filter
        speech_type: Speech type to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not speech_type or speech_type == "All":
        return df
    return df[df["speech_type"] == speech_type] if "speech_type" in df.columns else df


def filter_by_party(df, party=None):
    """Filter speeches by party.

    Args:
        df: DataFrame to filter
        party: Political party to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not party or party == "All":
        return df
    return df[df["party"] == party] if "party" in df.columns else df


def filter_by_sentiment(df, sentiment=None):
    """Filter speeches by sentiment category.

    Args:
        df: DataFrame to filter
        sentiment: Sentiment category to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not sentiment or sentiment == "All":
        return df
    return (
        df[df["sentiment_category"] == sentiment]
        if "sentiment_category" in df.columns
        else df
    )


def filter_by_readability(df, readability=None):
    """Filter speeches by readability category.

    Args:
        df: DataFrame to filter
        readability: Readability category to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not readability or readability == "All":
        return df
    return (
        df[df["readability_category"] == readability]
        if "readability_category" in df.columns
        else df
    )


def filter_by_lexical_diversity(df, lexical_diversity=None):
    """Filter speeches by lexical diversity category.

    Args:
        df: DataFrame to filter
        lexical_diversity: Lexical diversity category to filter by

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df.empty or not lexical_diversity or lexical_diversity == "All":
        return df
    return (
        df[df["lexical_diversity_category"] == lexical_diversity]
        if "lexical_diversity_category" in df.columns
        else df
    )


def apply_all_filters(
    df, president, decade, speech_type, party, sentiment, readability, lexical_diversity
):
    """Apply all filters to the dataframe.

    Args:
        df: Source dataframe
        president: President filter value
        decade: Decade filter value
        speech_type: Speech type filter value
        party: Party filter value
        sentiment: Sentiment filter value
        readability: Readability filter value
        lexical_diversity: Lexical diversity filter value

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    filtered = df.copy()
    filtered = filter_by_president(filtered, president)
    filtered = filter_by_decade(filtered, decade)
    filtered = filter_by_speech_type(filtered, speech_type)
    filtered = filter_by_party(filtered, party)
    filtered = filter_by_sentiment(filtered, sentiment)
    filtered = filter_by_readability(filtered, readability)
    filtered = filter_by_lexical_diversity(filtered, lexical_diversity)
    return filtered


def create_speech_type_summary(df):
    """Create a summary dataframe for speech types chart.

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Summary with speech_type and count columns
    """
    if df.empty or "speech_type" not in df.columns:
        return pd.DataFrame({"speech_type": [], "count": []})

    summary = df["speech_type"].value_counts().reset_index()
    summary.columns = ["speech_type", "count"]
    return summary


def calculate_statistics(df):
    """Calculate statistics from dataframe.

    Args:
        df: Input dataframe

    Returns:
        tuple: (total_speeches, date_range, avg_word_count, presidents_count)
    """
    if df.empty:
        return 0, "N/A", 0, 0

    total = len(df)

    # Calculate date range safely
    if "date" in df.columns and not df["date"].isna().all():
        valid_dates = df["date"][df["date"] != "Unknown"]
        if len(valid_dates) > 0:
            date_range = f"{valid_dates.min()} to {valid_dates.max()}"
        else:
            date_range = "N/A"
    else:
        date_range = "N/A"

    # Calculate average word count safely
    if "word_count" in df.columns and not df["word_count"].isna().all():
        avg_words = int(df["word_count"].mean())
    else:
        avg_words = 0

    # Calculate president count safely
    if "name" in df.columns:
        presidents = df["name"].nunique()
    else:
        presidents = 0

    return total, date_range, avg_words, presidents


# Initialize data
print("Loading presidential speech data...")
speech_data = load_speech_data()
print(f"Loaded {len(speech_data)} speeches")

# Get unique values for dropdowns
presidents = (
    ["All"] + sorted(speech_data["name"].unique().tolist())
    if "name" in speech_data.columns
    else ["All"]
)
decades = (
    ["All"] + sorted(speech_data["decade"].unique().tolist())
    if "decade" in speech_data.columns
    else ["All"]
)
speech_types = (
    ["All"] + sorted(speech_data["speech_type"].unique().tolist())
    if "speech_type" in speech_data.columns
    else ["All"]
)
parties = (
    ["All"] + sorted([p for p in speech_data["party"].unique().tolist() if p])
    if "party" in speech_data.columns
    else ["All"]
)
sentiments = (
    ["All"] + sorted(speech_data["sentiment_category"].unique().tolist())
    if "sentiment_category" in speech_data.columns
    else ["All"]
)
readabilities = (
    ["All"] + sorted(speech_data["readability_category"].unique().tolist())
    if "readability_category" in speech_data.columns
    else ["All"]
)
lexical_diversities = (
    ["All"] + sorted(speech_data["lexical_diversity_category"].unique().tolist())
    if "lexical_diversity_category" in speech_data.columns
    else ["All"]
)

# Dashboard state variables
selected_president = "All"
selected_decade = "All"
selected_speech_type = "All"
selected_party = "All"
selected_sentiment = "All"
selected_readability = "All"
selected_lexical_diversity = "All"
filtered_data = speech_data.copy()

# Calculate initial statistics
total_speeches, date_range, avg_word_count, presidents_count = calculate_statistics(
    filtered_data
)
speech_type_summary = create_speech_type_summary(filtered_data)


def update_dashboard_state(state):
    """Update dashboard state after filter changes.

    Args:
        state: Taipy state object containing filter selections
    """
    state.filtered_data = apply_all_filters(
        speech_data,
        state.selected_president,
        state.selected_decade,
        state.selected_speech_type,
        state.selected_party,
        state.selected_sentiment,
        state.selected_readability,
        state.selected_lexical_diversity,
    )
    (
        state.total_speeches,
        state.date_range,
        state.avg_word_count,
        state.presidents_count,
    ) = calculate_statistics(state.filtered_data)
    state.speech_type_summary = create_speech_type_summary(state.filtered_data)


# Event handlers
def on_president_change(state):
    """Handle president selection change."""
    update_dashboard_state(state)


def on_decade_change(state):
    """Handle decade selection change."""
    update_dashboard_state(state)


def on_speech_type_change(state):
    """Handle speech type selection change."""
    update_dashboard_state(state)


def on_party_change(state):
    """Handle party selection change."""
    update_dashboard_state(state)


def on_sentiment_change(state):
    """Handle sentiment selection change."""
    update_dashboard_state(state)


def on_readability_change(state):
    """Handle readability selection change."""
    update_dashboard_state(state)


def on_lexical_diversity_change(state):
    """Handle lexical diversity selection change."""
    update_dashboard_state(state)


# Dashboard layout

page = """
# Presidential Speech Analysis Dashboard

<|layout|columns=300px 1fr|
<|part|class_name=sidebar|
## Filters

**President**
<|{selected_president}|selector|lov={presidents}|dropdown|on_change=on_president_change|>

**Decade** 
<|{selected_decade}|selector|lov={decades}|dropdown|on_change=on_decade_change|>

**Speech Type**
<|{selected_speech_type}|selector|lov={speech_types}|dropdown|on_change=on_speech_type_change|>

**Party**
<|{selected_party}|selector|lov={parties}|dropdown|on_change=on_party_change|>

**Sentiment**
<|{selected_sentiment}|selector|lov={sentiments}|dropdown|on_change=on_sentiment_change|>

**Readability**
<|{selected_readability}|selector|lov={readabilities}|dropdown|on_change=on_readability_change|>

**Lexical Diversity**
<|{selected_lexical_diversity}|selector|lov={lexical_diversities}|dropdown|on_change=on_lexical_diversity_change|>

---

## Statistics
**Total Speeches**: <|{total_speeches}|text|>

**Date Range**: <|{date_range}|text|>

**Avg Word Count**: <|{avg_word_count}|text|>

**Presidents**: <|{presidents_count}|text|>
|>

<|part|class_name=main|
## Speech Data
<|{filtered_data}|table|page_size=15|columns=year;name;title;speech_type;word_count;party;sentiment_category;readability_category;lexical_diversity_category;ttr|>

## Word Count Over Time
<|{filtered_data}|chart|x=date|y=word_count|type=scatter|title=Speech Word Count Over Time|height=400px|>

## Sentiment Over Time
<|{filtered_data}|chart|x=date|y=sentiment_polarity|type=scatter|title=Speech Sentiment Over Time|height=400px|>

## Readability Over Time
<|{filtered_data}|chart|x=date|y=flesch_reading_ease|type=scatter|title=Speech Readability Over Time (Flesch Score)|height=400px|>

## Lexical Diversity Over Time
<|{filtered_data}|chart|x=date|y=ttr|type=scatter|title=Speech Lexical Diversity Over Time (TTR)|height=400px|>

## Speech Types Distribution  
<|{filtered_data}|chart|type=histogram|x=speech_type|title=Speech Types Distribution|height=400px|>

## Sentiment Distribution  
<|{filtered_data}|chart|type=histogram|x=sentiment_category|title=Sentiment Distribution|height=400px|>

## Readability Distribution  
<|{filtered_data}|chart|type=histogram|x=readability_category|title=Readability Distribution|height=400px|>

## Lexical Diversity Distribution  
<|{filtered_data}|chart|type=histogram|x=lexical_diversity_category|title=Lexical Diversity Distribution|height=400px|>
|>
|>
"""

# Create the Taipy GUI instance with production configuration
gui = Gui(page)

# Configure Taipy for production
from taipy import Config

Config.configure_data_node(
    "default", storage_type="memory"
)  # Use memory storage for better performance

if __name__ == "__main__":
    # Production settings for Taipy
    gui.run(
        debug=False,
        port=5000,
        host="0.0.0.0",
        use_reloader=False,  # Disable auto-reloader for production
        allow_unsafe_werkzeug=True,  # Allow Werkzeug in production (needed for Docker)
        threaded=True,  # Enable threading for better performance
        run_server=True,  # Explicitly run the server
        watermark="",  # Remove Taipy watermark for cleaner UI
    )
