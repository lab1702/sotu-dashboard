"""
Presidential Speech Analysis Dashboard

A Streamlit dashboard for analyzing presidential speeches with NLP features
including sentiment analysis, readability metrics, topic modeling, emotion
detection, named entity recognition, and more.
"""

import warnings
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import textstat
from textblob import TextBlob

# Suppress ResourceWarning from cmudict
warnings.filterwarnings("ignore", category=ResourceWarning, module="cmudict")
warnings.filterwarnings("ignore", category=UserWarning)

# Import constants
from utils.constants import (
    MAX_TEXT_LENGTH,
    MTLD_THRESHOLD,
    MATTR_WINDOW_SIZE,
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
    MTLD_MIN_LENGTH,
    COLORBLIND_PALETTE,
    READABILITY_ORDER,
    SENTIMENT_ORDER,
    LEXICAL_ORDER,
)

# Import view modules
from views.word_analysis import render_word_analysis_tab
from views.topics import render_topics_tab
from views.linguistic import render_linguistic_tab
from views.emotions import render_emotions_tab
from views.entities import render_entities_tab
from views.comparisons import render_comparisons_tab
from views.search import render_search_tab


def validate_text_input(text):
    """Validate and clean text input for analysis."""
    if pd.isna(text) or text == "":
        return ""
    text_str = str(text)
    if MAX_TEXT_LENGTH is not None and len(text_str) > MAX_TEXT_LENGTH:
        return text_str[:MAX_TEXT_LENGTH]
    return text_str


def calculate_sentiment(text):
    """Calculate sentiment polarity and subjectivity for a text."""
    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0
        blob = TextBlob(text_sample)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except (AttributeError, TypeError, ValueError):
        return 0.0, 0.0


def calculate_readability(text):
    """Calculate readability scores for a text."""
    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0, 0.0, 0.0
        flesch_reading_ease = textstat.flesch_reading_ease(text_sample)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text_sample)
        gunning_fog = textstat.gunning_fog(text_sample)
        coleman_liau = textstat.coleman_liau_index(text_sample)
        return flesch_reading_ease, flesch_kincaid_grade, gunning_fog, coleman_liau
    except (AttributeError, TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0


def calculate_lexical_diversity(text):
    """Calculate lexical diversity scores for a text."""
    import re

    try:
        text_sample = validate_text_input(text)
        if not text_sample:
            return 0.0, 0.0, 0.0

        words = re.findall(r"\b[a-zA-Z]+\b", text_sample.lower())
        if len(words) == 0:
            return 0.0, 0.0, 0.0

        total_words = len(words)
        unique_words = len(set(words))

        # Type-Token Ratio (TTR)
        ttr = unique_words / total_words if total_words > 0 else 0.0

        # Moving Average Type-Token Ratio (MATTR)
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
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 0.0, 0.0, 0.0


@st.cache_data
def load_speech_data():
    """Load presidential speech data using DuckDB."""
    conn = duckdb.connect(":memory:")
    query = "SELECT * FROM 'presidential_speeches/[0-9]*.json'"
    try:
        result = conn.execute(query)
        df = result.df()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            df["date"] = df["date"].fillna("Unknown")

        for col in df.columns:
            if df[col].dtype == "Int64":
                df[col] = df[col].fillna(0).astype(int)

        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(str)

        # Add sentiment analysis
        if "transcript" in df.columns:
            sentiment_data = df["transcript"].apply(calculate_sentiment)
            df["sentiment_polarity"] = sentiment_data.apply(lambda x: round(x[0], 3))
            df["sentiment_subjectivity"] = sentiment_data.apply(
                lambda x: round(x[1], 3)
            )
            df["sentiment_category"] = df["sentiment_polarity"].apply(
                lambda x: (
                    "Positive"
                    if x > SENTIMENT_POSITIVE_THRESHOLD
                    else "Negative" if x < SENTIMENT_NEGATIVE_THRESHOLD else "Neutral"
                )
            )

        # Add readability analysis
        if "transcript" in df.columns:
            readability_data = df["transcript"].apply(calculate_readability)
            df["flesch_reading_ease"] = readability_data.apply(lambda x: round(x[0], 1))
            df["flesch_kincaid_grade"] = readability_data.apply(
                lambda x: round(x[1], 1)
            )
            df["gunning_fog"] = readability_data.apply(lambda x: round(x[2], 1))
            df["coleman_liau"] = readability_data.apply(lambda x: round(x[3], 1))
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
        if "transcript" in df.columns:
            lexical_data = df["transcript"].apply(calculate_lexical_diversity)
            df["ttr"] = lexical_data.apply(lambda x: x[0])
            df["mattr"] = lexical_data.apply(lambda x: x[1])
            df["mtld"] = lexical_data.apply(lambda x: x[2])
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

        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        conn.close()
        return pd.DataFrame()


def apply_filters(df, presidents, decade, speech_type, party, sentiment, readability, lexical_diversity):
    """Apply all filters to the dataframe."""
    filtered = df.copy()
    if presidents and "name" in filtered.columns:
        filtered = filtered[filtered["name"].isin(presidents)]
    if decade != "All" and "decade" in filtered.columns:
        filtered = filtered[filtered["decade"] == decade]
    if speech_type != "All" and "speech_type" in filtered.columns:
        filtered = filtered[filtered["speech_type"] == speech_type]
    if party != "All" and "party" in filtered.columns:
        filtered = filtered[filtered["party"] == party]
    if sentiment != "All" and "sentiment_category" in filtered.columns:
        filtered = filtered[filtered["sentiment_category"] == sentiment]
    if readability != "All" and "readability_category" in filtered.columns:
        filtered = filtered[filtered["readability_category"] == readability]
    if lexical_diversity != "All" and "lexical_diversity_category" in filtered.columns:
        filtered = filtered[filtered["lexical_diversity_category"] == lexical_diversity]
    return filtered


# Page config
st.set_page_config(
    page_title="Presidential Speech Analysis",
    page_icon="🎤",
    layout="wide"
)

st.title("Presidential Speech Analysis Dashboard")

# Load data
with st.spinner("Loading speech data..."):
    speech_data = load_speech_data()

if speech_data.empty:
    st.error("No data loaded. Please check that presidential_speeches directory exists.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

presidents = sorted(speech_data["name"].unique().tolist()) if "name" in speech_data.columns else []
selected_presidents = st.sidebar.multiselect("Presidents", presidents, placeholder="All Presidents")

decades = ["All"] + sorted(speech_data["decade"].unique().tolist()) if "decade" in speech_data.columns else ["All"]
selected_decade = st.sidebar.selectbox("Decade", decades)

speech_types = ["All"] + sorted(speech_data["speech_type"].unique().tolist()) if "speech_type" in speech_data.columns else ["All"]
selected_speech_type = st.sidebar.selectbox("Speech Type", speech_types)

parties = ["All"] + sorted([p for p in speech_data["party"].unique().tolist() if p]) if "party" in speech_data.columns else ["All"]
selected_party = st.sidebar.selectbox("Party", parties)

sentiments = ["All"] + sorted(speech_data["sentiment_category"].unique().tolist()) if "sentiment_category" in speech_data.columns else ["All"]
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

readabilities = ["All"] + sorted(speech_data["readability_category"].unique().tolist()) if "readability_category" in speech_data.columns else ["All"]
selected_readability = st.sidebar.selectbox("Readability", readabilities)

lexical_diversities = ["All"] + sorted(speech_data["lexical_diversity_category"].unique().tolist()) if "lexical_diversity_category" in speech_data.columns else ["All"]
selected_lexical_diversity = st.sidebar.selectbox("Lexical Diversity", lexical_diversities)

# Apply filters
filtered_data = apply_filters(
    speech_data,
    selected_presidents,
    selected_decade,
    selected_speech_type,
    selected_party,
    selected_sentiment,
    selected_readability,
    selected_lexical_diversity
)

# Sidebar statistics
st.sidebar.markdown("---")
st.sidebar.header("Statistics")
st.sidebar.metric("Total Speeches", len(filtered_data))

if "date" in filtered_data.columns and len(filtered_data) > 0:
    valid_dates = filtered_data["date"][filtered_data["date"] != "Unknown"]
    if len(valid_dates) > 0:
        st.sidebar.text(f"Date Range:\n{valid_dates.min()} to\n{valid_dates.max()}")

if "word_count" in filtered_data.columns and len(filtered_data) > 0:
    st.sidebar.metric("Avg Word Count", int(filtered_data["word_count"].mean()))

if "name" in filtered_data.columns:
    st.sidebar.metric("Presidents", filtered_data["name"].nunique())

# Main content with tabs - 10 tabs as specified
tab_names = [
    "Data Table",
    "Time Trends",
    "Distributions",
    "Word Analysis",
    "Topics",
    "Linguistic",
    "Emotions",
    "Entities",
    "Comparisons",
    "Search & Export"
]

tabs = st.tabs(tab_names)

# Tab 1: Data Table
with tabs[0]:
    st.subheader("Speech Data")
    display_columns = [
        "year", "name", "title", "speech_type", "word_count", "party",
        "sentiment_category", "readability_category", "lexical_diversity_category", "ttr"
    ]
    available_columns = [col for col in display_columns if col in filtered_data.columns]
    st.dataframe(filtered_data[available_columns], use_container_width=True, height=600)

# Tab 2: Time Trends
with tabs[1]:
    col1, col2 = st.columns(2)
    chart_data = filtered_data[filtered_data["date"] != "Unknown"] if "date" in filtered_data.columns else filtered_data
    use_color = "name" in chart_data.columns and len(selected_presidents) > 0

    with col1:
        if "date" in chart_data.columns and "word_count" in chart_data.columns:
            fig = px.scatter(
                chart_data,
                x="date", y="word_count",
                color="name" if use_color else None,
                color_discrete_sequence=COLORBLIND_PALETTE,
                title="Word Count Over Time",
                hover_data=["name", "title"] if "name" in chart_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

        if "date" in chart_data.columns and "flesch_reading_ease" in chart_data.columns:
            fig = px.scatter(
                chart_data,
                x="date", y="flesch_reading_ease",
                color="name" if use_color else None,
                color_discrete_sequence=COLORBLIND_PALETTE,
                title="Readability Over Time (Flesch Score)",
                hover_data=["name", "title"] if "name" in chart_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "date" in chart_data.columns and "sentiment_polarity" in chart_data.columns:
            fig = px.scatter(
                chart_data,
                x="date", y="sentiment_polarity",
                color="name" if use_color else None,
                color_discrete_sequence=COLORBLIND_PALETTE,
                title="Sentiment Over Time",
                hover_data=["name", "title"] if "name" in chart_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

        if "date" in chart_data.columns and "ttr" in chart_data.columns:
            fig = px.scatter(
                chart_data,
                x="date", y="ttr",
                color="name" if use_color else None,
                color_discrete_sequence=COLORBLIND_PALETTE,
                title="Lexical Diversity Over Time (TTR)",
                hover_data=["name", "title"] if "name" in chart_data.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Distributions
with tabs[2]:
    col1, col2 = st.columns(2)
    use_color = "name" in filtered_data.columns and len(selected_presidents) > 0

    with col1:
        if "speech_type" in filtered_data.columns:
            speech_type_order = sorted(speech_data["speech_type"].unique().tolist())
            if use_color:
                rows = []
                for president in selected_presidents:
                    president_data = filtered_data[filtered_data["name"] == president]
                    counts = president_data["speech_type"].value_counts()
                    for cat in speech_type_order:
                        rows.append({"speech_type": cat, "count": counts.get(cat, 0), "name": president})
                speech_type_df = pd.DataFrame(rows)
                fig = px.bar(
                    speech_type_df, x="speech_type", y="count", color="name",
                    color_discrete_sequence=COLORBLIND_PALETTE,
                    title="Speech Types Distribution", barmode="group"
                )
            else:
                counts = filtered_data["speech_type"].value_counts()
                speech_type_df = pd.DataFrame({
                    "speech_type": speech_type_order,
                    "count": [counts.get(cat, 0) for cat in speech_type_order]
                })
                fig = px.bar(speech_type_df, x="speech_type", y="count", title="Speech Types Distribution")
            fig.update_layout(xaxis_title="Speech Type", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        if "readability_category" in filtered_data.columns:
            if use_color:
                rows = []
                for president in selected_presidents:
                    president_data = filtered_data[filtered_data["name"] == president]
                    counts = president_data["readability_category"].value_counts()
                    for cat in READABILITY_ORDER:
                        rows.append({"readability_category": cat, "count": counts.get(cat, 0), "name": president})
                readability_df = pd.DataFrame(rows)
                fig = px.bar(
                    readability_df, x="readability_category", y="count", color="name",
                    color_discrete_sequence=COLORBLIND_PALETTE,
                    title="Readability Distribution", barmode="group"
                )
            else:
                counts = filtered_data["readability_category"].value_counts()
                readability_df = pd.DataFrame({
                    "readability_category": READABILITY_ORDER,
                    "count": [counts.get(cat, 0) for cat in READABILITY_ORDER]
                })
                fig = px.bar(readability_df, x="readability_category", y="count", title="Readability Distribution")
            fig.update_layout(xaxis_title="Readability Category", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "sentiment_category" in filtered_data.columns:
            if use_color:
                rows = []
                for president in selected_presidents:
                    president_data = filtered_data[filtered_data["name"] == president]
                    counts = president_data["sentiment_category"].value_counts()
                    for cat in SENTIMENT_ORDER:
                        rows.append({"sentiment_category": cat, "count": counts.get(cat, 0), "name": president})
                sentiment_df = pd.DataFrame(rows)
                fig = px.bar(
                    sentiment_df, x="sentiment_category", y="count", color="name",
                    color_discrete_sequence=COLORBLIND_PALETTE,
                    title="Sentiment Distribution", barmode="group"
                )
            else:
                counts = filtered_data["sentiment_category"].value_counts()
                sentiment_df = pd.DataFrame({
                    "sentiment_category": SENTIMENT_ORDER,
                    "count": [counts.get(cat, 0) for cat in SENTIMENT_ORDER]
                })
                fig = px.bar(sentiment_df, x="sentiment_category", y="count", title="Sentiment Distribution")
            fig.update_layout(xaxis_title="Sentiment Category", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        if "lexical_diversity_category" in filtered_data.columns:
            if use_color:
                rows = []
                for president in selected_presidents:
                    president_data = filtered_data[filtered_data["name"] == president]
                    counts = president_data["lexical_diversity_category"].value_counts()
                    for cat in LEXICAL_ORDER:
                        rows.append({"lexical_diversity_category": cat, "count": counts.get(cat, 0), "name": president})
                lexical_df = pd.DataFrame(rows)
                fig = px.bar(
                    lexical_df, x="lexical_diversity_category", y="count", color="name",
                    color_discrete_sequence=COLORBLIND_PALETTE,
                    title="Lexical Diversity Distribution", barmode="group"
                )
            else:
                counts = filtered_data["lexical_diversity_category"].value_counts()
                lexical_df = pd.DataFrame({
                    "lexical_diversity_category": LEXICAL_ORDER,
                    "count": [counts.get(cat, 0) for cat in LEXICAL_ORDER]
                })
                fig = px.bar(lexical_df, x="lexical_diversity_category", y="count", title="Lexical Diversity Distribution")
            fig.update_layout(xaxis_title="Lexical Diversity Category", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: Word Analysis
with tabs[3]:
    render_word_analysis_tab(speech_data, filtered_data)

# Tab 5: Topics
with tabs[4]:
    render_topics_tab(speech_data, filtered_data)

# Tab 6: Linguistic Patterns
with tabs[5]:
    render_linguistic_tab(speech_data, filtered_data)

# Tab 7: Emotions
with tabs[6]:
    render_emotions_tab(speech_data, filtered_data)

# Tab 8: Named Entities
with tabs[7]:
    render_entities_tab(speech_data, filtered_data)

# Tab 9: Comparisons
with tabs[8]:
    render_comparisons_tab(speech_data, filtered_data)

# Tab 10: Search & Export
with tabs[9]:
    render_search_tab(speech_data, filtered_data)
