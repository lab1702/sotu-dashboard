"""
Word Analysis tab view for Presidential Speech Dashboard.
Displays word clouds and n-grams.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from analysis.topics import (
    generate_wordcloud,
    extract_ngrams,
    get_top_words,
    keyword_frequency_over_time,
)
from utils.constants import COLORBLIND_PALETTE, DEFAULT_KEYWORDS
from utils.loading import loading_indicator, StatusIndicator


@st.cache_data
def _cached_wordcloud(texts_tuple, max_words):
    """Cache word cloud generation."""
    return generate_wordcloud(list(texts_tuple), max_words)


@st.cache_data
def _cached_ngrams(texts_tuple, n, top_k):
    """Cache n-gram extraction."""
    return extract_ngrams(list(texts_tuple), n, top_k)


@st.cache_data
def _cached_keyword_trends(df_hash, keywords_tuple, df, time_col):
    """Cache keyword frequency computation."""
    return keyword_frequency_over_time(df, list(keywords_tuple), 'transcript', time_col)


def render_word_analysis_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Word Analysis tab content."""
    st.subheader("Word Analysis")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No transcript data available for analysis.")
        return

    # Get texts
    texts = filtered_df['transcript'].dropna().tolist()
    if not texts:
        st.warning("No text data available for word analysis.")
        return

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Word Cloud")

        max_words = st.slider("Maximum words", 50, 200, 100, key="wc_max_words")

        # Generate word cloud
        with loading_indicator("Generating word cloud..."):
            wc = _cached_wordcloud(tuple(texts), max_words)

        if wc is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Could not generate word cloud. Try selecting more speeches.")

    with col2:
        st.markdown("### Top Words")

        top_words = get_top_words(texts, top_k=30)
        if top_words:
            words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
            fig = px.bar(
                words_df.head(20),
                x='Count',
                y='Word',
                orientation='h',
                title="Top 20 Words",
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No word frequency data available.")

    # N-grams section
    st.markdown("---")
    st.markdown("### N-gram Analysis")

    ngram_col1, ngram_col2 = st.columns(2)

    with ngram_col1:
        st.markdown("#### Bigrams (2-word phrases)")
        with loading_indicator("Extracting bigrams..."):
            bigrams = _cached_ngrams(tuple(texts), 2, 20)

        if bigrams:
            bigram_df = pd.DataFrame(bigrams, columns=['Phrase', 'Count'])
            fig = px.bar(
                bigram_df,
                x='Count',
                y='Phrase',
                orientation='h',
                color_discrete_sequence=[COLORBLIND_PALETTE[0]],
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough text data for bigram analysis.")

    with ngram_col2:
        st.markdown("#### Trigrams (3-word phrases)")
        with loading_indicator("Extracting trigrams..."):
            trigrams = _cached_ngrams(tuple(texts), 3, 20)

        if trigrams:
            trigram_df = pd.DataFrame(trigrams, columns=['Phrase', 'Count'])
            fig = px.bar(
                trigram_df,
                x='Count',
                y='Phrase',
                orientation='h',
                color_discrete_sequence=[COLORBLIND_PALETTE[1]],
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough text data for trigram analysis.")

    # Keyword trends section
    st.markdown("---")
    st.markdown("### Keyword Trends Over Time")

    if 'year' not in filtered_df.columns:
        st.info("Year data not available for trend analysis.")
        return

    # Keyword selection
    available_keywords = DEFAULT_KEYWORDS.copy()
    selected_keywords = st.multiselect(
        "Select keywords to track",
        available_keywords,
        default=['economy', 'war', 'peace', 'freedom'],
        key="keyword_select"
    )

    custom_keyword = st.text_input("Add custom keyword", key="custom_keyword")
    if custom_keyword and custom_keyword.lower() not in [k.lower() for k in selected_keywords]:
        selected_keywords.append(custom_keyword.lower())

    if selected_keywords:
        with loading_indicator("Computing keyword trends..."):
            df_hash = hash(tuple(filtered_df.index.tolist()))
            trends_df = _cached_keyword_trends(
                df_hash,
                tuple(selected_keywords),
                filtered_df,
                'year'
            )

        if not trends_df.empty:
            # Melt for plotly
            melted = trends_df.melt(
                id_vars=['year'],
                value_vars=selected_keywords,
                var_name='Keyword',
                value_name='Frequency'
            )

            fig = px.line(
                melted,
                x='year',
                y='Frequency',
                color='Keyword',
                title="Keyword Frequency Over Time",
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Total Mentions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available for selected keywords.")
    else:
        st.info("Select keywords to see trends over time.")
