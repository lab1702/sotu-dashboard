"""
Search & Export tab view for Presidential Speech Dashboard.
Provides full-text search, similar speeches, and CSV export.
"""

import streamlit as st
import pandas as pd

from analysis.search import (
    SpeechSearchEngine,
    find_similar_speeches,
    export_to_csv,
    highlight_entities_in_text,
)
from utils.constants import COLORBLIND_PALETTE
from utils.loading import loading_indicator


def render_search_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Search & Export tab content."""
    st.subheader("Search & Export")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No data available for search.")
        return

    # Full-Text Search Section
    st.markdown("### Full-Text Search")
    st.markdown("*Search across all speeches using BM25 ranking.*")

    search_query = st.text_input(
        "Enter search query",
        placeholder="e.g., economy, war, freedom, civil rights",
        key="search_query"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        num_results = st.slider("Max results", 5, 50, 10, key="num_results")

    if search_query:
        with loading_indicator("Searching speeches..."):
            search_engine = SpeechSearchEngine(filtered_df)
            results = search_engine.search(search_query, top_k=num_results)

        if results:
            st.markdown(f"#### Found {len(results)} matching speeches")

            for i, result in enumerate(results):
                with st.expander(
                    f"{i+1}. {result['name']} - {result['title'][:60]}... ({result['year']})",
                    expanded=(i < 3)
                ):
                    st.markdown(f"**Score:** {result['score']:.2f}")
                    st.markdown(f"**Date:** {result['date']}")
                    st.markdown("**Excerpt:**")
                    st.markdown(f"*{result['excerpt']}*")

                    # Link to view full speech
                    if st.button(f"View Full Speech", key=f"view_speech_{i}"):
                        st.session_state['selected_speech_idx'] = result['index']
        else:
            st.info("No matching speeches found. Try different search terms.")

    # Similar Speeches Section
    st.markdown("---")
    st.markdown("### Find Similar Speeches")
    st.markdown("*Select a speech to find others with similar content using TF-IDF similarity.*")

    if 'title' in filtered_df.columns:
        # Create speech selector
        speech_options = []
        for idx, row in filtered_df.iterrows():
            title = row.get('title', 'Unknown')[:50]
            name = row.get('name', 'Unknown')
            year = row.get('year', 'Unknown')
            speech_options.append(f"{name} ({year}): {title}...")

        selected_idx = st.selectbox(
            "Select a reference speech",
            range(len(speech_options)),
            format_func=lambda x: speech_options[x],
            key="similar_speech_select"
        )

        if st.button("Find Similar Speeches", key="find_similar_btn"):
            with loading_indicator("Computing speech similarities..."):
                similar = find_similar_speeches(filtered_df, selected_idx, 'transcript', top_k=5)

            if similar:
                st.markdown("#### Most Similar Speeches")

                for i, sim in enumerate(similar):
                    st.markdown(
                        f"**{i+1}. {sim['name']}** - {sim['title'][:60]}... ({sim['year']})"
                    )
                    st.markdown(f"Similarity: {sim['similarity']:.3f}")
                    st.markdown("---")
            else:
                st.info("Could not compute similarities. Try with more speeches selected.")

    # Speech Viewer
    st.markdown("---")
    st.markdown("### Speech Viewer")

    if 'title' in filtered_df.columns:
        viewer_idx = st.selectbox(
            "Select a speech to view",
            range(len(filtered_df)),
            format_func=lambda x: f"{filtered_df.iloc[x].get('name', 'Unknown')} ({filtered_df.iloc[x].get('year', 'Unknown')}): {str(filtered_df.iloc[x].get('title', 'Unknown'))[:50]}...",
            key="viewer_select"
        )

        if viewer_idx is not None:
            speech_row = filtered_df.iloc[viewer_idx]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {speech_row.get('title', 'Unknown Title')}")
                st.markdown(f"**President:** {speech_row.get('name', 'Unknown')}")
                st.markdown(f"**Date:** {speech_row.get('date', 'Unknown')}")
                st.markdown(f"**Type:** {speech_row.get('speech_type', 'Unknown')}")

            with col2:
                st.markdown("**Quick Stats:**")
                if 'word_count' in speech_row:
                    st.metric("Word Count", f"{speech_row['word_count']:,}")
                if 'sentiment_polarity' in speech_row:
                    st.metric("Sentiment", f"{speech_row['sentiment_polarity']:.3f}")
                if 'flesch_reading_ease' in speech_row:
                    st.metric("Readability", f"{speech_row['flesch_reading_ease']:.1f}")

            # Display options
            show_entities = st.checkbox("Highlight named entities", value=False, key="show_entities")
            max_chars = st.slider("Text length (characters)", 500, 5000, 2000, key="text_length")

            transcript = str(speech_row.get('transcript', ''))

            if transcript:
                st.markdown("---")
                st.markdown("#### Speech Text")

                if show_entities:
                    with loading_indicator("Highlighting named entities..."):
                        highlighted_text = highlight_entities_in_text(transcript, max_chars)
                    st.markdown(highlighted_text)
                else:
                    display_text = transcript[:max_chars]
                    if len(transcript) > max_chars:
                        display_text += "..."
                    st.markdown(display_text)
            else:
                st.info("No transcript available for this speech.")

    # Export Section
    st.markdown("---")
    st.markdown("### Export Data")

    # Column selection for export
    all_columns = filtered_df.columns.tolist()
    default_export_cols = [c for c in ['name', 'year', 'title', 'speech_type', 'party',
                                        'word_count', 'sentiment_polarity', 'flesch_reading_ease']
                           if c in all_columns]

    export_columns = st.multiselect(
        "Select columns to export",
        all_columns,
        default=default_export_cols,
        key="export_columns"
    )

    include_transcript = st.checkbox(
        "Include full transcript (large file)",
        value=False,
        key="include_transcript"
    )

    if include_transcript and 'transcript' not in export_columns:
        export_columns.append('transcript')

    if st.button("Generate CSV", key="generate_csv_btn"):
        if export_columns:
            csv_data = export_to_csv(filtered_df, export_columns)

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="presidential_speeches_export.csv",
                mime="text/csv",
                key="download_csv_btn"
            )

            st.success(f"CSV ready for download with {len(filtered_df)} speeches and {len(export_columns)} columns.")
        else:
            st.warning("Please select at least one column to export.")

    # Statistics Summary
    st.markdown("---")
    st.markdown("### Quick Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Speeches", len(filtered_df))
        if 'name' in filtered_df.columns:
            st.metric("Unique Presidents", filtered_df['name'].nunique())

    with col2:
        if 'year' in filtered_df.columns:
            years = filtered_df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())} - {int(years.max())}")
        if 'word_count' in filtered_df.columns:
            st.metric("Total Words", f"{filtered_df['word_count'].sum():,}")

    with col3:
        if 'party' in filtered_df.columns:
            st.metric("Parties Represented", filtered_df['party'].nunique())
        if 'speech_type' in filtered_df.columns:
            st.metric("Speech Types", filtered_df['speech_type'].nunique())
