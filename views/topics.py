"""
Topics tab view for Presidential Speech Dashboard.
Displays LDA topic modeling results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.topics import build_topic_model
from utils.constants import COLORBLIND_PALETTE, DEFAULT_N_TOPICS
from utils.loading import loading_indicator, StatusIndicator


@st.cache_data
def _cached_topic_model(texts_tuple, n_topics):
    """Cache topic model building."""
    return build_topic_model(list(texts_tuple), n_topics)


def render_topics_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Topics tab content."""
    st.subheader("Topic Modeling")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No transcript data available for topic modeling.")
        return

    texts = filtered_df['transcript'].dropna().tolist()
    if len(texts) < 10:
        st.warning("Need at least 10 speeches for topic modeling. Currently have: " + str(len(texts)))
        return

    # Topic modeling settings
    col1, col2 = st.columns([1, 3])

    with col1:
        n_topics = st.slider(
            "Number of topics",
            min_value=3,
            max_value=20,
            value=DEFAULT_N_TOPICS,
            key="n_topics_slider"
        )

        st.info(f"Analyzing {len(texts)} speeches...")

    # Build topic model
    with loading_indicator(f"Building topic model with {n_topics} topics..."):
        topic_result = _cached_topic_model(tuple(texts), n_topics)

    if not topic_result['topics']:
        st.warning("Could not build topic model. Try selecting more speeches or adjusting settings.")
        return

    # Display topics
    st.markdown("### Discovered Topics")

    topics = topic_result['topics']
    doc_topics = topic_result['doc_topics']

    # Create topic cards in a grid
    cols_per_row = 2
    for i in range(0, len(topics), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(topics):
                topic = topics[i + j]
                with col:
                    topic_count = doc_topics.count(topic['topic_id'])
                    st.markdown(f"**Topic {topic['topic_id'] + 1}** ({topic_count} speeches)")
                    words = ', '.join(topic['words'][:8])
                    st.markdown(f"*{words}*")

    st.markdown("---")

    # Topic word visualization
    st.markdown("### Topic Word Weights")

    selected_topic = st.selectbox(
        "Select topic to visualize",
        options=range(len(topics)),
        format_func=lambda x: f"Topic {x + 1}: {', '.join(topics[x]['words'][:3])}...",
        key="topic_select"
    )

    topic = topics[selected_topic]
    word_df = pd.DataFrame({
        'Word': topic['words'],
        'Weight': topic['weights']
    })

    fig = px.bar(
        word_df,
        x='Weight',
        y='Word',
        orientation='h',
        title=f"Top Words in Topic {selected_topic + 1}",
        color_discrete_sequence=[COLORBLIND_PALETTE[selected_topic % len(COLORBLIND_PALETTE)]],
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Topic distribution by president
    st.markdown("### Topic Distribution by President")

    if 'name' in filtered_df.columns:
        # Add topic to filtered dataframe
        analysis_df = filtered_df.copy()
        analysis_df = analysis_df[analysis_df['transcript'].notna()].reset_index(drop=True)

        if len(analysis_df) == len(doc_topics):
            analysis_df['dominant_topic'] = doc_topics

            # Get top presidents by speech count
            top_presidents = analysis_df['name'].value_counts().head(10).index.tolist()
            president_df = analysis_df[analysis_df['name'].isin(top_presidents)]

            if not president_df.empty:
                # Create crosstab
                topic_dist = pd.crosstab(
                    president_df['name'],
                    president_df['dominant_topic'],
                    normalize='index'
                ) * 100

                # Rename columns
                topic_dist.columns = [f'Topic {i+1}' for i in topic_dist.columns]

                # Plot heatmap
                fig = px.imshow(
                    topic_dist,
                    labels=dict(x="Topic", y="President", color="% Speeches"),
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Topic Distribution by President (%)",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topic distribution could not be computed for all speeches.")
    else:
        st.info("President name data not available.")

    # Topic trends over time
    st.markdown("---")
    st.markdown("### Topic Trends Over Time")

    if 'year' in filtered_df.columns:
        analysis_df = filtered_df.copy()
        analysis_df = analysis_df[analysis_df['transcript'].notna()].reset_index(drop=True)

        if len(analysis_df) == len(doc_topics):
            analysis_df['dominant_topic'] = doc_topics

            # Group by decade for smoother trends
            analysis_df['decade'] = (analysis_df['year'] // 10) * 10

            topic_trends = pd.crosstab(
                analysis_df['decade'],
                analysis_df['dominant_topic'],
                normalize='index'
            ) * 100

            topic_trends.columns = [f'Topic {i+1}' for i in topic_trends.columns]

            # Melt for line plot
            melted = topic_trends.reset_index().melt(
                id_vars=['decade'],
                var_name='Topic',
                value_name='Percentage'
            )

            fig = px.line(
                melted,
                x='decade',
                y='Percentage',
                color='Topic',
                title="Topic Prevalence by Decade",
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            fig.update_layout(xaxis_title="Decade", yaxis_title="% of Speeches")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topic trends could not be computed.")
    else:
        st.info("Year data not available for trend analysis.")

    # Document-topic table
    st.markdown("---")
    st.markdown("### Speech Topic Assignments")

    if len(filtered_df) == len(doc_topics):
        display_df = filtered_df[['name', 'year', 'title']].copy().reset_index(drop=True)
        display_df['Dominant Topic'] = [f"Topic {t+1}" for t in doc_topics]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
