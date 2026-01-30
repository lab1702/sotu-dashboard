"""
Emotions tab view for Presidential Speech Dashboard.
Displays multi-emotion detection and tone analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis.emotions import (
    analyze_emotions,
    get_emotional_intensity,
    get_dominant_emotion,
    analyze_tone_shifts,
    emotion_trends_over_time,
    compare_emotions_by_group,
)
from utils.constants import COLORBLIND_PALETTE, EMOTION_CATEGORIES
from utils.loading import loading_indicator


@st.cache_data
def _cached_emotion_analysis(texts_tuple):
    """Cache emotion analysis for all texts."""
    results = []
    for text in texts_tuple:
        if not text:
            continue
        emotions = analyze_emotions(text)
        intensity = get_emotional_intensity(text)
        dominant = get_dominant_emotion(text)
        emotions['intensity'] = intensity
        emotions['dominant'] = dominant
        results.append(emotions)
    return results


@st.cache_data
def _cached_emotion_trends(df_hash, df_years, df_texts):
    """Cache emotion trends over time."""
    temp_df = pd.DataFrame({'year': df_years, 'transcript': df_texts})
    return emotion_trends_over_time(temp_df, 'transcript', 'year')


def render_emotions_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Emotions tab content."""
    st.subheader("Emotional Analysis")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No transcript data available for emotion analysis.")
        return

    texts = filtered_df['transcript'].fillna('').tolist()
    if not any(texts):
        st.warning("No text data available for analysis.")
        return

    # Compute emotions
    with loading_indicator("Analyzing emotions across all speeches..."):
        emotion_results = _cached_emotion_analysis(tuple(texts))

    if not emotion_results:
        st.warning("Could not analyze emotions.")
        return

    emotions_df = pd.DataFrame(emotion_results)

    # Add metadata from filtered_df
    meta_cols = ['name', 'year', 'title', 'party']
    for col in meta_cols:
        if col in filtered_df.columns and len(filtered_df) == len(emotions_df):
            emotions_df[col] = filtered_df[col].values

    # Overview Section
    st.markdown("### Emotion Overview")

    col1, col2, col3 = st.columns(3)

    core_emotions = [e for e in EMOTION_CATEGORIES if e not in ['positive', 'negative']]

    with col1:
        # Average emotion profile
        avg_emotions = emotions_df[core_emotions].mean()
        fig = px.bar(
            x=avg_emotions.values,
            y=avg_emotions.index,
            orientation='h',
            title="Average Emotion Profile",
            labels={'x': 'Score', 'y': 'Emotion'},
            color=avg_emotions.index,
            color_discrete_sequence=COLORBLIND_PALETTE,
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dominant emotion distribution
        dominant_counts = emotions_df['dominant'].value_counts()
        fig = px.pie(
            values=dominant_counts.values,
            names=dominant_counts.index,
            title="Dominant Emotion Distribution",
            color_discrete_sequence=COLORBLIND_PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Key metrics
        st.markdown("#### Key Metrics")
        st.metric("Average Intensity", f"{emotions_df['intensity'].mean():.2f}")
        st.metric("Most Common Emotion", emotions_df['dominant'].mode().iloc[0] if not emotions_df['dominant'].mode().empty else "N/A")
        st.metric("Positive vs Negative", f"{emotions_df['positive'].mean():.2f} / {emotions_df['negative'].mean():.2f}")

    # Emotion Distribution
    st.markdown("---")
    st.markdown("### Emotion Distributions")

    selected_emotion = st.selectbox(
        "Select emotion to visualize",
        core_emotions,
        key="emotion_select"
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            emotions_df,
            x=selected_emotion,
            nbins=30,
            title=f"Distribution of {selected_emotion.title()} Score",
            color_discrete_sequence=[COLORBLIND_PALETTE[core_emotions.index(selected_emotion) % len(COLORBLIND_PALETTE)]],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'name' in emotions_df.columns:
            pres_emotion = emotions_df.groupby('name')[selected_emotion].mean().reset_index()
            pres_emotion = pres_emotion.sort_values(selected_emotion, ascending=True).tail(15)

            fig = px.bar(
                pres_emotion,
                y='name',
                x=selected_emotion,
                orientation='h',
                title=f"{selected_emotion.title()} by President",
                color_discrete_sequence=[COLORBLIND_PALETTE[core_emotions.index(selected_emotion) % len(COLORBLIND_PALETTE)]],
            )
            st.plotly_chart(fig, use_container_width=True)

    # Emotion Correlation
    st.markdown("---")
    st.markdown("### Emotion Correlations")

    corr_matrix = emotions_df[core_emotions].corr()
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Emotion Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Emotion Trends Over Time
    st.markdown("---")
    st.markdown("### Emotion Trends Over Time")

    if 'year' in emotions_df.columns:
        # Select emotions to track
        tracked_emotions = st.multiselect(
            "Select emotions to track",
            core_emotions,
            default=['trust', 'fear', 'joy'],
            key="tracked_emotions"
        )

        if tracked_emotions:
            year_emotions = emotions_df.groupby('year')[tracked_emotions].mean().reset_index()

            melted = year_emotions.melt(
                id_vars=['year'],
                value_vars=tracked_emotions,
                var_name='Emotion',
                value_name='Score'
            )

            fig = px.line(
                melted,
                x='year',
                y='Score',
                color='Emotion',
                title="Emotion Trends Over Time",
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Average Score")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Year data not available for trend analysis.")

    # Party Comparison
    st.markdown("---")
    st.markdown("### Emotion by Party")

    if 'party' in emotions_df.columns:
        party_emotions = emotions_df.groupby('party')[core_emotions].mean()
        party_emotions = party_emotions.T

        fig = go.Figure()
        for i, party in enumerate(party_emotions.columns):
            fig.add_trace(go.Bar(
                name=party,
                x=party_emotions.index,
                y=party_emotions[party],
                marker_color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
            ))

        fig.update_layout(
            title="Emotion Profile by Party",
            barmode='group',
            xaxis_title="Emotion",
            yaxis_title="Average Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Party data not available for comparison.")

    # Tone Shifts in Selected Speech
    st.markdown("---")
    st.markdown("### Tone Shifts Within a Speech")
    st.markdown("*Select a speech to see how emotions change throughout the text.*")

    if 'title' in filtered_df.columns:
        speech_options = filtered_df['title'].tolist()
        selected_speech_idx = st.selectbox(
            "Select a speech",
            range(len(speech_options)),
            format_func=lambda x: speech_options[x][:80] + "..." if len(speech_options[x]) > 80 else speech_options[x],
            key="tone_shift_speech"
        )

        if selected_speech_idx is not None:
            selected_text = filtered_df.iloc[selected_speech_idx]['transcript']

            if selected_text and len(str(selected_text)) > 100:
                with loading_indicator("Analyzing tone shifts in speech..."):
                    shifts_df = analyze_tone_shifts(str(selected_text), segment_size=300)

                if not shifts_df.empty and len(shifts_df) > 1:
                    # Select emotions to show
                    shift_emotions = ['trust', 'fear', 'joy', 'anger']
                    shift_melted = shifts_df.melt(
                        id_vars=['segment'],
                        value_vars=[e for e in shift_emotions if e in shifts_df.columns],
                        var_name='Emotion',
                        value_name='Score'
                    )

                    fig = px.line(
                        shift_melted,
                        x='segment',
                        y='Score',
                        color='Emotion',
                        title="Emotion Changes Through Speech",
                        color_discrete_sequence=COLORBLIND_PALETTE,
                        markers=True,
                    )
                    fig.update_layout(xaxis_title="Segment", yaxis_title="Emotion Score")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Speech is too short for tone shift analysis.")
            else:
                st.info("Selected speech has insufficient text for analysis.")

    # Summary Table
    st.markdown("---")
    st.markdown("### Emotion Summary Table")

    display_cols = ['name', 'title', 'year', 'dominant', 'intensity'] + core_emotions[:4]
    available_cols = [c for c in display_cols if c in emotions_df.columns]

    if available_cols:
        summary_df = emotions_df[available_cols].copy()
        for col in core_emotions[:4]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(3)
        summary_df['intensity'] = summary_df['intensity'].round(3)

        st.dataframe(summary_df, use_container_width=True, height=400)
