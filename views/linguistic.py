"""
Linguistic Patterns tab view for Presidential Speech Dashboard.
Displays pronoun analysis, sentence lengths, question ratios, and tense analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis.linguistic import (
    analyze_pronouns,
    analyze_sentence_lengths,
    analyze_question_ratio,
    analyze_tense,
)
from utils.constants import COLORBLIND_PALETTE
from utils.loading import loading_indicator


@st.cache_data
def _cached_linguistic_analysis(texts_tuple, names_tuple):
    """Cache linguistic analysis for all texts."""
    results = []
    for text, name in zip(texts_tuple, names_tuple):
        if not text:
            continue
        pronouns = analyze_pronouns(text)
        sentences = analyze_sentence_lengths(text)
        questions = analyze_question_ratio(text)
        tense = analyze_tense(text)

        results.append({
            'name': name,
            'first_singular_rate': pronouns['first_singular_rate'],
            'first_plural_rate': pronouns['first_plural_rate'],
            'i_we_ratio': pronouns['i_we_ratio'],
            'mean_sentence_length': sentences['mean_length'],
            'median_sentence_length': sentences['median_length'],
            'sentence_count': sentences['sentence_count'],
            'question_ratio': questions['question_ratio'],
            'question_count': questions['question_count'],
            'past_ratio': tense['past_ratio'],
            'present_ratio': tense['present_ratio'],
            'future_ratio': tense['future_ratio'],
        })

    return pd.DataFrame(results)


def render_linguistic_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Linguistic Patterns tab content."""
    st.subheader("Linguistic Patterns")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No transcript data available for linguistic analysis.")
        return

    # Prepare data
    texts = filtered_df['transcript'].fillna('').tolist()
    names = filtered_df['name'].fillna('Unknown').tolist() if 'name' in filtered_df.columns else ['Unknown'] * len(texts)

    if not any(texts):
        st.warning("No text data available for analysis.")
        return

    # Compute linguistic features
    with loading_indicator("Analyzing linguistic patterns..."):
        ling_df = _cached_linguistic_analysis(tuple(texts), tuple(names))

    if ling_df.empty:
        st.warning("Could not compute linguistic features.")
        return

    # Pronoun Analysis Section
    st.markdown("### Pronoun Usage")
    st.markdown("*The I/We ratio indicates leadership style: higher values suggest individual focus, lower values suggest collective focus.*")

    col1, col2 = st.columns(2)

    with col1:
        # I vs We rates by president
        if 'name' in ling_df.columns:
            pres_pronouns = ling_df.groupby('name').agg({
                'first_singular_rate': 'mean',
                'first_plural_rate': 'mean',
                'i_we_ratio': 'mean'
            }).reset_index()

            # Sort by I/We ratio
            pres_pronouns = pres_pronouns.sort_values('i_we_ratio', ascending=True).tail(15)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=pres_pronouns['name'],
                x=pres_pronouns['first_singular_rate'],
                name='I/me/my (%)',
                orientation='h',
                marker_color=COLORBLIND_PALETTE[0]
            ))
            fig.add_trace(go.Bar(
                y=pres_pronouns['name'],
                x=pres_pronouns['first_plural_rate'],
                name='We/us/our (%)',
                orientation='h',
                marker_color=COLORBLIND_PALETTE[1]
            ))
            fig.update_layout(
                title="First Person Pronoun Usage by President",
                barmode='group',
                xaxis_title="% of Words",
                yaxis_title="",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # I/We ratio distribution
        fig = px.histogram(
            ling_df,
            x='i_we_ratio',
            nbins=20,
            title="Distribution of I/We Ratio",
            labels={'i_we_ratio': 'I/We Ratio'},
            color_discrete_sequence=COLORBLIND_PALETTE,
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Equal I and We")
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        avg_ratio = ling_df['i_we_ratio'].mean()
        st.metric("Average I/We Ratio", f"{avg_ratio:.2f}")
        st.caption("Values > 1 indicate more individual focus; < 1 indicates more collective focus")

    # Sentence Length Section
    st.markdown("---")
    st.markdown("### Sentence Length")
    st.markdown("*Sentence length reflects rhetorical style: shorter sentences are more direct; longer sentences can be more complex or formal.*")

    col1, col2 = st.columns(2)

    with col1:
        # Mean sentence length by president
        if 'name' in ling_df.columns:
            pres_sentences = ling_df.groupby('name')['mean_sentence_length'].mean().reset_index()
            pres_sentences = pres_sentences.sort_values('mean_sentence_length', ascending=True).tail(15)

            fig = px.bar(
                pres_sentences,
                y='name',
                x='mean_sentence_length',
                orientation='h',
                title="Average Sentence Length by President",
                labels={'mean_sentence_length': 'Words per Sentence', 'name': ''},
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sentence length distribution
        fig = px.histogram(
            ling_df,
            x='mean_sentence_length',
            nbins=20,
            title="Distribution of Mean Sentence Length",
            labels={'mean_sentence_length': 'Words per Sentence'},
            color_discrete_sequence=[COLORBLIND_PALETTE[2]],
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_length = ling_df['mean_sentence_length'].mean()
        st.metric("Average Sentence Length", f"{avg_length:.1f} words")

    # Question/Statement Analysis
    st.markdown("---")
    st.markdown("### Rhetorical Questions")
    st.markdown("*Question ratio shows how often rhetorical questions are used for emphasis or engagement.*")

    col1, col2 = st.columns(2)

    with col1:
        if 'name' in ling_df.columns:
            pres_questions = ling_df.groupby('name')['question_ratio'].mean().reset_index()
            pres_questions = pres_questions.sort_values('question_ratio', ascending=True).tail(15)

            fig = px.bar(
                pres_questions,
                y='name',
                x='question_ratio',
                orientation='h',
                title="Question Usage by President",
                labels={'question_ratio': '% Questions', 'name': ''},
                color_discrete_sequence=[COLORBLIND_PALETTE[3]],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            ling_df,
            x='question_ratio',
            nbins=20,
            title="Distribution of Question Usage",
            labels={'question_ratio': '% Sentences as Questions'},
            color_discrete_sequence=[COLORBLIND_PALETTE[3]],
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_q = ling_df['question_ratio'].mean()
        st.metric("Average Question Ratio", f"{avg_q:.1f}%")

    # Tense Analysis
    st.markdown("---")
    st.markdown("### Temporal Orientation (Verb Tense)")
    st.markdown("*Tense usage reveals whether speeches focus on the past, present, or future.*")

    col1, col2 = st.columns(2)

    with col1:
        # Aggregate tense data
        tense_means = ling_df[['past_ratio', 'present_ratio', 'future_ratio']].mean()
        tense_df = pd.DataFrame({
            'Tense': ['Past', 'Present', 'Future'],
            'Percentage': [tense_means['past_ratio'], tense_means['present_ratio'], tense_means['future_ratio']]
        })

        fig = px.pie(
            tense_df,
            values='Percentage',
            names='Tense',
            title="Overall Tense Distribution",
            color_discrete_sequence=COLORBLIND_PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'name' in ling_df.columns:
            pres_tense = ling_df.groupby('name')[['past_ratio', 'present_ratio', 'future_ratio']].mean().reset_index()

            # Get top presidents by future focus
            pres_tense = pres_tense.sort_values('future_ratio', ascending=False).head(15)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=pres_tense['name'],
                x=pres_tense['past_ratio'],
                name='Past',
                orientation='h',
                marker_color=COLORBLIND_PALETTE[0]
            ))
            fig.add_trace(go.Bar(
                y=pres_tense['name'],
                x=pres_tense['present_ratio'],
                name='Present',
                orientation='h',
                marker_color=COLORBLIND_PALETTE[1]
            ))
            fig.add_trace(go.Bar(
                y=pres_tense['name'],
                x=pres_tense['future_ratio'],
                name='Future',
                orientation='h',
                marker_color=COLORBLIND_PALETTE[2]
            ))
            fig.update_layout(
                title="Tense Distribution by President",
                barmode='stack',
                xaxis_title="% of Verbs",
                yaxis_title="",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.markdown("---")
    st.markdown("### Linguistic Summary Table")

    if 'name' in ling_df.columns:
        summary = ling_df.groupby('name').agg({
            'i_we_ratio': 'mean',
            'mean_sentence_length': 'mean',
            'question_ratio': 'mean',
            'future_ratio': 'mean',
        }).round(2).reset_index()

        summary.columns = ['President', 'I/We Ratio', 'Avg Sentence Length', 'Question %', 'Future Focus %']
        summary = summary.sort_values('President')

        st.dataframe(summary, use_container_width=True, height=400)
