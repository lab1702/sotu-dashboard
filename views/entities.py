"""
Named Entity Recognition tab view for Presidential Speech Dashboard.
Displays entity extraction, trends, and co-occurrence.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.entities import (
    get_top_entities,
    entity_mentions_over_time,
    entity_cooccurrence,
    ENTITY_TYPES,
)
from utils.constants import COLORBLIND_PALETTE
from utils.loading import loading_indicator


@st.cache_data
def _cached_top_entities(texts_tuple, entity_type, top_k):
    """Cache entity extraction."""
    return get_top_entities(list(texts_tuple), entity_type, top_k)


@st.cache_data
def _cached_entity_mentions(df_hash, df_years, df_texts, entity_name):
    """Cache entity mention tracking."""
    temp_df = pd.DataFrame({'year': df_years, 'transcript': df_texts})
    return entity_mentions_over_time(temp_df, entity_name, 'transcript', 'year')


def render_entities_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Named Entities tab content."""
    st.subheader("Named Entity Recognition")

    if filtered_df.empty or 'transcript' not in filtered_df.columns:
        st.warning("No transcript data available for entity extraction.")
        return

    texts = filtered_df['transcript'].dropna().tolist()
    if not texts:
        st.warning("No text data available for analysis.")
        return

    st.info(f"Analyzing {len(texts)} speeches for named entities. This may take a moment for large selections.")

    # Entity type selection
    entity_types = list(ENTITY_TYPES.keys())
    entity_labels = list(ENTITY_TYPES.values())

    selected_type = st.selectbox(
        "Select entity type",
        entity_types,
        format_func=lambda x: f"{x} - {ENTITY_TYPES[x]}",
        key="entity_type_select"
    )

    # Top entities for selected type
    st.markdown(f"### Top {ENTITY_TYPES[selected_type]}")

    with loading_indicator(f"Extracting {ENTITY_TYPES[selected_type]} from {len(texts)} speeches..."):
        top_entities = _cached_top_entities(tuple(texts), selected_type, 30)

    col1, col2 = st.columns(2)

    with col1:
        if top_entities:
            entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])

            fig = px.bar(
                entities_df.head(20),
                y='Entity',
                x='Count',
                orientation='h',
                title=f"Top 20 {ENTITY_TYPES[selected_type]}",
                color_discrete_sequence=[COLORBLIND_PALETTE[entity_types.index(selected_type) % len(COLORBLIND_PALETTE)]],
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {selected_type} entities found in selected speeches.")

    with col2:
        if top_entities:
            st.markdown("#### Entity Table")
            st.dataframe(
                entities_df,
                use_container_width=True,
                height=550
            )

    # Entity comparison across types
    st.markdown("---")
    st.markdown("### Entity Comparison Across Types")

    comparison_types = ['PERSON', 'GPE', 'ORG', 'NORP']
    available_types = [t for t in comparison_types if t in entity_types]

    comparison_data = []
    with loading_indicator("Comparing entity types across speeches..."):
        for etype in available_types:
            entities = _cached_top_entities(tuple(texts), etype, 10)
            for entity, count in entities:
                comparison_data.append({
                    'Entity Type': ENTITY_TYPES[etype],
                    'Entity': entity,
                    'Count': count
                })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        fig = px.bar(
            comparison_df,
            x='Entity',
            y='Count',
            color='Entity Type',
            title="Top Entities by Type",
            color_discrete_sequence=COLORBLIND_PALETTE,
        )
        fig.update_layout(xaxis_tickangle=45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Entity trends over time
    st.markdown("---")
    st.markdown("### Entity Mentions Over Time")

    if 'year' in filtered_df.columns:
        # Get some notable entities for tracking
        if top_entities:
            entity_options = [e for e, _ in top_entities[:20]]

            selected_entities = st.multiselect(
                "Select entities to track over time",
                entity_options,
                default=entity_options[:3] if len(entity_options) >= 3 else entity_options,
                key="entity_track_select"
            )

            custom_entity = st.text_input("Add custom entity to track", key="custom_entity")
            if custom_entity:
                selected_entities.append(custom_entity)

            if selected_entities:
                df_hash = hash(tuple(filtered_df.index.tolist()))
                df_years = filtered_df['year'].tolist()
                df_texts = filtered_df['transcript'].fillna('').tolist()

                trends_data = []
                with loading_indicator("Computing entity trends over time..."):
                    for entity in selected_entities:
                        mentions_df = _cached_entity_mentions(df_hash, tuple(df_years), tuple(df_texts), entity)
                        if not mentions_df.empty:
                            for _, row in mentions_df.iterrows():
                                trends_data.append({
                                    'year': row['year'],
                                    'entity': entity,
                                    'mentions': row['mentions']
                                })

                if trends_data:
                    trends_df = pd.DataFrame(trends_data)
                    fig = px.line(
                        trends_df,
                        x='year',
                        y='mentions',
                        color='entity',
                        title="Entity Mentions Over Time",
                        color_discrete_sequence=COLORBLIND_PALETTE,
                        markers=True,
                    )
                    fig.update_layout(xaxis_title="Year", yaxis_title="Mentions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No mentions found for selected entities.")
        else:
            st.info("Extract entities first to see trends.")
    else:
        st.info("Year data not available for trend analysis.")

    # Entity by President
    st.markdown("---")
    st.markdown("### Top Entities by President")

    if 'name' in filtered_df.columns:
        presidents = filtered_df['name'].unique().tolist()

        if len(presidents) <= 10:
            selected_presidents = presidents
        else:
            selected_presidents = st.multiselect(
                "Select presidents to compare",
                presidents,
                default=presidents[:5],
                key="entity_president_select"
            )

        if selected_presidents:
            pres_entity_data = []

            with loading_indicator("Analyzing entities by president..."):
                for pres in selected_presidents:
                    pres_texts = filtered_df[filtered_df['name'] == pres]['transcript'].dropna().tolist()
                    if pres_texts:
                        pres_entities = _cached_top_entities(tuple(pres_texts), selected_type, 5)
                        for entity, count in pres_entities:
                            pres_entity_data.append({
                                'President': pres,
                                'Entity': entity,
                                'Count': count
                            })

            if pres_entity_data:
                pres_df = pd.DataFrame(pres_entity_data)

                fig = px.bar(
                    pres_df,
                    x='President',
                    y='Count',
                    color='Entity',
                    title=f"Top {ENTITY_TYPES[selected_type]} by President",
                    color_discrete_sequence=COLORBLIND_PALETTE,
                    barmode='group',
                )
                fig.update_layout(xaxis_tickangle=45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No entity data for selected presidents.")

    # Entity Co-occurrence
    st.markdown("---")
    st.markdown("### Entity Co-occurrence")
    st.markdown("*Entities that frequently appear together in the same speeches.*")

    with loading_indicator("Computing entity co-occurrences..."):
        cooccur_df = entity_cooccurrence(texts, ['PERSON', 'ORG', 'GPE'], min_count=3)

    if not cooccur_df.empty:
        st.dataframe(
            cooccur_df.head(30),
            use_container_width=True,
            height=400
        )
    else:
        st.info("Not enough entity co-occurrences found. Try selecting more speeches.")
