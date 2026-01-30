"""
Comparative Views tab for Presidential Speech Dashboard.
Displays party, era, wartime, and term comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.constants import (
    COLORBLIND_PALETTE,
    PARTY_COLORS,
    WAR_PERIODS,
    HISTORICAL_ERAS,
)


def get_war_status(year: int) -> str:
    """Determine if a year falls within a war period."""
    for war_name, (start, end) in WAR_PERIODS.items():
        if start <= year <= end:
            return war_name
    return "Peacetime"


def get_historical_era(year: int) -> str:
    """Determine the historical era for a year."""
    for era_name, (start, end) in HISTORICAL_ERAS.items():
        if start <= year <= end:
            return era_name
    return "Unknown"


def render_comparisons_tab(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render the Comparisons tab content."""
    st.subheader("Comparative Analysis")

    if filtered_df.empty:
        st.warning("No data available for comparison.")
        return

    # Add computed columns
    analysis_df = filtered_df.copy()

    if 'year' in analysis_df.columns:
        analysis_df['war_status'] = analysis_df['year'].apply(get_war_status)
        analysis_df['is_wartime'] = analysis_df['war_status'] != 'Peacetime'
        analysis_df['historical_era'] = analysis_df['year'].apply(get_historical_era)

    # Available metrics for comparison
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    comparison_metrics = [c for c in numeric_cols if c not in ['year', 'presidential_number', 'terms_served']]

    if not comparison_metrics:
        st.warning("No numeric metrics available for comparison.")
        return

    # Party Comparison
    st.markdown("### Party Comparison")

    if 'party' in analysis_df.columns:
        party_counts = analysis_df['party'].value_counts()
        parties_with_data = party_counts[party_counts >= 5].index.tolist()

        if len(parties_with_data) >= 2:
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                comparison_metrics,
                default=comparison_metrics[:4] if len(comparison_metrics) >= 4 else comparison_metrics,
                key="party_metrics"
            )

            if selected_metrics:
                party_df = analysis_df[analysis_df['party'].isin(parties_with_data)]
                party_means = party_df.groupby('party')[selected_metrics].mean()

                col1, col2 = st.columns(2)

                with col1:
                    # Grouped bar chart
                    melted = party_means.reset_index().melt(
                        id_vars=['party'],
                        value_vars=selected_metrics,
                        var_name='Metric',
                        value_name='Value'
                    )

                    fig = px.bar(
                        melted,
                        x='Metric',
                        y='Value',
                        color='party',
                        barmode='group',
                        title="Average Metrics by Party",
                        color_discrete_map=PARTY_COLORS,
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Summary table
                    st.markdown("#### Party Averages")
                    display_df = party_means.T.round(3)
                    st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Need at least 2 parties with 5+ speeches each for comparison.")
    else:
        st.info("Party data not available.")

    # Era Comparison
    st.markdown("---")
    st.markdown("### Historical Era Comparison")

    if 'historical_era' in analysis_df.columns:
        era_counts = analysis_df['historical_era'].value_counts()
        eras_with_data = era_counts[era_counts >= 3].index.tolist()

        # Sort eras chronologically
        era_order = [e for e in HISTORICAL_ERAS.keys() if e in eras_with_data]

        if len(era_order) >= 2:
            era_metric = st.selectbox(
                "Select metric for era comparison",
                comparison_metrics,
                key="era_metric"
            )

            era_df = analysis_df[analysis_df['historical_era'].isin(era_order)]
            era_means = era_df.groupby('historical_era')[era_metric].agg(['mean', 'std', 'count']).reset_index()
            era_means.columns = ['Era', 'Mean', 'Std', 'Count']

            # Reorder by historical order
            era_means['order'] = era_means['Era'].map({e: i for i, e in enumerate(era_order)})
            era_means = era_means.sort_values('order')

            fig = px.bar(
                era_means,
                x='Era',
                y='Mean',
                error_y='Std',
                title=f"{era_metric} by Historical Era",
                color='Era',
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            # Trend line across eras
            st.markdown("#### Era Trends")
            fig = px.line(
                era_means,
                x='Era',
                y='Mean',
                title=f"{era_metric} Trend Across Eras",
                markers=True,
                color_discrete_sequence=COLORBLIND_PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical eras with data for comparison.")
    else:
        st.info("Year data needed for era comparison.")

    # Wartime vs Peacetime
    st.markdown("---")
    st.markdown("### Wartime vs Peacetime Comparison")

    if 'is_wartime' in analysis_df.columns:
        wartime_counts = analysis_df['is_wartime'].value_counts()

        if True in wartime_counts.index and False in wartime_counts.index:
            col1, col2 = st.columns(2)

            with col1:
                # Speech counts
                st.markdown("#### Speech Counts")
                fig = px.pie(
                    values=[wartime_counts.get(True, 0), wartime_counts.get(False, 0)],
                    names=['Wartime', 'Peacetime'],
                    title="Speeches by Period",
                    color_discrete_sequence=[COLORBLIND_PALETTE[6], COLORBLIND_PALETTE[2]],
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # War periods breakdown
                war_breakdown = analysis_df[analysis_df['is_wartime']]['war_status'].value_counts()
                fig = px.bar(
                    x=war_breakdown.index,
                    y=war_breakdown.values,
                    title="Speeches by War Period",
                    labels={'x': 'War Period', 'y': 'Speech Count'},
                    color_discrete_sequence=COLORBLIND_PALETTE,
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            # Metric comparison
            war_metrics = st.multiselect(
                "Select metrics to compare",
                comparison_metrics,
                default=comparison_metrics[:3] if len(comparison_metrics) >= 3 else comparison_metrics,
                key="war_metrics"
            )

            if war_metrics:
                war_comparison = analysis_df.groupby('is_wartime')[war_metrics].mean()
                war_comparison.index = war_comparison.index.map({True: 'Wartime', False: 'Peacetime'})

                melted = war_comparison.reset_index().melt(
                    id_vars=['is_wartime'],
                    value_vars=war_metrics,
                    var_name='Metric',
                    value_name='Value'
                )

                fig = px.bar(
                    melted,
                    x='Metric',
                    y='Value',
                    color='is_wartime',
                    barmode='group',
                    title="Wartime vs Peacetime Metrics",
                    color_discrete_sequence=[COLORBLIND_PALETTE[6], COLORBLIND_PALETTE[2]],
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.markdown("#### Summary Statistics")
                st.dataframe(war_comparison.T.round(3), use_container_width=True)
        else:
            st.info("Need both wartime and peacetime speeches for comparison.")
    else:
        st.info("Year data needed for wartime analysis.")

    # Term Comparison (First vs Later Terms)
    st.markdown("---")
    st.markdown("### First Term vs Later Terms")

    if 'terms_served' in analysis_df.columns and 'name' in analysis_df.columns:
        # Find presidents with multiple terms
        pres_terms = analysis_df.groupby('name')['terms_served'].max()
        multi_term_pres = pres_terms[pres_terms > 1].index.tolist()

        if multi_term_pres:
            st.markdown(f"*Comparing speeches from {len(multi_term_pres)} presidents who served multiple terms.*")

            # This is a simplification - would need term start dates for precise analysis
            # Using year order as proxy
            multi_term_df = analysis_df[analysis_df['name'].isin(multi_term_pres)].copy()

            term_metric = st.selectbox(
                "Select metric for term comparison",
                comparison_metrics,
                key="term_metric"
            )

            # Approximate first vs later term by year order
            term_data = []
            for pres in multi_term_pres:
                pres_df = multi_term_df[multi_term_df['name'] == pres].sort_values('year')
                if len(pres_df) >= 4:
                    midpoint = len(pres_df) // 2
                    first_term_avg = pres_df.head(midpoint)[term_metric].mean()
                    later_term_avg = pres_df.tail(len(pres_df) - midpoint)[term_metric].mean()
                    term_data.append({
                        'President': pres,
                        'First Half': first_term_avg,
                        'Second Half': later_term_avg,
                        'Change': later_term_avg - first_term_avg
                    })

            if term_data:
                term_df = pd.DataFrame(term_data)
                term_df = term_df.sort_values('Change')

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=term_df['President'],
                    x=term_df['First Half'],
                    name='First Half of Tenure',
                    orientation='h',
                    marker_color=COLORBLIND_PALETTE[0]
                ))
                fig.add_trace(go.Bar(
                    y=term_df['President'],
                    x=term_df['Second Half'],
                    name='Second Half of Tenure',
                    orientation='h',
                    marker_color=COLORBLIND_PALETTE[1]
                ))
                fig.update_layout(
                    title=f"{term_metric} - First vs Second Half of Tenure",
                    barmode='group',
                    xaxis_title=term_metric,
                    height=400 + len(term_data) * 25
                )
                st.plotly_chart(fig, use_container_width=True)

                # Change summary
                avg_change = term_df['Change'].mean()
                st.metric(
                    "Average Change (Later - Earlier)",
                    f"{avg_change:+.3f}",
                    delta_color="normal" if avg_change >= 0 else "inverse"
                )
            else:
                st.info("Not enough speeches per president for term comparison.")
        else:
            st.info("No multi-term presidents in selected data.")
    else:
        st.info("Term data not available for comparison.")

    # Box Plots for Selected Metric
    st.markdown("---")
    st.markdown("### Metric Distribution by President")

    if 'name' in analysis_df.columns:
        box_metric = st.selectbox(
            "Select metric for box plot",
            comparison_metrics,
            key="box_metric"
        )

        # Get top presidents by speech count
        top_presidents = analysis_df['name'].value_counts().head(20).index.tolist()
        box_df = analysis_df[analysis_df['name'].isin(top_presidents)]

        fig = px.box(
            box_df,
            x='name',
            y=box_metric,
            title=f"Distribution of {box_metric} by President",
            color='name',
            color_discrete_sequence=COLORBLIND_PALETTE * 3,
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=45, height=500)
        st.plotly_chart(fig, use_container_width=True)
