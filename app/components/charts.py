"""
app/components/charts.py
Phase 9: Plotly Charting Utilities for BioScan Dashboard

Provides reusable, polished visualization components for:
1. Batch prediction status donut charts
2. Taxonomic composition bar charts
3. Station-level Shannon diversity maps
4. Diversity vs Depth regression / scatter views
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def plot_status_distribution(batch_df: pd.DataFrame) -> go.Figure:
    """Create a donut chart of prediction status counts in batch analysis."""
    if batch_df.empty or "prediction_status" not in batch_df.columns:
        return go.Figure()

    status_counts = batch_df["prediction_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    color_map = {
        "known_reference_like": "#10b981", # Emerald
        "low_confidence": "#f59e0b",       # Amber
        "out_of_reference": "#ef4444",     # Red
        "qc_failed": "#94a3b8"             # Slate
    }
    colors = [color_map.get(s, "#3b82f6") for s in status_counts["Status"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=status_counts["Status"],
                values=status_counts["Count"],
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+value+percent",
                hoverinfo="label+value+percent"
            )
        ]
    )
    fig.update_layout(
        title="Prediction Status Distribution",
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_taxonomic_breakdown(df: pd.DataFrame, rank: str = "class") -> go.Figure:
    """Create a horizontal bar chart of taxonomic frequencies."""
    if df.empty or rank not in df.columns:
        return go.Figure()

    clean_series = df[rank].fillna("Unassigned").replace("", "Unassigned")
    counts = clean_series.value_counts().head(10).reset_index()
    counts.columns = [rank.capitalize(), "Count"]

    fig = px.bar(
        counts,
        x="Count",
        y=rank.capitalize(),
        orientation="h",
        color="Count",
        color_continuous_scale="Viridis",
        title=f"Top Predicted {rank.capitalize()} Frequencies"
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    return fig


def plot_station_diversity_map(diversity_df: pd.DataFrame) -> go.Figure:
    """Create an interactive Mapbox plot of sampling stations colored by Shannon diversity."""
    if diversity_df.empty or "station_latitude" not in diversity_df.columns:
        return go.Figure()

    scatter_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
    style_kw = {"map_style": "open-street-map"} if hasattr(px, "scatter_map") else {"mapbox_style": "open-street-map"}

    fig = scatter_fn(
        diversity_df,
        lat="station_latitude",
        lon="station_longitude",
        hover_name="station_name",
        hover_data={
            "shannon_diversity_H_reads": ":.3f",
            "taxonomic_richness_S": True,
            "depth_meters": ":.1f m",
            "dominant_taxon": True,
            "station_latitude": False,
            "station_longitude": False
        },
        color="shannon_diversity_H_reads",
        size="taxonomic_richness_S",
        color_continuous_scale="Tealgrn",
        zoom=6.8,
        center=dict(lat=39.5, lon=-74.3),
        title="Survey Stations: Shannon Diversity (H_reads) & Species Richness",
        **style_kw
    )
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_diversity_vs_depth(diversity_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of station Shannon diversity vs water depth."""
    if diversity_df.empty or "depth_meters" not in diversity_df.columns:
        return go.Figure()

    fig = px.scatter(
        diversity_df,
        x="depth_meters",
        y="shannon_diversity_H_reads",
        hover_name="station_name",
        color="dominant_taxon",
        title="Shannon Diversity (H_reads) vs Station Depth (meters)",
        labels={
            "depth_meters": "Station Depth (m)",
            "shannon_diversity_H_reads": "Shannon Diversity (H_reads)"
        }
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
