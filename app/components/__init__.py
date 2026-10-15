"""app/components/__init__.py"""
from .prediction_card import render_prediction_card
from .taxonomy_view import render_taxonomy_hierarchy
from .confidence_view import render_confidence_and_evidence
from .similarity_view import render_similarity_table
from .biodiversity_view import render_biodiversity_context
from .rank_resolution import render_rank_resolution
from .charts import (
    plot_status_distribution,
    plot_taxonomic_breakdown,
    plot_station_diversity_map,
    plot_diversity_vs_depth
)

__all__ = [
    "render_prediction_card",
    "render_taxonomy_hierarchy",
    "render_confidence_and_evidence",
    "render_similarity_table",
    "render_biodiversity_context",
    "render_rank_resolution",
    "plot_status_distribution",
    "plot_taxonomic_breakdown",
    "plot_station_diversity_map",
    "plot_diversity_vs_depth"
]
