"""
BioScan Phase 7: Biodiversity Intelligence, Geographic Analysis & Analytics Layer
Package: src.biodiversity

Provides tools for:
- Geographic data quality auditing and coordinate validation
- Taxonomic hierarchy analysis and richness metrics
- Geographic occurrence and sampling coverage analysis
- Scientifically defensible Shannon diversity and evenness metrics
- Integration of Phase 6 taxonomy/status predictions with reference biodiversity metadata
"""

from .data_quality import (
    audit_geographic_data_quality,
    validate_coordinates,
    categorize_geographic_records
)
from .taxonomy_analysis import (
    compute_taxonomic_richness,
    analyze_taxonomic_hierarchy,
    build_taxonomic_summary
)
from .geographic_analysis import (
    build_biodiversity_reference_records,
    summarize_country_biodiversity,
    summarize_species_distribution,
    analyze_sampling_coverage
)
from .diversity_metrics import (
    calculate_shannon_diversity,
    calculate_simpson_diversity,
    calculate_pielou_evenness,
    compute_station_diversity_metrics
)
from .prediction_biodiversity import (
    analyze_prediction,
    BioScanBiodiversityIntelligence
)

__all__ = [
    "audit_geographic_data_quality",
    "validate_coordinates",
    "categorize_geographic_records",
    "compute_taxonomic_richness",
    "analyze_taxonomic_hierarchy",
    "build_taxonomic_summary",
    "build_biodiversity_reference_records",
    "summarize_country_biodiversity",
    "summarize_species_distribution",
    "analyze_sampling_coverage",
    "calculate_shannon_diversity",
    "calculate_simpson_diversity",
    "calculate_pielou_evenness",
    "compute_station_diversity_metrics",
    "analyze_prediction",
    "BioScanBiodiversityIntelligence",
]
