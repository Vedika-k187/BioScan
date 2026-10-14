"""
BioScan Prediction Engine
Package: src.prediction

Public Prediction API:
- BioScanPredictor: main prediction engine class
- predict_sequence: predict a single DNA sequence string
- predict_batch: predict a list of sequences or dictionaries
- predict_fasta: predict sequences from a FASTA file or string
- export_predictions_csv: export batch results to CSV
- parse_fasta: robust FASTA parser
- validate_dna_sequence: biological DNA QC and validation
"""

from .predictor import (
    BioScanPredictor,
    get_predictor,
    predict_sequence,
    predict_batch,
    predict_fasta,
    export_predictions_csv,
    validate_and_clean_sequence
)
from .input_validation import validate_dna_sequence
from .fasta_parser import parse_fasta, parse_fasta_file, parse_fasta_string
from .batch_predict import batch_results_to_dataframe, export_batch_results_csv
from .result_schema import format_prediction_result, format_qc_failure_result

__all__ = [
    "BioScanPredictor",
    "get_predictor",
    "predict_sequence",
    "predict_batch",
    "predict_fasta",
    "export_predictions_csv",
    "validate_dna_sequence",
    "validate_and_clean_sequence",
    "parse_fasta",
    "parse_fasta_file",
    "parse_fasta_string",
    "batch_results_to_dataframe",
    "export_batch_results_csv",
    "format_prediction_result",
    "format_qc_failure_result",
]
