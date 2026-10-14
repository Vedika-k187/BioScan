"""
src/prediction/fasta_parser.py
Phase 8: BioScan Robust FASTA Parser

Parses single and multi-sequence FASTA files and strings.
Handles:
- Standard FASTA headers (>id [description])
- Multi-line sequence folding
- Empty lines, trailing whitespace
- Malformed inputs (e.g. sequence without header, empty headers, empty files)
- Preserves sequence identifiers and exact order
"""

import os
from typing import List, Dict, Tuple, Union, Optional


def parse_fasta_string(fasta_text: str) -> List[Dict[str, str]]:
    """
    Parse a FASTA formatted string into a list of sequence dictionaries.

    Returns:
    - List of dicts: [{"id": str, "description": str, "sequence": str}, ...]

    Raises:
    - ValueError if the input is empty or malformed FASTA.
    """
    if fasta_text is None or not isinstance(fasta_text, str):
        raise ValueError("FASTA input must be a non-null string.")

    cleaned_text = fasta_text.strip()
    if len(cleaned_text) == 0:
        raise ValueError("FASTA input is empty.")

    lines = cleaned_text.splitlines()
    records: List[Dict[str, str]] = []
    
    current_id: Optional[str] = None
    current_desc: str = ""
    current_seq_parts: List[str] = []
    line_number = 0

    for raw_line in lines:
        line_number += 1
        line = raw_line.strip()
        if not line:
            continue  # ignore blank lines

        if line.startswith(">"):
            # If we were already collecting a sequence, save it
            if current_id is not None:
                full_seq = "".join(current_seq_parts)
                if not full_seq:
                    raise ValueError(f"FASTA header '>{current_id}' at line {line_number} has no associated sequence.")
                records.append({
                    "id": current_id,
                    "description": current_desc,
                    "sequence": full_seq
                })

            header_content = line[1:].strip()
            if not header_content:
                raise ValueError(f"Empty FASTA header detected at line {line_number} ('>' with no identifier).")

            parts = header_content.split(maxsplit=1)
            current_id = parts[0]
            current_desc = parts[1] if len(parts) > 1 else ""
            current_seq_parts = []
        else:
            if current_id is None:
                raise ValueError(
                    f"Malformed FASTA at line {line_number}: sequence content found before any header line ('{line[:30]}...')."
                )
            current_seq_parts.append(line)

    # Save the final record
    if current_id is not None:
        full_seq = "".join(current_seq_parts)
        if not full_seq:
            raise ValueError(f"Final FASTA header '>{current_id}' has no associated sequence.")
        records.append({
            "id": current_id,
            "description": current_desc,
            "sequence": full_seq
        })
    else:
        raise ValueError("No valid FASTA headers (lines starting with '>') found in input.")

    if not records:
        raise ValueError("FASTA input contained no valid sequence records.")

    return records


def parse_fasta_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parse a FASTA file from a filesystem path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return parse_fasta_string(content)


def parse_fasta(source: Union[str, os.PathLike]) -> List[Dict[str, str]]:
    """
    Convenience function: automatically distinguishes between a file path and a raw FASTA string.
    """
    if isinstance(source, (str, os.PathLike)) and os.path.exists(str(source)):
        return parse_fasta_file(str(source))
    elif isinstance(source, str):
        return parse_fasta_string(source)
    else:
        raise ValueError(f"Unsupported FASTA input type: {type(source)}")


def validate_fasta_format(text_or_path: Union[str, os.PathLike]) -> Tuple[bool, List[str], int]:
    """
    Validate FASTA format without raising exceptions.
    Returns: (is_valid, error_messages, sequence_count)
    """
    try:
        records = parse_fasta(text_or_path)
        return True, [], len(records)
    except Exception as e:
        return False, [str(e)], 0
