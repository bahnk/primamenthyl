"""
Helpers for loading FASTA sequences.
"""

from __future__ import annotations

from pathlib import Path


def load_fasta(fasta_path: str | Path) -> str:
    """
    Load a FASTA file into an uppercase sequence string.

    Header lines beginning with `>` are ignored and sequence
    lines are concatenated in file order.

    Args:
        fasta_path (str | Path):
            Path to a FASTA file.

    Returns:
        str:
            The concatenated FASTA sequence.

    Raises:
        FileNotFoundError:
            Raised when `fasta_path` does not exist.
        ValueError:
            Raised when the FASTA file does not contain any
            sequence lines.
    """

    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {path}")

    sequence_parts: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            sequence_parts.append(stripped.upper())

    if not sequence_parts:
        raise ValueError(f"FASTA file does not contain a sequence: {path}")

    return "".join(sequence_parts)
