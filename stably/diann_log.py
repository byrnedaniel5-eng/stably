"""Parse a DIA-NN log file to capture the upstream provenance needed for
HC-INTER-03 (peptide/protein identifiers referenced to a specific database
version) and HC-FDR-06 (MBR-transferred vs MS2-confirmed IDs).

The parser is intentionally permissive: if fields cannot be located it returns
None for those fields rather than raising, so a partial manifest is still
better than no manifest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def find_diann_log(data_file: str | Path) -> Optional[Path]:
    """
    Heuristic search for a DIA-NN log near `data_file`.

    Looks in:
      1. The directory of `data_file` itself.
      2. Immediate subdirectories (one level deep).
      3. Sibling directories (one level up, one level down).

    Returns the first match or None.
    """
    data_path = Path(data_file).resolve()
    candidates = []

    if data_path.parent.exists():
        candidates.extend(sorted(data_path.parent.glob("*.log.txt")))
        for child in sorted(data_path.parent.iterdir()):
            if child.is_dir():
                candidates.extend(sorted(child.glob("*.log.txt")))

    parent = data_path.parent.parent
    if parent.exists():
        candidates.extend(sorted(parent.glob("*.log.txt")))

    # De-duplicate while preserving order
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and _looks_like_diann_log(resolved):
            return resolved
    return None


def _looks_like_diann_log(path: Path) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            head = f.read(512)
    except OSError:
        return False
    return head.lstrip().startswith("DIA-NN")


_FASTA_RE = re.compile(r"--fasta\s+(\S+)")
_LIB_RE = re.compile(r"--lib\s+(\S+)")
_QVALUE_RE = re.compile(r"--qvalue\s+(\S+)")
_THREADS_RE = re.compile(r"--threads\s+(\d+)")
_REANALYSE_RE = re.compile(r"--reanalyse(?!\w)")
_MATRICES_RE = re.compile(r"--matrices(?!\w)")
_MBR_LINE_RE = re.compile(r"MBR\s+enabled", re.IGNORECASE)
_VERSION_RE = re.compile(r"^DIA-NN\s+(\S+)(?:\s+(\S+))?", re.MULTILINE)
_COMPILED_RE = re.compile(r"^Compiled on\s+(.+)$", re.MULTILINE)
_FASTA_DATE_TAIL_RE = re.compile(r"_([A-Z][a-z]{2}\d{2})\.", re.IGNORECASE)


def parse_diann_log(path: str | Path, head_chars: int = 8192) -> dict:
    """
    Parse the header region of a DIA-NN log.

    Only the first `head_chars` are read, which is enough for every field of
    interest on a standard DIA-NN log. Returns a dict with keys:
        diann_version, diann_edition, compile_date, command_line,
        fasta_path, fasta_name, fasta_release_tag,
        library_path, qvalue, mbr_enabled, reanalyse,
        matrices, threads, log_file.
    Missing fields are None.
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        head = f.read(head_chars)

    version_match = _VERSION_RE.search(head)
    diann_version = version_match.group(1) if version_match else None
    diann_edition = version_match.group(2) if version_match and version_match.group(2) else None

    compiled_match = _COMPILED_RE.search(head)
    compile_date = compiled_match.group(1).strip() if compiled_match else None

    command_line = _extract_command_line(head)

    fasta_match = _FASTA_RE.search(head)
    fasta_path = fasta_match.group(1) if fasta_match else None
    fasta_name = Path(fasta_path).name if fasta_path else None
    fasta_tag_match = _FASTA_DATE_TAIL_RE.search(fasta_name or "")
    fasta_release_tag = fasta_tag_match.group(1) if fasta_tag_match else None

    lib_match = _LIB_RE.search(head)
    library_path = lib_match.group(1) if lib_match else None

    qvalue_match = _QVALUE_RE.search(head)
    qvalue = float(qvalue_match.group(1)) if qvalue_match else None

    threads_match = _THREADS_RE.search(head)
    threads = int(threads_match.group(1)) if threads_match else None

    reanalyse = bool(_REANALYSE_RE.search(head))
    matrices = bool(_MATRICES_RE.search(head))
    mbr_enabled = bool(_MBR_LINE_RE.search(head))

    return {
        'diann_version': diann_version,
        'diann_edition': diann_edition,
        'compile_date': compile_date,
        'command_line': command_line,
        'fasta_path': fasta_path,
        'fasta_name': fasta_name,
        'fasta_release_tag': fasta_release_tag,
        'library_path': library_path,
        'qvalue': qvalue,
        'mbr_enabled': mbr_enabled,
        'reanalyse': reanalyse,
        'matrices': matrices,
        'threads': threads,
        'log_file': str(path),
    }


def _extract_command_line(head: str) -> Optional[str]:
    """The DIA-NN command line is the first line containing --fasta or --lib."""
    for line in head.splitlines():
        if '--fasta' in line or '--lib' in line:
            return line.strip()
    return None
