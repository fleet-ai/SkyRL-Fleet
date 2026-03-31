"""
Semantic ASCII encoder for arc-witness-envs puzzle games.

Converts 64×64 pixel frames into 16×16 ASCII boards where each symbol
represents a game element's ROLE rather than its raw color index.

All 13 Witness games share the same color palette (witness_grid.py),
so a single static mapping covers every game — no exploration or
per-game discovery needed.

Usage:
    from .semantic_ascii import encode_grid
    ascii_board = encode_grid(frame)  # frame: 64×64 numpy int array
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# ── Color palette from witness_grid.py ─────────────────────────────────
# Format: color_index → (role_name, ascii_symbol)

UNIVERSAL_COLOR_MAP: Dict[int, Tuple[str, str]] = {
    0:  ("cell",      "."),   # CELL_BG (white) — empty cell interior
    1:  ("unused",    "."),   # light gray — not used in current games
    2:  ("unused",    "."),   # gray — not used in current games
    3:  ("background", " "),  # GRID_BG (dark gray) — panel background
    4:  ("eraser",    "Y"),   # ERASER_COLOR (near black) — elimination symbol
    5:  ("grid",      "#"),   # GRID_LINE (black) — grid lines / edges / nodes
    6:  ("square_a",  "A"),   # SQUARE_A (magenta) — colored square constraint A
    7:  ("unused",    "."),   # light magenta — not used in current games
    8:  ("endpoint",  "E"),   # END_COLOR (red) — end point / error flash
    9:  ("path",      "+"),   # PATH_COLOR (blue) — drawn path / trail
    10: ("square_b",  "B"),   # SQUARE_B (light blue) — colored square constraint B
    11: ("cursor",    "@"),   # CURSOR_COLOR (yellow) — cursor / dots / stars
    12: ("triangle",  "T"),   # TRI_COLOR / SQUARE_C (orange) — triangles
    13: ("filter",    "F"),   # FILTER_COLOR (maroon) — color filter marker
    14: ("start",     "S"),   # START_COLOR (green) — start point / success flash
    15: ("polyomino", "P"),   # POLY_COLOR (purple) — polyomino / wrap marker
}

# Priority order for rendering: higher priority colors show even if
# they are a minority in the 4×4 block. This prevents small but
# important symbols (1px cursor, 5px dots) from being hidden.
PRIORITY_COLORS: Dict[int, int] = {
    11: 6,   # cursor (@) — highest, always show agent position
    14: 5,   # start (S)
    8:  4,   # end (E)
    6:  3,   # square_a (A)
    10: 3,   # square_b (B)
    12: 3,   # triangle (T)
    15: 3,   # polyomino (P)
    4:  3,   # eraser (Y)
    13: 3,   # filter (F)
    9:  2,   # path (+)
}


def _get_symbol(color: int) -> str:
    """Map a color index to its ASCII symbol."""
    if color in UNIVERSAL_COLOR_MAP:
        return UNIVERSAL_COLOR_MAP[color][1]
    return "?"


def _render_block(patch: np.ndarray) -> str:
    """
    Determine the display symbol for a 4×4 pixel block.

    Uses priority rendering: if any high-priority color is present
    (even as 1 pixel out of 16), it wins. Otherwise majority vote.
    """
    flat = patch.ravel()

    # Check priority colors (highest first)
    best_priority = -1
    best_color = -1
    for pixel in flat:
        p = PRIORITY_COLORS.get(int(pixel), -1)
        if p > best_priority:
            best_priority = p
            best_color = int(pixel)

    if best_priority > 0:
        return _get_symbol(best_color)

    # Fallback: majority vote
    counts = Counter(int(x) for x in flat)
    majority_color = counts.most_common(1)[0][0]
    return _get_symbol(majority_color)


def encode_grid(grid: np.ndarray, block_size: int = 4) -> str:
    """
    Convert a 64×64 integer frame to a 16×16 semantic ASCII board.

    Parameters
    ----------
    grid : np.ndarray
        64×64 array of color indices (0-15).
    block_size : int
        Downsampling factor (default 4 → 16×16 output).

    Returns
    -------
    str
        Multi-line ASCII board with row/col headers and legend.
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    out_h, out_w = h // block_size, w // block_size

    # Build ASCII rows
    rows: List[str] = []
    for r in range(out_h):
        row_chars: List[str] = []
        for c in range(out_w):
            patch = grid[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size,
            ]
            row_chars.append(_render_block(patch))
        rows.append(row_chars)

    # Format with headers
    col_header = "   " + " ".join(f"{c:2d}" for c in range(out_w))
    lines = [col_header]
    for r, row_chars in enumerate(rows):
        line = f"{r:2d} " + " ".join(f" {ch}" for ch in row_chars)
        lines.append(line)

    # Append legend
    lines.append("")
    lines.append(build_legend())

    return "\n".join(lines)


def build_legend() -> str:
    """Return a compact legend explaining the ASCII symbols."""
    return (
        "@=cursor  +=path  #=grid  .=cell  S=start  E=end  "
        "A/B=squares  T=triangle  P=polyomino  Y=eraser  F=filter"
    )
