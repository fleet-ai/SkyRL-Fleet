"""
Semantic ASCII encoder for arc-witness-envs puzzle games.

Converts 64×64 pixel frames into 16×16 ASCII boards where each symbol
represents a game element's ROLE rather than its raw color index.

All 13 Witness games share the same color palette (witness_grid.py),
so a single static mapping covers every game — no exploration needed.

Key design: color 11 (yellow) is used for THREE different things:
  - Cursor (player position): 5px cross, at path tip → '@'
  - Dots (mandatory waypoints): 5px cross, static → 'o'
  - Stars (pairing constraints): 13px diamond → '*'
We distinguish them by spatial context (path adjacency) and size.

Usage:
    from .semantic_ascii import encode_grid
    ascii_board = encode_grid(frame)  # frame: 64×64 numpy int array
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── Color constants ────────────────────────────────────────────────────
_YELLOW = 11      # CURSOR_COLOR / DOT_COLOR / STAR_COLOR
_BLUE = 9         # PATH_COLOR
_GREEN = 14       # START_COLOR
_STAR_PIXEL_THRESHOLD = 8  # Stars have 13px; dots/cursor have 5px

# ── Color palette from witness_grid.py ─────────────────────────────────
# Format: color_index → (role_name, ascii_symbol)
# NOTE: color 11 is handled specially — see _classify_yellow_block()

UNIVERSAL_COLOR_MAP: Dict[int, Tuple[str, str]] = {
    0:  ("cell",      "."),   # CELL_BG (white)
    1:  ("unused",    "."),   # light gray
    2:  ("unused",    "."),   # gray
    3:  ("background", " "),  # GRID_BG (dark gray)
    4:  ("eraser",    "Y"),   # ERASER_COLOR (near black)
    5:  ("grid",      "#"),   # GRID_LINE (black)
    6:  ("square_a",  "A"),   # SQUARE_A (magenta)
    7:  ("unused",    "."),   # light magenta
    8:  ("endpoint",  "E"),   # END_COLOR (red)
    9:  ("path",      "+"),   # PATH_COLOR (blue)
    10: ("square_b",  "B"),   # SQUARE_B (light blue)
    # 11: handled by _classify_yellow_block()
    12: ("triangle",  "T"),   # TRI_COLOR / SQUARE_C (orange)
    13: ("filter",    "F"),   # FILTER_COLOR (maroon)
    14: ("start",     "S"),   # START_COLOR (green)
    15: ("polyomino", "P"),   # POLY_COLOR (purple)
}

# Priority for non-yellow colors (yellow handled separately)
PRIORITY_COLORS: Dict[int, int] = {
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
    """Map a non-yellow color index to its ASCII symbol."""
    if color in UNIVERSAL_COLOR_MAP:
        return UNIVERSAL_COLOR_MAP[color][1]
    return "?"


def _find_cursor_block(grid: np.ndarray, block_size: int = 4) -> Optional[Tuple[int, int]]:
    """
    Find the block (row, col) containing the cursor.

    The cursor is always at the tip of the blue path (color 9).
    If no path exists (initial frame), it's near the green start (color 14).

    Returns (block_row, block_col) or None if no yellow found.
    """
    h, w = grid.shape
    bh, bw = h // block_size, w // block_size

    # Collect blocks containing each relevant color
    yellow_blocks: Set[Tuple[int, int]] = set()
    blue_blocks: Set[Tuple[int, int]] = set()
    green_blocks: Set[Tuple[int, int]] = set()

    for r in range(bh):
        for c in range(bw):
            patch = grid[r * block_size:(r + 1) * block_size,
                         c * block_size:(c + 1) * block_size]
            colors_present = set(int(x) for x in patch.ravel())
            if _YELLOW in colors_present:
                yellow_blocks.add((r, c))
            if _BLUE in colors_present:
                blue_blocks.add((r, c))
            if _GREEN in colors_present:
                green_blocks.add((r, c))

    if not yellow_blocks:
        return None

    # Find yellow blocks adjacent to blue (path tip)
    if blue_blocks:
        for yb in yellow_blocks:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                neighbor = (yb[0] + dr, yb[1] + dc)
                if neighbor in blue_blocks:
                    return yb

    # No blue path yet (initial frame): find yellow near green (start)
    if green_blocks:
        for yb in yellow_blocks:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                neighbor = (yb[0] + dr, yb[1] + dc)
                if neighbor in green_blocks:
                    return yb

    # Fallback: pick yellow block with fewest yellow pixels (cursor = smallest)
    best = None
    min_count = float('inf')
    for yb in yellow_blocks:
        patch = grid[yb[0] * block_size:(yb[0] + 1) * block_size,
                     yb[1] * block_size:(yb[1] + 1) * block_size]
        count = int(np.sum(patch == _YELLOW))
        if count < min_count:
            min_count = count
            best = yb
    return best


def _classify_yellow_block(
    patch: np.ndarray,
    block_row: int,
    block_col: int,
    cursor_block: Optional[Tuple[int, int]],
) -> str:
    """
    Classify a yellow-containing block as cursor, dot, or star.

    - Cursor (@): this block is the identified cursor block
    - Star (*): >8 yellow pixels in the block (13px diamond)
    - Dot (o): small yellow, not the cursor (mandatory waypoint)
    """
    if cursor_block and (block_row, block_col) == cursor_block:
        return "@"

    yellow_count = int(np.sum(patch == _YELLOW))
    if yellow_count > _STAR_PIXEL_THRESHOLD:
        return "*"

    return "o"


def _render_block(
    patch: np.ndarray,
    block_row: int,
    block_col: int,
    cursor_block: Optional[Tuple[int, int]],
) -> str:
    """
    Determine the display symbol for a 4×4 pixel block.

    Yellow (color 11) is classified contextually.
    Other colors use priority rendering.
    """
    flat = patch.ravel()
    has_yellow = _YELLOW in flat

    # Yellow takes highest priority — classify it
    if has_yellow:
        return _classify_yellow_block(patch, block_row, block_col, cursor_block)

    # Non-yellow: check priority colors
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

    # Pass 1: find cursor location
    cursor_block = _find_cursor_block(grid, block_size)

    # Pass 2: encode each block
    rows: List[str] = []
    for r in range(out_h):
        row_chars: List[str] = []
        for c in range(out_w):
            patch = grid[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size,
            ]
            row_chars.append(_render_block(patch, r, c, cursor_block))
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
        "@=cursor  o=dot(waypoint)  *=star(pair)  +=path  #=grid  .=cell  "
        "S=start  E=end  A/B=squares  T=triangle  P=polyomino  Y=eraser  F=filter"
    )
