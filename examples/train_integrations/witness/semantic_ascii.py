"""
Semantic ASCII encoder for arc-witness-envs puzzle games.

Converts 64×64 pixel frames into game-aligned ASCII boards where each
character maps 1:1 to a game element (node, edge, or cell).

Key design: auto-detect the game's grid structure from pixel patterns,
then output at native game resolution. A 3×3 cell game outputs a 7×7
ASCII grid (not 16×16), so one player move = a predictable shift in
the ASCII view.

Usage:
    from .semantic_ascii import encode_grid
    ascii_board = encode_grid(frame)  # frame: 64×64 numpy int array
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Color constants (from witness_grid.py) ─────────────────────────────
_BG = 3         # GRID_BG (dark gray panel background)
_GRID = 5       # GRID_LINE (black)
_CELL = 0       # CELL_BG (white)
_BLUE = 9       # PATH_COLOR
_YELLOW = 11    # CURSOR_COLOR / DOT_COLOR / STAR_COLOR
_GREEN = 14     # START_COLOR
_RED = 8        # END_COLOR
_MAGENTA = 6    # SQUARE_A
_LIGHT_BLUE = 10  # SQUARE_B
_ORANGE = 12    # TRI_COLOR / SQUARE_C
_PURPLE = 15    # POLY_COLOR
_NEAR_BLACK = 4 # ERASER_COLOR
_MAROON = 13    # FILTER_COLOR

_STAR_PIXEL_THRESHOLD = 8  # Stars have 13px diamond; dots/cursor have 5px cross


# ══════════════════════════════════════════════════════════════════════
# Grid detection
# ══════════════════════════════════════════════════════════════════════

def _detect_grid_nodes(grid: np.ndarray) -> Optional[Tuple[List[int], List[int]]]:
    """
    Detect game node positions from the 64×64 frame.

    Nodes sit on grid line intersections. At a node's x-coordinate,
    the entire column (within the grid area) is black (color 5) because
    of the vertical edges connecting nodes. Cell columns have far fewer
    black pixels (only at horizontal edge rows).

    Returns (node_xs, node_ys) or None if detection fails.
    """
    h, w = grid.shape

    # Find grid area bounds (exclude dark gray background margin)
    non_bg_mask = grid != _BG
    col_has_content = np.any(non_bg_mask, axis=0)
    row_has_content = np.any(non_bg_mask, axis=1)

    col_indices = np.where(col_has_content)[0]
    row_indices = np.where(row_has_content)[0]

    if len(col_indices) < 3 or len(row_indices) < 3:
        return None

    min_x, max_x = int(col_indices[0]), int(col_indices[-1])
    min_y, max_y = int(row_indices[0]), int(row_indices[-1])
    grid_h = max_y - min_y + 1
    grid_w = max_x - min_x + 1

    # Count black pixels per column within grid area
    grid_region = grid[min_y:max_y + 1, :]
    col_black = np.sum(grid_region == _GRID, axis=0)  # shape: (64,)

    # Node columns have many more black pixels than cell columns
    # Threshold: at least 40% of grid height
    threshold = grid_h * 0.4
    candidate_xs = [x for x in range(min_x, max_x + 1) if col_black[x] >= threshold]

    # Cluster consecutive x's into single positions (nodes are 1px wide)
    node_xs = _cluster_positions(candidate_xs)

    # Same for rows
    grid_region_t = grid[:, min_x:max_x + 1]
    row_black = np.sum(grid_region_t == _GRID, axis=1)  # shape: (64,)
    threshold_h = grid_w * 0.4
    candidate_ys = [y for y in range(min_y, max_y + 1) if row_black[y] >= threshold_h]
    node_ys = _cluster_positions(candidate_ys)

    # Sanity check: need at least 2 nodes in each direction
    if len(node_xs) < 2 or len(node_ys) < 2:
        return None

    return node_xs, node_ys


def _cluster_positions(positions: List[int], gap: int = 2) -> List[int]:
    """Cluster consecutive positions into single representative points."""
    if not positions:
        return []
    clusters: List[List[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    # Return the center of each cluster
    return [sum(c) // len(c) for c in clusters]


# ══════════════════════════════════════════════════════════════════════
# Element reading (node / edge / cell)
# ══════════════════════════════════════════════════════════════════════

def _sample_area(grid: np.ndarray, cx: int, cy: int, radius: int = 2) -> np.ndarray:
    """Get pixels in a small area around (cx, cy), clipped to frame bounds."""
    h, w = grid.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    return grid[y0:y1, x0:x1]


def _read_node(grid: np.ndarray, nx: int, ny: int) -> str:
    """Determine what's at a grid node position."""
    area = _sample_area(grid, nx, ny, radius=2)
    flat = area.ravel()
    colors = set(int(c) for c in flat)

    # Priority: cursor > start > end > dot > regular node
    if _YELLOW in colors:
        # Could be cursor or dot — check if adjacent to path or start
        if _BLUE in colors or _GREEN in colors:
            return "@"  # cursor (at path tip or start)
        # Check broader area for path adjacency
        wide = _sample_area(grid, nx, ny, radius=4)
        wide_colors = set(int(c) for c in wide.ravel())
        if _BLUE in wide_colors or _GREEN in wide_colors:
            return "@"
        return "o"  # dot (waypoint)
    if _GREEN in colors:
        return "S"
    if _RED in colors:
        return "E"
    if _BLUE in colors:
        return "+"  # path passes through this node
    return "#"


def _read_h_edge(grid: np.ndarray, x1: int, x2: int, y: int) -> str:
    """Read a horizontal edge between two nodes at (x1,y) and (x2,y)."""
    mid_x = (x1 + x2) // 2
    area = _sample_area(grid, mid_x, y, radius=1)
    flat = area.ravel()
    colors = set(int(c) for c in flat)

    if _BLUE in colors:
        return "+"  # path drawn on this edge
    if _BG in colors and _GRID not in colors:
        return " "  # breakpoint (gap)
    if _GRID in colors:
        return "-"  # normal grid line
    return " "


def _read_v_edge(grid: np.ndarray, x: int, y1: int, y2: int) -> str:
    """Read a vertical edge between two nodes at (x,y1) and (x,y2)."""
    mid_y = (y1 + y2) // 2
    area = _sample_area(grid, x, mid_y, radius=1)
    flat = area.ravel()
    colors = set(int(c) for c in flat)

    if _BLUE in colors:
        return "+"  # path drawn on this edge
    if _BG in colors and _GRID not in colors:
        return " "  # breakpoint
    if _GRID in colors:
        return "|"  # normal grid line
    return " "


def _read_cell(grid: np.ndarray, node_xs: List[int], node_ys: List[int],
               col: int, row: int) -> str:
    """Read the content of a cell (area between 4 surrounding nodes)."""
    # Cell center is midpoint between the 4 surrounding nodes
    cx = (node_xs[col] + node_xs[col + 1]) // 2
    cy = (node_ys[row] + node_ys[row + 1]) // 2

    # Sample a small area around cell center
    cell_half = max(1, (node_xs[col + 1] - node_xs[col]) // 3)
    area = _sample_area(grid, cx, cy, radius=cell_half)
    flat = area.ravel()

    # Count each color
    counts = Counter(int(c) for c in flat)

    # Check for constraint symbols (by priority)
    if _YELLOW in counts:
        yellow_n = counts[_YELLOW]
        if yellow_n > _STAR_PIXEL_THRESHOLD:
            return "*"  # star (large yellow diamond)
        return "o"  # dot in cell area (unusual but possible)
    if _MAGENTA in counts:
        return "A"  # square constraint A
    if _LIGHT_BLUE in counts:
        return "B"  # square constraint B
    if _ORANGE in counts:
        return "T"  # triangle
    if _PURPLE in counts:
        return "P"  # polyomino
    if _NEAR_BLACK in counts:
        return "Y"  # eraser
    if _MAROON in counts:
        return "F"  # filter
    return "."  # empty cell


# ══════════════════════════════════════════════════════════════════════
# Main encoder
# ══════════════════════════════════════════════════════════════════════

def encode_grid(grid: np.ndarray) -> str:
    """
    Convert a 64×64 integer frame to a game-aligned semantic ASCII board.

    Auto-detects the game grid structure, then outputs at native resolution.
    Falls back to fixed 16×16 block encoding if detection fails.
    """
    grid = np.asarray(grid, dtype=np.int32)
    result = _detect_grid_nodes(grid)

    if result is None:
        return _fallback_encode(grid)

    node_xs, node_ys = result
    cols = len(node_xs) - 1  # cell columns
    rows = len(node_ys) - 1  # cell rows

    lines: List[str] = []

    # Column header (node indices)
    header = "  " + " ".join(f"{c}" for c in range(cols + 1))
    lines.append(header)

    for r in range(rows + 1):
        # Node row
        parts: List[str] = []
        for c in range(cols + 1):
            parts.append(_read_node(grid, node_xs[c], node_ys[r]))
            if c < cols:
                parts.append(_read_h_edge(grid, node_xs[c], node_xs[c + 1], node_ys[r]))
        lines.append(f"{r} " + " ".join(parts))

        # Cell row (between this node row and the next)
        if r < rows:
            parts = []
            for c in range(cols + 1):
                parts.append(_read_v_edge(grid, node_xs[c], node_ys[r], node_ys[r + 1]))
                if c < cols:
                    parts.append(_read_cell(grid, node_xs, node_ys, c, r))
            lines.append("  " + " ".join(parts))

    lines.append("")
    lines.append(build_legend())

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Fallback: fixed 16×16 block encoding (original approach)
# ══════════════════════════════════════════════════════════════════════

# Color map for fallback mode
_FALLBACK_MAP: Dict[int, str] = {
    0: ".", 1: ".", 2: ".", 3: " ", 4: "Y", 5: "#",
    6: "A", 7: ".", 8: "E", 9: "+", 10: "B", 11: "@",
    12: "T", 13: "F", 14: "S", 15: "P",
}


def encode_grid_fixed(grid: np.ndarray, block_size: int = 4) -> str:
    """Fixed 16×16 block encoding (obs_mode='ascii')."""
    return _fallback_encode(grid, block_size)


def _fallback_encode(grid: np.ndarray, block_size: int = 4) -> str:
    """Fixed 16×16 block encoding when grid detection fails."""
    h, w = grid.shape
    out_h, out_w = h // block_size, w // block_size

    lines: List[str] = []
    header = "   " + " ".join(f"{c:2d}" for c in range(out_w))
    lines.append(header)

    for r in range(out_h):
        row_chars: List[str] = []
        for c in range(out_w):
            patch = grid[r * block_size:(r + 1) * block_size,
                         c * block_size:(c + 1) * block_size]
            flat = patch.ravel()
            counts = Counter(int(x) for x in flat)
            majority = counts.most_common(1)[0][0]
            row_chars.append(_FALLBACK_MAP.get(majority, "?"))
        lines.append(f"{r:2d} " + " ".join(f" {ch}" for ch in row_chars))

    lines.append("")
    lines.append(build_legend())
    return "\n".join(lines)


def build_legend() -> str:
    """Return a compact legend explaining the ASCII symbols."""
    return (
        "@=cursor  o=dot(waypoint)  *=star(pair)  +=path  #=node  "
        "-|=edge  .=cell  S=start  E=end  A/B=squares  "
        "T=triangle  P=polyomino  Y=eraser  F=filter"
    )
