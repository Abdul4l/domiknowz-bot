"""
Domino Dreams — game state & move selection
=============================================

Turns raw DominoTile detections (from domino_vision.PipClusterDetector)
into a decision: which board tile to tap next, and where.

Rules this encodes (as described by the user, unverified against live
play yet — adjust the two flagged assumptions below once tested):
  - You have one active "hand" tile. A board tile is a legal match if
    EITHER of its exposed pip counts equals EITHER of the hand tile's
    pip counts. (Assumption: not confirmed which specific sides count.)
  - Playing a board tile clears it and becomes your new hand tile.
  - Stacked tiles: a fully-buried tile produces no pips and is simply
    never detected, so every confidently-detected board tile is already
    "clear to tap first" by construction — no extra stacking logic needed.
  - "Keep flexible ends open" (sequencing strategy) is NOT implemented
    here — choose_move() currently just picks a legal match; see the
    TODO in choose_move() for where to add lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

from domino_vision import DominoTile


# Fractional (x0, y0, x1, y1) screen zone where the single active "hand"
# tile sits: the lone domino next to the face-down draw pile at the
# bottom of the board, separate from both fans. (Earlier revision wrongly
# used where the in-game pointing-hand GRAPHIC touched — that graphic
# actually highlights a suggested board move, not the hand slot.)
HAND_ZONE = (0.55, 0.72, 0.82, 1.0)


@dataclass
class PlacedTile:
    tile: DominoTile
    tap_point: tuple[int, int]

    @property
    def sides(self) -> set[int]:
        return {self.tile.side_a_count, self.tile.side_b_count}


def to_placed_tiles(tiles: list[DominoTile]) -> list[PlacedTile]:
    """Wraps confident DominoTile detections with a tap point (the
    midpoint between its two side centroids, in frame pixel coordinates —
    the same coordinate space the screenshot was captured in)."""
    placed = []
    for t in tiles:
        if not t.confident or t.side_a is None or t.side_b is None:
            continue
        ax, ay = t.side_a.centroid
        bx, by = t.side_b.centroid
        tap_point = (int((ax + bx) / 2), int((ay + by) / 2))
        placed.append(PlacedTile(tile=t, tap_point=tap_point))
    return placed


def find_hand_tile(
    placed_tiles: list[PlacedTile],
    frame_w: int,
    frame_h: int,
    hand_zone: tuple[float, float, float, float] = HAND_ZONE,
) -> tuple[PlacedTile | None, list[PlacedTile]]:
    """Splits detections into (hand_tile, board_tiles) using hand_zone.
    Returns (None, placed_tiles) if nothing fell inside the zone — caller
    should treat that as "couldn't find a hand this frame" rather than
    guessing."""
    fx0, fy0, fx1, fy1 = hand_zone
    x0, y0 = fx0 * frame_w, fy0 * frame_h
    x1, y1 = fx1 * frame_w, fy1 * frame_h

    hand, board = None, []
    for p in placed_tiles:
        px, py = p.tap_point
        if x0 <= px <= x1 and y0 <= py <= y1 and hand is None:
            hand = p
        else:
            board.append(p)
    return hand, board


def legal_moves(board_tiles: list[PlacedTile], hand: PlacedTile) -> list[PlacedTile]:
    """Board tiles sharing at least one pip count with the hand tile."""
    return [t for t in board_tiles if t.sides & hand.sides]


def choose_move(moves: list[PlacedTile], hand: PlacedTile) -> PlacedTile | None:
    """Picks which legal move to play. Currently naive: first match found.

    TODO (strategy upgrade, not yet implemented): "keep flexible ends
    open" means, among legal moves, prefer the one whose NON-matching side
    is more common across the remaining board tiles (keeps your next hand
    useful) rather than a rare value that's more likely to dead-end you.
    That needs the full board_tiles list passed in to compute frequency —
    left as a follow-up once basic play is confirmed working.
    """
    if not moves:
        return None
    return moves[0]