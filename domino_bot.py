"""
Domino Dreams bot — main loop.

Loop: capture screen -> detect tiles -> find hand + legal moves ->
choose a move -> tap it -> wait for the clear/replace animation -> repeat.

This is intentionally simple and verbose about what it's doing (prints
each decision) so you can watch it reason during early testing rather
than running silently. Tighten TURN_DELAY_SEC and remove the prints once
you trust it.

Run:
    python domino_bot.py                 # live, taps for real
    python domino_bot.py --dry-run        # detect + decide, never taps
    python domino_bot.py --once           # single cycle then exit (debugging)
"""

from __future__ import annotations

import argparse
import time

from adb_capture import grab_frame, tap
from domino_vision import PipClusterDetector
from domino_game import to_placed_tiles, find_hand_tile, legal_moves, choose_move

TURN_DELAY_SEC = 1.2   # wait after a tap for the board's clear/replace animation
MAX_CONSECUTIVE_MISSES = 5  # stop after this many frames with no hand/move found
NO_HAND_RETRY_COUNT = 3     # re-capture this many times before giving up on a turn
NO_HAND_RETRY_DELAY_SEC = 0.5  # pause between those retries (rides out animations)


def run_once(detector: PipClusterDetector, dry_run: bool) -> bool:
    """Runs a single capture-decide-act cycle. Returns True if a move was
    made (or would have been, in dry-run), False if nothing playable was
    found this frame."""
    hand, board, w, h = None, [], 0, 0
    for attempt in range(1, NO_HAND_RETRY_COUNT + 1):
        frame = grab_frame()
        h, w = frame.shape[:2]
        tiles = detector.detect(frame)
        placed = to_placed_tiles(tiles)
        hand, board = find_hand_tile(placed, w, h)
        if hand is not None:
            break
        if attempt < NO_HAND_RETRY_COUNT:
            time.sleep(NO_HAND_RETRY_DELAY_SEC)  # likely mid-animation — ride it out

    if hand is None:
        print(f"  no hand tile found after {NO_HAND_RETRY_COUNT} attempts "
              "(if this keeps happening, check HAND_ZONE or for a popup covering the board)")
        return False

    print(f"  hand={hand.tile.value}  board tiles detected={len(board)}")

    moves = legal_moves(board, hand)
    if not moves:
        print("  no legal move for current hand — board tiles:",
              [t.tile.value for t in board])
        return False

    move = choose_move(moves, hand)
    print(f"  playing board tile {move.tile.value} at {move.tap_point}")

    if not dry_run:
        tap(*move.tap_point)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Domino Dreams autoplay loop")
    parser.add_argument("--dry-run", action="store_true",
                         help="Detect and decide but never actually tap the device")
    parser.add_argument("--once", action="store_true",
                         help="Run a single cycle then exit (for debugging)")
    args = parser.parse_args()

    detector = PipClusterDetector()
    misses = 0
    turn = 0

    while True:
        turn += 1
        print(f"[turn {turn}]")
        try:
            made_move = run_once(detector, args.dry_run)
        except Exception as e:  # noqa: BLE001 — keep the bot alive on a bad frame
            print(f"  error this cycle, skipping: {e}")
            made_move = False

        if args.once:
            break

        misses = 0 if made_move else misses + 1
        if misses >= MAX_CONSECUTIVE_MISSES:
            print(f"No playable move found for {MAX_CONSECUTIVE_MISSES} frames in a row — "
                  "stopping (level may be complete, a popup may be covering the board, "
                  "or detection needs recalibrating). Check debug_output.png.")
            break

        time.sleep(TURN_DELAY_SEC)


if __name__ == "__main__":
    main()
