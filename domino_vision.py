"""
Domino Dreams — Pip-First Clustering detector
===============================================

Why this approach: tiles in this game are shown in overlapping fans, so a
tile's cream body is often mostly covered by the tile in front of it — only
its pips (and maybe a sliver of body) are visible. Detecting the tile body
first and cropping it (the old approach) breaks under that occlusion.
Colored pips, however, are usually visible even when the body is stacked
behind another tile, so we detect pips first and reconstruct tiles from
them.

Pipeline:
  1. Detect every colored pip blob in the whole frame (broad HSV, any tile)
  2. Cluster nearby pips into "half-groups" (one die-face worth of pips,
     1-6 pips close together at a consistent local scale)
  3. Pair half-groups into tiles: two half-groups at a plausible
     center-to-center distance/scale become one tile's left/right side
  4. Blank-side inference: an unpaired half-group is checked for a
     cream-colored neighbor at a plausible partner location — if found,
     that side is a blank (0 pips) rather than "not visible"
  5. Confidence gate: a side is only accepted if its pip count is legal
     (0-6); oversized/undersized or unpaired-with-no-blank-evidence groups
     are reported as low confidence instead of being guessed at

Usage:
    from domino_vision import PipClusterDetector

    det = PipClusterDetector()
    tiles = det.detect(frame_bgr)
    for t in tiles:
        print(t.value, t.confident)

    debug_img = det.draw_debug(frame_bgr, tiles)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Tunables — calibrate against your own screenshots with --debug (see
# bottom of file). Defaults are rough guesses from a ~1600px-wide capture;
# rescale if your screenshots are a different resolution.
# --------------------------------------------------------------------------

# Broad HSV bands per pip color. Add/remove colors to match your skin.
PIP_COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red":    [(np.array([0, 90, 90]), np.array([8, 255, 255])),
               (np.array([170, 90, 90]), np.array([179, 255, 255]))],
    "orange": [(np.array([9, 120, 140]), np.array([20, 255, 255]))],
    "yellow": [(np.array([21, 120, 140]), np.array([33, 255, 255]))],
    "green":  [(np.array([40, 60, 60]), np.array([85, 255, 255]))],
    "blue":   [(np.array([95, 80, 70]), np.array([130, 255, 255]))],
    "purple": [(np.array([131, 40, 60]), np.array([160, 255, 255]))],
    "pink":   [(np.array([161, 40, 120]), np.array([169, 255, 255]))],
}

# Absolute pip blob area bounds in pixels^2. No tile-body scale reference
# exists in this approach, so these are absolute, not relative — tune to
# your resolution using --debug.
PIP_AREA_MIN = 150
PIP_AREA_MAX = 4000
PIP_MIN_CIRCULARITY = 0.6

# A color blob larger than PIP_AREA_MAX but smaller than this multiple of it
# is treated as "pip(s) merged with same-colored tile body" rather than
# plain background, and gets a second look via Hough circles (see below).
# Bigger than that, it's assumed to just be background/UI and is dropped.
MERGE_BLOB_MAX_FACTOR = 60

# Hough-circle fallback for color-collided pips (e.g. purple pips on a
# purple tile body, where the color mask can't separate pip from
# background). This leverages pip SIZE instead of color boundary: pips are
# small, roughly uniform circles, so searching for circles near the radius
# learned from unambiguous pips elsewhere in the same frame recovers them
# even when contour-based separation fails.
DEFAULT_PIP_RADIUS_GUESS = 18     # used only if zero unambiguous pips found
HOUGH_RADIUS_TOLERANCE_LOW = 0.65
HOUGH_RADIUS_TOLERANCE_HIGH = 1.35
HOUGH_PARAM1 = 80     # Canny high threshold used internally by HoughCircles
HOUGH_PARAM2 = 28     # accumulator threshold — lower finds more circles
                       # (and more false positives); tune with --debug

# Two pips belong to the same half-group (same die face) if their centers
# are closer than this many pip-diameters. Domino pip grids are tight
# (roughly a 2x2 or 3x2 arrangement within one half), so this should be a
# bit more than one pip-diameter but well under the gap to the next tile.
WITHIN_HALF_DIST_FACTOR = 2.4

# Two half-groups are the two sides of one tile if their centroid distance,
# measured in pip-diameters, falls in this range. Tune with --debug: draw
# a line between two halves you know belong to one tile and check the
# printed distance/diameter ratio.
PAIR_DIST_MIN_FACTOR = 1.3
PAIR_DIST_MAX_FACTOR = 4.2

# Cream/ivory tile-body color, used only for the lightweight blank-side
# check (not for primary detection).
CREAM_HSV_LOW = np.array([10, 10, 170])
CREAM_HSV_HIGH = np.array([30, 90, 255])
BLANK_SEARCH_RADIUS_FACTOR = 0.6   # search radius around a candidate blank
                                    # point, as a fraction of pair distance
BLANK_CREAM_FRACTION_MIN = 0.35    # min fraction of that search disk that
                                    # must be cream-colored to call it blank
BLANK_SEARCH_DIRECTIONS = 8        # candidate directions tried per
                                    # unpaired half-group (every 360/N deg)

MAX_PIPS_PER_SIDE = 6


@dataclass
class Pip:
    x: float
    y: float
    radius: float
    color: str


@dataclass
class HalfGroup:
    pips: list[Pip]
    centroid: tuple[float, float]
    mean_radius: float

    @property
    def count(self) -> int:
        return len(self.pips)


@dataclass
class DominoTile:
    side_a: HalfGroup | None          # None only for an inferred blank
    side_b: HalfGroup | None
    side_a_count: int
    side_b_count: int
    confident: bool
    inferred_blank_side: str | None = None   # "a" or "b" if that side had
                                              # no pips and was inferred
                                              # from cream-color evidence
    note: str = ""

    @property
    def value(self) -> tuple[int, int]:
        return tuple(sorted((self.side_a_count, self.side_b_count)))


class PipClusterDetector:
    def __init__(self, pip_ranges: dict | None = None):
        self.pip_ranges = pip_ranges or PIP_COLOR_RANGES

    # ---------------------------------------------------------------- #
    # Step 1: find every pip in the frame
    # ---------------------------------------------------------------- #
    def _find_pips(self, frame_bgr: np.ndarray) -> list[Pip]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pips: list[Pip] = []
        # Blobs too big to be a single pip but not huge enough to be plain
        # background — likely pip(s) color-merged with the tile body.
        # Revisited in pass 2 below via Hough circles.
        merged_blobs: list[tuple[str, np.ndarray, np.ndarray]] = []  # (color, contour, mask)

        # ---- Pass 1: normal color + contour + circularity detection ----
        for color_name, ranges in self.pip_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < PIP_AREA_MIN:
                    continue
                if area > PIP_AREA_MAX:
                    if area <= PIP_AREA_MAX * MERGE_BLOB_MAX_FACTOR:
                        merged_blobs.append((color_name, c, mask))
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < PIP_MIN_CIRCULARITY:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(c)
                pips.append(Pip(x=x, y=y, radius=radius, color=color_name))

        if not merged_blobs:
            return pips

        # ---- Pass 2: Hough-circle recovery inside merged/oversized blobs ----
        # Use the pip radius we're actually seeing elsewhere in this frame
        # as the search prior, rather than a hardcoded pixel value, so this
        # adapts to resolution/zoom automatically.
        if pips:
            median_r = statistics.median(p.radius for p in pips)
        else:
            median_r = DEFAULT_PIP_RADIUS_GUESS
        min_r = max(3, int(median_r * HOUGH_RADIUS_TOLERANCE_LOW))
        max_r = max(min_r + 1, int(median_r * HOUGH_RADIUS_TOLERANCE_HIGH))

        for color_name, c, mask in merged_blobs:
            x, y, w, h = cv2.boundingRect(c)
            pad = int(median_r)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            roi_gray = cv2.medianBlur(gray[y0:y1, x0:x1], 5)
            if roi_gray.size == 0:
                continue

            circles = cv2.HoughCircles(
                roi_gray, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=max(4, int(median_r * 1.3)),
                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is None:
                continue

            for cx, cy, r in circles[0]:
                gx, gy = cx + x0, cy + y0
                gxi, gyi = int(round(gx)), int(round(gy))
                if not (0 <= gyi < mask.shape[0] and 0 <= gxi < mask.shape[1]):
                    continue
                # Confirm the circle center is actually this pip color, not
                # a circular edge elsewhere in the padded ROI.
                if mask[gyi, gxi] == 0:
                    continue
                pips.append(Pip(x=float(gx), y=float(gy), radius=float(r), color=color_name))

        return pips

    # ---------------------------------------------------------------- #
    # Step 2: cluster pips into half-groups (union-find on proximity)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _cluster_halves(pips: list[Pip]) -> list[HalfGroup]:
        n = len(pips)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(pips[i].x - pips[j].x, pips[i].y - pips[j].y)
                avg_diam = (pips[i].radius + pips[j].radius)
                if avg_diam == 0:
                    continue
                if d < WITHIN_HALF_DIST_FACTOR * avg_diam:
                    union(i, j)

        groups: dict[int, list[Pip]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(pips[i])

        halves: list[HalfGroup] = []
        for members in groups.values():
            if len(members) > MAX_PIPS_PER_SIDE:
                continue  # almost certainly two merged halves / noise — drop
            cx = sum(p.x for p in members) / len(members)
            cy = sum(p.y for p in members) / len(members)
            mean_r = sum(p.radius for p in members) / len(members)
            halves.append(HalfGroup(pips=members, centroid=(cx, cy), mean_radius=mean_r))
        return halves

    # ---------------------------------------------------------------- #
    # Step 3: pair half-groups into tiles (greedy mutual-nearest-neighbor)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _pair_halves(halves: list[HalfGroup]) -> tuple[list[tuple[HalfGroup, HalfGroup]], list[HalfGroup]]:
        n = len(halves)
        candidates = []  # (distance, i, j)
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = halves[i], halves[j]
                d = math.hypot(hi.centroid[0] - hj.centroid[0], hi.centroid[1] - hj.centroid[1])
                diam = (hi.mean_radius + hj.mean_radius)
                if diam == 0:
                    continue
                ratio = d / diam
                if PAIR_DIST_MIN_FACTOR <= ratio <= PAIR_DIST_MAX_FACTOR:
                    candidates.append((d, i, j))

        candidates.sort(key=lambda t: t[0])
        used = set()
        pairs = []
        for d, i, j in candidates:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            pairs.append((halves[i], halves[j]))

        unpaired = [h for idx, h in enumerate(halves) if idx not in used]
        return pairs, unpaired

    # ---------------------------------------------------------------- #
    # Step 4: infer a blank (0-pip) partner for an unpaired half-group by
    # checking for cream tile color at a plausible partner location.
    # ---------------------------------------------------------------- #
    @staticmethod
    def _infer_blank_partner(frame_bgr: np.ndarray, half: HalfGroup) -> tuple[float, float] | None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        diam = half.mean_radius * 2
        pair_dist = diam * (PAIR_DIST_MIN_FACTOR + PAIR_DIST_MAX_FACTOR) / 2
        search_r = max(3, int(diam * BLANK_SEARCH_RADIUS_FACTOR))

        cx, cy = half.centroid
        best = None
        for k in range(BLANK_SEARCH_DIRECTIONS):
            theta = 2 * math.pi * k / BLANK_SEARCH_DIRECTIONS
            px = int(cx + pair_dist * math.cos(theta))
            py = int(cy + pair_dist * math.sin(theta))
            if not (0 <= px < w and 0 <= py < h):
                continue

            x0, x1 = max(0, px - search_r), min(w, px + search_r)
            y0, y1 = max(0, py - search_r), min(h, py + search_r)
            region = hsv[y0:y1, x0:x1]
            if region.size == 0:
                continue
            cream_mask = cv2.inRange(region, CREAM_HSV_LOW, CREAM_HSV_HIGH)
            frac = float(np.count_nonzero(cream_mask)) / cream_mask.size
            if frac >= BLANK_CREAM_FRACTION_MIN:
                if best is None or frac > best[0]:
                    best = (frac, (px, py))

        return best[1] if best else None

    # ---------------------------------------------------------------- #
    # Public entry point
    # ---------------------------------------------------------------- #
    def detect(self, frame_bgr: np.ndarray) -> list[DominoTile]:
        pips = self._find_pips(frame_bgr)
        halves = self._cluster_halves(pips)
        pairs, unpaired = self._pair_halves(halves)

        tiles: list[DominoTile] = []

        for a, b in pairs:
            count_a, count_b = a.count, b.count
            confident = (0 <= count_a <= MAX_PIPS_PER_SIDE) and (0 <= count_b <= MAX_PIPS_PER_SIDE)
            tiles.append(
                DominoTile(
                    side_a=a, side_b=b,
                    side_a_count=count_a, side_b_count=count_b,
                    confident=confident,
                )
            )

        for half in unpaired:
            blank_point = self._infer_blank_partner(frame_bgr, half)
            if blank_point is not None:
                blank_group = HalfGroup(pips=[], centroid=blank_point, mean_radius=half.mean_radius)
                tiles.append(
                    DominoTile(
                        side_a=half, side_b=blank_group,
                        side_a_count=half.count, side_b_count=0,
                        confident=(half.count <= MAX_PIPS_PER_SIDE),
                        inferred_blank_side="b",
                        note="partner side inferred blank from cream color, not pips",
                    )
                )
            else:
                tiles.append(
                    DominoTile(
                        side_a=half, side_b=None,
                        side_a_count=half.count, side_b_count=-1,
                        confident=False,
                        note="no partner found — other side likely fully occluded",
                    )
                )

        return tiles

    # ---------------------------------------------------------------- #
    # Debug visualization
    # ---------------------------------------------------------------- #
    def draw_debug(self, frame_bgr: np.ndarray, tiles: list[DominoTile]) -> np.ndarray:
        out = frame_bgr.copy()
        for tile in tiles:
            color = (0, 255, 0) if tile.confident else (0, 0, 255)
            for side, label_count in ((tile.side_a, tile.side_a_count), (tile.side_b, tile.side_b_count)):
                if side is None:
                    continue
                cx, cy = int(side.centroid[0]), int(side.centroid[1])
                r = max(6, int(side.mean_radius * 2.2))
                cv2.circle(out, (cx, cy), r, color, 2)
                cv2.putText(out, str(label_count), (cx - 8, cy - r - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                for p in side.pips:
                    cv2.circle(out, (int(p.x), int(p.y)), int(p.radius), (255, 255, 0), 1)
            if tile.side_a is not None and tile.side_b is not None:
                pa = (int(tile.side_a.centroid[0]), int(tile.side_a.centroid[1]))
                pb = (int(tile.side_b.centroid[0]), int(tile.side_b.centroid[1]))
                cv2.line(out, pa, pb, color, 1)
        return out


# --------------------------------------------------------------------------
# CLI test / calibration helper.
#   python domino_vision.py screenshot.png
#   python domino_vision.py                        # grab a live frame over ADB
#   python domino_vision.py screenshot.png --debug  # also save pip mask
#   python domino_vision.py --debug
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Domino pip-cluster detector")
    parser.add_argument("image", nargs="?", help="Screenshot path (omit to capture via ADB)")
    parser.add_argument("--debug", action="store_true",
                        help="Save per-color pip masks to ./debug/")
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Could not read image: {args.image}")
    else:
        from adb_capture import grab_frame
        img = grab_frame()

    detector = PipClusterDetector()
    found = detector.detect(img)
    print(f"Found {len(found)} tile(s):")
    for t in found:
        status = "OK" if t.confident else "LOW-CONFIDENCE"
        print(f"  value={t.value}  [{status}]" + (f"  note={t.note}" if t.note else ""))

    cv2.imwrite("debug_output.png", detector.draw_debug(img, found))
    print("Saved overlay to debug_output.png (green=confident, red=low-confidence, "
          "yellow circles=individual pips)")

    if args.debug:
        os.makedirs("debug", exist_ok=True)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for color_name, ranges in PIP_COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, low, high)
            cv2.imwrite(f"debug/pip_mask_{color_name}.png", mask)
        print("Saved debug/pip_mask_<color>.png for each configured pip color")
