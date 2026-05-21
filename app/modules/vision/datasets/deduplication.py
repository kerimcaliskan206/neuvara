"""
Perceptual hash-based duplicate detection.

Two strategies are combined:
  1. Exact deduplication  — SHA-256 content hash. Zero cost to check.
  2. Near-duplicate detection — difference hash (dhash). Catches resized,
     lightly compressed, or slightly cropped copies of the same image.

dhash algorithm (64-bit)
------------------------
1. Convert image to grayscale.
2. Resize to 9×8 (9 columns, 8 rows).
3. For each row, compare adjacent pixels left→right.
   If left > right → 1, else → 0.
   This yields 8×8 = 64 bits.
4. Encode as a 16-character hex string.

Hamming distance thresholds (empirical):
  0       — identical images (different compression or metadata)
  1–4     — visually near-identical
  5–10    — similar but not the same (use caution)
  > 10    — different images

For medical images, a conservative threshold of 8 is recommended.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Maximum Hamming distance between two hashes to declare them near-duplicates.
DEFAULT_HAMMING_THRESHOLD = 8


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(data).hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """
    Compute a 64-bit difference hash (dhash) and return as a 16-char hex string.

    Parameters
    ----------
    image : PIL.Image
        Source image (any mode, any size).

    Returns
    -------
    str
        16-character lowercase hex string representing the 64-bit hash.
    """
    gray = image.convert("L").resize((9, 8), Image.LANCZOS)
    arr = np.array(gray, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]        # (8, 8) bool array
    bits = diff.flatten()                   # 64 bits
    value = int(np.packbits(bits).view(np.uint64)[0])
    return f"{value:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """
    Compute Hamming distance between two 16-char hex hashes.

    Returns the number of differing bits (0–64).
    """
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    xor = a ^ b
    return bin(xor).count("1")


class DuplicateDetector:
    """
    Detects exact and near-duplicate images in a collection.

    Usage
    -----
        detector = DuplicateDetector(hamming_threshold=8)
        groups = detector.find_duplicate_groups(records)
        for group in groups:
            print("Duplicates:", [r.filename for r in group])
    """

    def __init__(self, hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD) -> None:
        self.hamming_threshold = hamming_threshold

    def find_duplicate_groups(
        self, records: list
    ) -> list[list]:
        """
        Find groups of duplicate or near-duplicate images.

        Parameters
        ----------
        records : list[ImageRecord]
            All records to check. Records must have ``content_hash`` and
            ``perceptual_hash`` fields.

        Returns
        -------
        list of groups, where each group is a list of ImageRecord objects
        that are considered duplicates of each other. Singleton groups
        (no duplicates) are not returned.
        """
        # Phase 1: group by exact content hash
        exact: dict[str, list] = {}
        for rec in records:
            exact.setdefault(rec.content_hash, []).append(rec)

        exact_groups = [g for g in exact.values() if len(g) > 1]
        logger.info(
            "DuplicateDetector: exact duplicates — %d groups (%d images)",
            len(exact_groups),
            sum(len(g) for g in exact_groups),
        )

        # Phase 2: near-duplicate detection via dhash
        # Use the representative (first) from each exact group
        representatives = [g[0] for g in exact.values()]
        near_groups = self._find_near_duplicates(representatives)

        # Merge exact groups that appear in the same near-duplicate cluster
        all_groups: list[list] = []
        for near_group in near_groups:
            merged: list = []
            for rep in near_group:
                merged.extend(exact[rep.content_hash])
            all_groups.append(merged)

        # Add exact groups whose representative appeared only as singletons in near-dup
        near_hashes = {rec.content_hash for group in near_groups for rec in group}
        for content_h, group in exact.items():
            if content_h not in near_hashes and len(group) > 1:
                all_groups.append(group)

        logger.info(
            "DuplicateDetector: total duplicate groups (exact + near) — %d (%d images)",
            len(all_groups),
            sum(len(g) for g in all_groups),
        )
        return all_groups

    def _find_near_duplicates(self, records: list) -> list[list]:
        """Union-Find clustering of near-duplicates by dhash Hamming distance."""
        n = len(records)
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
                dist = hamming_distance(
                    records[i].perceptual_hash,
                    records[j].perceptual_hash,
                )
                if dist <= self.hamming_threshold:
                    union(i, j)

        clusters: dict[int, list] = {}
        for i, rec in enumerate(records):
            root = find(i)
            clusters.setdefault(root, []).append(rec)

        return [g for g in clusters.values() if len(g) > 1]

    def is_duplicate_of_any(
        self,
        candidate_hash: str,
        existing_hashes: list[str],
    ) -> bool:
        """
        Check if a candidate perceptual hash is a near-duplicate of any
        existing hash. Used for fast online duplicate checking at ingestion.
        """
        for existing in existing_hashes:
            if hamming_distance(candidate_hash, existing) <= self.hamming_threshold:
                return True
        return False


def hash_image_file(path: Path) -> tuple[str, str]:
    """
    Read an image file and compute both hashes.

    Returns
    -------
    (content_hash, perceptual_hash) as hex strings.
    """
    data = path.read_bytes()
    c_hash = content_hash(data)

    try:
        image = Image.open(path)
        p_hash = perceptual_hash(image)
    except Exception as exc:
        logger.warning("Could not compute perceptual hash for %s: %s", path, exc)
        p_hash = "0" * 16

    return c_hash, p_hash