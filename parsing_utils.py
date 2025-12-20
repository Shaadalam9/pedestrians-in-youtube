"""Parsing utilities used by the pipeline.

This module contains small, reusable parsing helpers that convert
string-encoded values into structured Python types.

The functions in this module are intentionally conservative:
  - They return empty/None on invalid input rather than throwing.
  - They avoid implicit assumptions about data cleanliness.
"""

from typing import List, Optional


class ParsingUtils:
    """Provides parsing helpers for common pipeline string formats.

    The pipeline reads values from CSVs and external tools (e.g., ffmpeg)
    that frequently encode structured data as strings. This class consolidates
    those conversions so the rest of the codebase remains readable.

    Note:
      The methods are side-effect free and safe to call repeatedly.
    """

    def __init__(self) -> None:
        """Initializes the parsing utility container.

        This is a stateless helper class; no initialization is required.
        """
        # Nothing to initialize.
        pass

    def _parse_bracket_list(self, s: Optional[str]) -> List[str]:
        """Parses a CSV-style bracket list string into a list of strings.

        The mapping CSV uses a lightweight list encoding such as:
          "[id1,id2]" or "['id1', 'id2']" (sometimes with whitespace)
        Values can also be missing, NaN-like, or None.

        Parsing rules:
          - None / "" / "nan" / "none" => []
          - Outer brackets are removed if present.
          - Items are split on commas.
          - Each item is stripped; empty items are dropped.

        Args:
          s: String value to parse (may be None).

        Returns:
          A list of non-empty items in the order they appear.
        """
        # Handle missing input early to keep the rest of the function simple.
        if s is None:
            return []

        # Normalize to string and strip whitespace.
        # Some CSV loaders pass non-string types; str(...) keeps us robust.
        text = str(s).strip()

        # Treat common "missing value" spellings as empty.
        if not text or text.lower() in ("nan", "none"):
            return []

        # Remove brackets (if any) and split into tokens.
        # Keep parsing simple because the expected input format is simple.
        tokens = text.strip().strip("[]").split(",")

        # Strip each token and drop empties.
        # This avoids returning [""] for inputs like "[]".
        return [t.strip().strip("'").strip('"') for t in tokens if t.strip()]

    def _hms_to_seconds(self, hms: str) -> Optional[float]:
        """Converts an HH:MM:SS[.fraction] timestamp string to seconds.

        This is commonly used for parsing ffmpeg progress output, e.g.:
          "00:00:12.345678"

        The function is tolerant:
          - Returns None if the format is not exactly 3 colon-separated parts.
          - Returns None if any part fails numeric conversion.

        Args:
          hms: Timestamp string in "HH:MM:SS" or "HH:MM:SS.sss" format.

        Returns:
          Total seconds as a float, or None if parsing fails.
        """
        # Defensive programming:
        # ffmpeg output can be noisy; parsing failures should not crash the run.
        try:
            # Split into hours, minutes, and seconds components.
            parts = hms.strip().split(":")
            if len(parts) != 3:
                # Unexpected format (e.g., "12.3" or "00:12").
                return None

            # Parse each component as float to support fractional seconds.
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])

            # Convert to total seconds.
            return hh * 3600.0 + mm * 60.0 + ss

        except Exception:
            # Any parsing error results in a None return for caller handling.
            return None
