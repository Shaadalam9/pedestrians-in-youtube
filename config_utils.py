"""Configuration access helpers.

This module provides a small wrapper around `common.get_configs` that:
  - Returns a default value when a configuration key is missing or the
    configuration backend errors.
  - Centralizes configuration access behind a single API, which makes the code
    easier to understand and test.

Notes:
  - The `common` module is expected to expose `get_configs(key: str)`.
  - This wrapper intentionally swallows exceptions to keep callers simple and to
    provide resilient defaults in production pipelines.
"""

import common


class ConfigUtils:
    """Provides safe access to application configuration values.

    This class wraps `common.get_configs` and guarantees that callers receive a
    value (either the configured value or a default) rather than an exception.

    Typical usage:
      cfg = Config()
      max_workers = cfg.safe_get("max_workers", default=4)

    Design:
      - Public methods are intentionally small and predictable.
      - Errors are handled within the class so the rest of the application does
        not need repetitive try/except blocks.
    """

    def __init__(self) -> None:
        """Initializes the configuration wrapper.

        The wrapper does not preload configuration. It defers reads to
        `common.get_configs` so that configuration can remain dynamic if the
        underlying backend supports it.

        Args:
          None.
        """
        # Nothing to initialize right now.
        # Keeping an explicit constructor improves discoverability and provides
        # a natural place to add caching or validation later if needed.
        pass

    def safe_get(self, key: str, default=None):
        """Returns the configuration value for `key`, or `default` on failure.

        This method is the preferred entry point for configuration reads.
        It catches broad exceptions so that callers do not need to worry about
        the underlying configuration backend's error types.

        Args:
          key: Configuration key to look up.
          default: Value to return when the key is missing or the backend fails.

        Returns:
          The configuration value returned by `common.get_configs(key)` if
          successful; otherwise `default`.
        """
        # Delegate to the underlying configuration provider.
        # We isolate the dependency here so the rest of the codebase stays clean.
        try:
            # If the backend is available and the key exists, return it.
            return common.get_configs(key)
        except Exception:
            # If anything goes wrong (missing key, backend not initialized, etc.),
            # degrade gracefully by returning the caller-provided default.
            return default

    # Backwards-compatibility alias:
    # Some code may still call `_safe_get_config`; keeping it avoids breakage
    # during refactors. Prefer `safe_get` for new code.
    def _safe_get_config(self, key: str, default=None):
        """Backward-compatible wrapper for `safe_get`.

        Args:
          key: Configuration key to look up.
          default: Value to return when lookup fails.

        Returns:
          Same as `safe_get(key, default)`.
        """
        # Route legacy callers to the new API.
        return self.safe_get(key, default)
