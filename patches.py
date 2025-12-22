"""Runtime patches for third-party libraries.

This module contains small, targeted monkey-patches that improve robustness of
third-party dependencies used by the pipeline.

Guiding principles:
  - Patches must be narrowly scoped and easy to remove later.
  - Patches must be idempotent (safe to apply multiple times).
  - Patches must preserve behavior except for the specific bug fix.
"""
import sys
import numpy as np
import scipy.spatial.distance as ssd
from functools import wraps
from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class Patches:
    """Applies runtime patches for known third-party library issues.

    This class groups monkey-patches in a single, discoverable place.
    Keeping patches centralized prevents scattering fragile code throughout the
    pipeline and makes audits/removals straightforward.
    """

    def __init__(self) -> None:
        """Initializes the patch container.

        No state is required. The methods are idempotent and rely on flags placed
        on patched third-party objects to prevent double patching.
        """
        # Intentionally empty: this class acts as a namespace for patch methods.
        pass

    def patch_ultralytics_botsort_numpy_cpu_bug(self, logger=None) -> None:
        """Fixes a BoT-SORT crash when ReID features are numpy arrays.

        Background:
          Under certain configurations (BoT-SORT + with_reid=True + model="auto"),
          Ultralytics may provide ReID feature vectors as numpy arrays. Some code
          paths assume torch tensors and call `.cpu()`, causing:
            AttributeError: 'numpy.ndarray' object has no attribute 'cpu'

        What this patch does:
          - Monkey-patches `ultralytics.trackers.bot_sort.BOTSORT.__init__`.
          - Only when (with_reid=True AND model="auto"):
              - Replaces `self.encoder` with a wrapper that safely converts:
                  - torch.Tensor-like -> tensor.cpu().numpy()
                  - numpy / list-like -> np.asarray(...)
          - All other tracker configurations remain unchanged.

        What this patch does NOT do:
          - It does not change tracking logic, association, thresholds, or ReID
            behavior. It only makes feature conversion safe.

        Idempotency:
          - The patch sets `BOTSORT._numpy_safe_patched = True`.
          - If that flag is present, the patch is treated as already applied.

        Args:
          logger: Optional logger instance exposing `.info()` and `.warning()`.
            If not provided, messages are printed to stdout.

        Returns:
          None.
        """
        # Import inside the method so:
        #   - Importing this module does not require ultralytics to be installed.
        #   - Failure to import ultralytics does not break the pipeline.
        try:
            import numpy as np
            import ultralytics.trackers.bot_sort as bot_sort  # type: ignore

            # Avoid applying the patch more than once.
            # This keeps behavior predictable across repeated initializations.
            if getattr(bot_sort.BOTSORT, "_numpy_safe_patched", False):
                return

            # Capture the original constructor so the patch can delegate.
            # This is critical: we do not want to change initialization behavior.
            orig_init = bot_sort.BOTSORT.__init__

            def _safe_auto_encoder(feats, s):
                """Converts feature vectors into numpy arrays without assuming torch.

                Args:
                  feats: Iterable of feature vectors. Items may be torch tensors,
                    numpy arrays, or other array-like objects.
                  s: Placeholder required by the encoder signature.

                Returns:
                  List of numpy arrays corresponding to the input features.
                """
                # Accumulate output explicitly for clarity.
                out = []

                # Convert each feature independently to handle mixed-type batches.
                for f in feats:
                    # Torch path: detect tensor-like objects via `.cpu()` rather
                    # than importing torch (keeps patch lightweight).
                    if hasattr(f, "cpu"):
                        try:
                            # Detach when available to avoid keeping graph refs.
                            f = f.detach() if hasattr(f, "detach") else f

                            # Convert tensor -> cpu -> numpy.
                            out.append(f.cpu().numpy())
                            continue
                        except Exception:
                            # If conversion fails, fall back to numpy coercion.
                            pass

                    # Numpy / array-like path: coerce to numpy array.
                    out.append(np.asarray(f))

                return out

            def patched_init(self, args, frame_rate=30):
                """Wraps BOTSORT.__init__ and swaps encoder only in one scenario."""
                # Always call the original constructor first.
                # This preserves all default initialization behavior.
                orig_init(self, args, frame_rate=frame_rate)

                # Read tracker args safely. The nested getattr chain avoids
                # AttributeError if the args structure changes in new releases.
                model = getattr(getattr(self, "args", None), "model", None)
                with_reid = bool(getattr(getattr(self, "args", None), "with_reid", False))

                # Only override the encoder in the exact problematic configuration.
                if with_reid and str(model).lower() == "auto":
                    self.encoder = _safe_auto_encoder

            # Apply the monkey-patch to the class.
            bot_sort.BOTSORT.__init__ = patched_init  # type: ignore

            # Mark as patched for idempotency across repeated calls.
            bot_sort.BOTSORT._numpy_safe_patched = True  # type: ignore

            # Log success for traceability.
            if logger:
                logger.info("Applied BoT-SORT numpy-safe patch for with_reid + model:auto.")
            else:
                print("Applied BoT-SORT numpy-safe patch for with_reid + model:auto.")

        except Exception as e:
            # This patch must never be fatal to the pipeline.
            # If it cannot be applied, log and continue.
            if logger:
                logger.warning(f"BoT-SORT patch not applied (continuing): {e!r}")
            else:
                print(f"BoT-SORT patch not applied (continuing): {e!r}")

    def patch_scipy_cdist_accept_1d(self, logger):
        """
        Fix BoT-SORT / SciPy cdist edge-case:
        SciPy's cdist requires XA and XB to be 2D. BoT-SORT sometimes passes a 1D embedding.
        This patch reshapes 1D -> (1, -1) and handles empty inputs.
        Also patches any already-imported Ultralytics aliases of cdist.
        """
        try:

            # Idempotent
            if getattr(ssd.cdist, "__patched_accept_1d__", False):
                return

            orig_cdist = ssd.cdist

            @wraps(orig_cdist)
            def cdist_safe(XA, XB, *args, **kwargs):
                XA = np.asarray(XA)
                XB = np.asarray(XB)

                # Empty -> return empty distance matrix
                if XA.size == 0 or XB.size == 0:
                    na = 0 if XA.size == 0 else (1 if XA.ndim == 1 else XA.shape[0])
                    nb = 0 if XB.size == 0 else (1 if XB.ndim == 1 else XB.shape[0])
                    return np.empty((na, nb), dtype=np.float32)

                # 1D -> 2D
                if XA.ndim == 1:
                    XA = XA.reshape(1, -1)
                if XB.ndim == 1:
                    XB = XB.reshape(1, -1)

                return orig_cdist(XA, XB, *args, **kwargs)

            cdist_safe.__patched_accept_1d__ = True  # type: ignore

            # Patch SciPy canonical entrypoint
            ssd.cdist = cdist_safe

            # Patch any module-level aliases (Ultralytics often does: from scipy.spatial.distance import cdist)
            for m in list(sys.modules.values()):
                if m is None:
                    continue
                try:
                    d = vars(m)
                except Exception:
                    continue
                for name, val in list(d.items()):
                    if val is orig_cdist:
                        try:
                            setattr(m, name, cdist_safe)
                        except Exception:
                            pass

            logger.info("Patched SciPy/Ultralytics cdist to accept 1D embeddings (BoT-SORT XB 2D fix).")

        except Exception as e:
            logger.warning(f"Failed to patch SciPy cdist accept-1d: {e!r}")

def patch_scipy_cho_factor_jitter(self, logger, eps_list=None):
    """
    Fix Ultralytics Kalman filter numerical crash:
      numpy.linalg.LinAlgError: '<k>-th leading minor ... is not positive definite'

    Root cause:
      Ultralytics' Kalman filter uses scipy.linalg.cho_factor() on a covariance
      matrix that can occasionally become non positive definite due to numerical
      error or extreme measurements.

    Patch:
      Monkey-patch scipy.linalg.cho_factor (and the internal implementation module)
      to, on LinAlgError, symmetrize the matrix and add a small diagonal jitter
      (eps * I), retrying a few eps values.

    This patch is narrowly scoped (only affects cho_factor) and idempotent.

    Args:
      logger: logger instance
      eps_list: optional iterable of eps values to try (default progressive)
    """
    try:
        import sys
        import numpy as np
        import scipy.linalg as sla

        if getattr(sla.cho_factor, "__patched_jitter__", False):
            return

        orig_cho_factor = sla.cho_factor
        if eps_list is None:
            eps_list = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

        def cho_factor_safe(a, lower=False, overwrite_a=False, check_finite=True):
            try:
                return orig_cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
            except np.linalg.LinAlgError:
                A = np.asarray(a)
                # Only attempt jitter for square 2D matrices
                if A.ndim != 2 or A.shape[0] != A.shape[1]:
                    raise
                # Symmetrize to reduce numerical asymmetry
                A2 = (A + A.T) * 0.5
                n = A2.shape[0]
                I = np.eye(n, dtype=A2.dtype)
                last_err = None
                for eps in eps_list:
                    try:
                        return orig_cho_factor(A2 + (eps * I), lower=lower, overwrite_a=False, check_finite=check_finite)
                    except np.linalg.LinAlgError as e:
                        last_err = e
                        continue
                # Give up with original error context
                raise (last_err if last_err is not None else np.linalg.LinAlgError('cho_factor failed after jitter retries'))

        cho_factor_safe.__patched_jitter__ = True  # type: ignore

        # Patch canonical entrypoint
        sla.cho_factor = cho_factor_safe  # type: ignore

        # Patch internal module reference (where traceback points)
        try:
            import scipy.linalg._decomp_cholesky as _dc  # type: ignore
            _dc.cho_factor = cho_factor_safe  # type: ignore
        except Exception:
            pass

        # Patch any module-level aliases (defensive)
        for m in list(sys.modules.values()):
            if m is None:
                continue
            try:
                d = vars(m)
            except Exception:
                continue
            for name, val in list(d.items()):
                if val is orig_cho_factor:
                    try:
                        setattr(m, name, cho_factor_safe)
                    except Exception:
                        pass

        logger.info("Patched SciPy cho_factor with diagonal jitter to avoid Kalman PD LinAlgError.")
    except Exception as e:
        logger.warning(f"Failed to patch SciPy cho_factor jitter: {e!r}")
