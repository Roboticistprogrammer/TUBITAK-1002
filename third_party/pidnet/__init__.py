# ------------------------------------------------------------------------------
# firecls addition (not part of upstream PIDNet).
# Package marker so the two vendored upstream files can be imported through
# ``firecls.baselines.pidnet.model.vendored_pidnet`` without putting third_party/
# on sys.path. ``pidnet.py`` and ``model_utils.py`` are byte-identical to upstream.
# ------------------------------------------------------------------------------
from .pidnet import PIDNet, get_pred_model  # noqa: F401
