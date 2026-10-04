"""Hand-labeling toolkit for reflex-sentry's gold evaluation sets.

Draws stratified samples from the data pipeline's candidate pools
(``sample.py``), turns a sample into a labeling sheet or Label Studio task
file (``export.py``), validates and ingests filled-in labels back into
``data/gold/<split>.csv`` and the merged ``data/processed/<split>.parquet``
(``ingest.py``), and checks pass-1-vs-pass-2 self-agreement (``agreement.py``).

Label taxonomy and tag vocabulary are defined here (configs/labeling_guide.md) rather
than imported from ``reflex_sentry.data``, which another workstream owns.
"""
from __future__ import annotations

# The three gold labels used for evaluation, plus the fourth label a labeler
# may assign during hand-labeling to mean "does not belong in this eval set at
# all" (out_of_scope rows are dropped when a gold CSV is merged into a pool).
GOLD_LABELS = ("dangerous", "benign", "ambiguous")
ALL_GOLD_LABELS = GOLD_LABELS + ("out_of_scope",)

# dangerous subcategories, tagged cat:<name> (configs/labeling_guide.md)
CAT_TAGS = (
    "cat:malware_dev",
    "cat:evasion",
    "cat:exploit_weaponization",
    "cat:phishing_se",
    "cat:credential_theft",
    "cat:intrusion_ops",
    "cat:ddos",
)

# benign hard-negative subcategories, tagged hard_negative + hn:<name>
HN_TAGS = (
    "hn:detection",
    "hn:analysis",
    "hn:vuln_explain",
    "hn:secure_dev",
    "hn:ir",
    "hn:training",
    "hn:cti",
)

HARD_NEGATIVE_TAG = "hard_negative"

# Every tag ingest.py will accept in a gold sheet's tags column.
ALLOWED_TAGS = frozenset({HARD_NEGATIVE_TAG, *CAT_TAGS, *HN_TAGS})

# data/gold/<split>.csv column order (docs/DATA_CONTRACT.md)
GOLD_CSV_COLUMNS = ("id", "text", "source", "gold", "tags", "labeler_pass", "labeled_at", "notes")


def split_tags(s) -> list[str]:
    """Parse a semicolon-separated tags string into a list, dropping blanks."""
    if s is None or (isinstance(s, float) and s != s):  # NaN
        return []
    return [t.strip() for t in str(s).split(";") if t.strip()]


def join_tags(tags) -> str:
    return ";".join(dict.fromkeys(t for t in tags if t))
