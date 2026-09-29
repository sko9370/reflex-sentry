"""Build the local word list used by the precheck natural-language signal.

Run with ``python -m reflex_sentry.models.build_precheck_words``. This reads
an already installed Hunspell dictionary; it never downloads word data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_DICTIONARY = Path("/usr/share/hunspell/en_US.dic")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "common_words.txt"
OUTPUT_ENV = "REFLEX_SENTRY_COMMON_WORDS"

# Project-authored terms absent from the reference dictionary and stopwords.
SUPPLEMENT = frozenset({
    "authentication", "backdoors", "botnets", "credentials", "ddos",
    "deauthorized", "endpoints", "escalation", "exfiltrate",
    "exfiltration", "forensics", "keylogger", "keyloggers",
    "misconfiguration", "misconfigured", "mitigation", "obfuscated",
    "obfuscation", "payloads", "phishing", "remediation", "rootkits",
    "sandboxing", "spoofed", "spoofing", "unauthorized", "untrusted",
    "usernames", "vulnerabilities",
})
LOWERCASE_WORD = re.compile(r"[a-z]{2,}\Z")


def build(dictionary: Path, output: Path) -> dict:
    """Write sorted unique words and a JSON record of inputs and results."""
    source = dictionary.read_bytes()
    lines = source.decode("utf-8").splitlines()
    if not lines or not lines[0].strip().isdigit():
        raise ValueError(f"Expected a Hunspell .dic entry-count header: {dictionary}")

    dictionary_words = {
        word
        for line in lines[1:]
        if LOWERCASE_WORD.fullmatch(word := line.strip().split("/", 1)[0])
    }
    stopwords = {word.lower() for word in ENGLISH_STOP_WORDS}
    words = sorted(dictionary_words | stopwords | SUPPLEMENT)
    data = ("\n".join(words) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    metadata = {
        "dictionary": str(dictionary.resolve()),
        "dictionary_sha256": hashlib.sha256(source).hexdigest(),
        "dictionary_words": len(dictionary_words),
        "sklearn_version": sklearn_version,
        "stopwords": len(stopwords),
        "supplement": len(SUPPLEMENT),
        "output_words": len(words),
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }
    metadata_path = output.with_name(output.stem + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY,
                        help="installed Hunspell .dic file (default: %(default)s)")
    parser.add_argument("--out", type=Path,
                        default=Path(os.environ.get(OUTPUT_ENV) or DEFAULT_OUTPUT),
                        help=f"word list path (default: {OUTPUT_ENV} or package data directory)")
    args = parser.parse_args(argv)
    try:
        result = build(args.dictionary, args.out)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"build_precheck_words: {exc}\n")
    print(f"Wrote {result['output_words']} words to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
