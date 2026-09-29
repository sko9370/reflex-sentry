# Local precheck word list

`reflex_sentry.models.precheck` uses `common_words.txt` to estimate how much
of a prompt reads as ordinary English. The list is generated locally and is
not committed. Without the file, the common-word signal has no reference
vocabulary, so build it before running the precheck. Unit tests use a
synthetic vocabulary and do not require the generated file.

From the repository root, with project dependencies installed:

```sh
python -m reflex_sentry.models.build_precheck_words
```

The command reads `/usr/share/hunspell/en_US.dic` by default. To use an
installed dictionary elsewhere, pass `--dictionary /path/to/en_US.dic`.
It writes `reflex_sentry/models/data/common_words.txt` and an adjacent
`common_words.meta.json` by default. Set `REFLEX_SENTRY_COMMON_WORDS` or pass
`--out /absolute/path/common_words.txt` to choose an output path. Point the
precheck loader at that same path with `REFLEX_SENTRY_COMMON_WORDS`.

The builder combines three sources, then sorts and deduplicates the result:

1. `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` from the installed
   scikit-learn package.
2. Entries in the local Hunspell dictionary whose headwords, before any
   `/FLAGS`, are already lowercase ASCII letters and at least two letters
   long. This excludes capitalized names and acronyms.
3. The 29 explicit project-authored terms in
   `reflex_sentry/models/build_precheck_words.py`.

The metadata records the source dictionary SHA-256, output SHA-256, source
counts, and scikit-learn version. The exact output depends on the dictionary
and scikit-learn versions installed locally. The currently available Fedora
`hunspell-en-US` dictionary and scikit-learn generate 36,432 words. These
files contain third-party word data and are ignored by Git; consult the
licenses of your installed packages if distributing the generated list.
