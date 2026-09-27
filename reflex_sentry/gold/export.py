"""Turn a drawn sample (see `sample.py`) into labeling-ready files.

Writes two interchangeable formats of the same queue plus the Label Studio
project config:

(a) a spreadsheet CSV with empty gold/tags/notes columns, ready to open in
    Excel/Sheets. Allowed values are listed in extra `_ref_*` columns at the
    far right so a spreadsheet's data-validation dropdown can point at them
    (Data > Data validation > list from a range).
(b) a Label Studio JSON task list, plus `configs/label_studio.xml`: a single
    choice for `gold`, two conditional multi-selects for `cat:*` (shown only
    when gold=dangerous) and `hn:*` (shown only when gold=benign; ingest.py
    adds the `hard_negative` tag automatically whenever any hn:* is chosen,
    so the labeler never has to tick it separately), and a notes textarea.

Both outputs carry only what the input sample carries: run `sample.py`
without `--no-blind` (the default) and source/source_label stay hidden here
too.

    python -m reflex_sentry.gold.export --sample data/gold/samples/val_sample.parquet \
        --csv data/gold/samples/val_sheet.csv --label-studio data/gold/samples/val_tasks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import ALL_GOLD_LABELS, CAT_TAGS, HARD_NEGATIVE_TAG, HN_TAGS

LABEL_STUDIO_XML_PATH = "configs/label_studio.xml"

SHEET_COLUMNS = ["id", "text", "split", "gold", "tags", "notes", "labeler_pass", "labeled_at"]
OPTIONAL_SHEET_COLUMNS = ["source", "source_label"]  # only kept if present in the sample (non-blind)


def read_sample(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def to_spreadsheet(sample: pd.DataFrame) -> pd.DataFrame:
    cols = SHEET_COLUMNS[:3] + [c for c in OPTIONAL_SHEET_COLUMNS if c in sample.columns] + SHEET_COLUMNS[3:]
    out = pd.DataFrame(index=sample.index)
    for c in cols:
        out[c] = sample[c] if c in sample.columns else ""
    # Reference lists for spreadsheet data validation, padded to the sheet's row count.
    refs = {"_ref_gold": list(ALL_GOLD_LABELS), "_ref_tags": [HARD_NEGATIVE_TAG, *CAT_TAGS, *HN_TAGS]}
    n = len(out)
    for name, values in refs.items():
        col = list(values) + [""] * (n - len(values)) if n > len(values) else list(values)[:n]
        out[name] = pd.Series(col, index=out.index)
    return out


def to_label_studio_tasks(sample: pd.DataFrame) -> list[dict]:
    tasks = []
    for _, row in sample.iterrows():
        data = {"item_id": row["id"], "text": row["text"], "split": row.get("split")}
        if "source" in sample.columns:
            data["source"] = row.get("source")
        if "source_label" in sample.columns:
            data["source_label"] = row.get("source_label")
        tasks.append({"data": data})
    return tasks


def label_studio_xml() -> str:
    cat_choices = "\n".join(f'    <Choice value="{c}"/>' for c in CAT_TAGS)
    hn_choices = "\n".join(f'    <Choice value="{c}"/>' for c in HN_TAGS)
    return f"""<View>
  <Header value="Prompt"/>
  <Text name="text" value="$text"/>

  <Header value="Gold label"/>
  <Choices name="gold" toName="text" choice="single" required="true" showInline="true">
    <Choice value="dangerous"/>
    <Choice value="benign"/>
    <Choice value="ambiguous"/>
    <Choice value="out_of_scope"/>
  </Choices>

  <Header value="Dangerous subcategory (only if gold = dangerous)"
          visibleWhen="choice-selected" whenTagName="gold" whenChoiceValue="dangerous"/>
  <Choices name="cat_tags" toName="text" choice="multiple" showInline="true"
           visibleWhen="choice-selected" whenTagName="gold" whenChoiceValue="dangerous">
{cat_choices}
  </Choices>

  <Header value="Hard-negative subcategory (only if gold = benign; hard_negative is added for you)"
          visibleWhen="choice-selected" whenTagName="gold" whenChoiceValue="benign"/>
  <Choices name="hn_tags" toName="text" choice="multiple" showInline="true"
           visibleWhen="choice-selected" whenTagName="gold" whenChoiceValue="benign">
{hn_choices}
  </Choices>

  <Header value="Notes (judgment call rationale, optional)"/>
  <TextArea name="notes" toName="text" placeholder="why this call" rows="3" maxSubmissions="1" editable="true"/>
</View>
"""


def write_label_studio_config(path: str = LABEL_STUDIO_XML_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(label_studio_xml())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="output of sample.py")
    ap.add_argument("--csv", help="spreadsheet CSV path to write")
    ap.add_argument("--label-studio", help="Label Studio JSON task file path to write")
    ap.add_argument("--label-studio-config", default=LABEL_STUDIO_XML_PATH,
                    help="Label Studio labeling config XML to (re)write")
    ap.add_argument("--no-config", action="store_true", help="skip writing the Label Studio XML config")
    a = ap.parse_args()

    sample = read_sample(a.sample)
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        to_spreadsheet(sample).to_csv(a.csv, index=False)
        print(f"wrote {len(sample)} rows -> {a.csv}")
    if a.label_studio:
        Path(a.label_studio).parent.mkdir(parents=True, exist_ok=True)
        Path(a.label_studio).write_text(json.dumps(to_label_studio_tasks(sample), indent=2))
        print(f"wrote {len(sample)} tasks -> {a.label_studio}")
    if not a.no_config:
        write_label_studio_config(a.label_studio_config)
        print(f"wrote Label Studio config -> {a.label_studio_config}")


if __name__ == "__main__":
    main()
