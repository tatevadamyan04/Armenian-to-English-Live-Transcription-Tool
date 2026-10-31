"""Score a hypothesis TSV against a reference TSV (TESTS.md §3.3).

Both files are UTF-8, tab-separated, with the header `id<TAB>hy<TAB>en`
(the format of eval/refs/e1.tsv). Rows are matched by id; an id present in one
file but not the other is an error, so nothing is scored silently on a subset.

Usage:
    python eval/score.py eval/refs/e1.tsv data/e1/hyp_a.tsv
    python eval/score.py REF HYP --rating-sheet data/e1/rate_a.tsv   # sheet for Tatev's 1–3 rating
    python eval/score.py REF HYP --ratings data/e1/rate_a.tsv        # read the filled sheet back

Prints per-item lines and the corpus CER, WER, chrF (and mean rating) to stdout.
Exit code 1 on any input problem.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from metrics import cer, chrf, normalise, wer

COLUMNS = ["id", "hy", "en"]
RATING_COLUMNS = ["id", "hy_ref", "hy_hyp", "en_ref", "en_hyp", "rating"]


class InputError(Exception):
    pass


def read_tsv(path: Path, columns: list[str]) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != columns:
            raise InputError(f"{path}: header must be {columns}, found {reader.fieldnames}")
        rows: dict[str, dict[str, str]] = {}
        for line_no, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise InputError(f"{path}:{line_no}: expected {len(columns)} tab-separated fields")
            if row["id"] in rows:
                raise InputError(f"{path}:{line_no}: duplicate id {row['id']!r}")
            rows[row["id"]] = row
    if not rows:
        raise InputError(f"{path}: no rows")
    return rows


def matched_ids(ref: dict, hyp: dict, ref_path: Path, hyp_path: Path) -> list[str]:
    missing = sorted(set(ref) - set(hyp))
    extra = sorted(set(hyp) - set(ref))
    if missing or extra:
        raise InputError(
            f"ids differ between {ref_path} and {hyp_path}: "
            f"missing in hypothesis {missing or '-'}; not in reference {extra or '-'}"
        )
    return sorted(ref)


def write_rating_sheet(path: Path, ids: list[str], ref: dict, hyp: dict) -> None:
    if path.exists():
        raise InputError(f"{path} exists; not overwriting a sheet that may hold ratings")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        w.writerow(RATING_COLUMNS)
        for i in ids:
            w.writerow([i, ref[i]["hy"], hyp[i]["hy"], ref[i]["en"], hyp[i]["en"], ""])


def read_ratings(path: Path, ids: list[str]) -> dict[str, int]:
    rows = read_tsv(path, RATING_COLUMNS)
    if sorted(rows) != ids:
        raise InputError(f"{path}: ids do not match the scored set")
    ratings = {}
    for i, row in rows.items():
        if row["rating"] not in {"1", "2", "3"}:
            raise InputError(f"{path}: id {i!r} has rating {row['rating']!r}; must be 1, 2 or 3")
        ratings[i] = int(row["rating"])
    return ratings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("hyp", type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rating-sheet", type=Path, help="write a sheet for 1–3 ratings and exit")
    g.add_argument("--ratings", type=Path, help="a filled rating sheet to include in the report")
    args = ap.parse_args(argv)

    try:
        ref = read_tsv(args.ref, COLUMNS)
        hyp = read_tsv(args.hyp, COLUMNS)
        ids = matched_ids(ref, hyp, args.ref, args.hyp)
        if args.rating_sheet:
            write_rating_sheet(args.rating_sheet, ids, ref, hyp)
            print(f"wrote {len(ids)} rows to {args.rating_sheet}; fill the 'rating' column with 1, 2 or 3")
            return 0
        ratings = read_ratings(args.ratings, ids) if args.ratings else None
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for i in ids:
        r_hy, h_hy = ref[i]["hy"], hyp[i]["hy"]
        c = cer([r_hy], [h_hy])
        same = "=" if normalise(r_hy) == normalise(h_hy) else " "
        rating = f"  rating {ratings[i]}" if ratings else ""
        print(f"{i}  {same} CER {100 * c.rate:5.1f}%  chrF {chrf([ref[i]['en']], [hyp[i]['en']]):5.1f}{rating}")
        if same != "=":
            print(f"    ref: {normalise(r_hy)}\n    hyp: {normalise(h_hy)}")

    refs_hy = [ref[i]["hy"] for i in ids]
    hyps_hy = [hyp[i]["hy"] for i in ids]
    c, w = cer(refs_hy, hyps_hy), wer(refs_hy, hyps_hy)
    exact = sum(normalise(a) == normalise(b) for a, b in zip(refs_hy, hyps_hy))
    print(f"\nitems: {len(ids)}   identical after normalisation: {exact}")
    print(f"CER: {100 * c.rate:.2f}%  ({c.edits} edits / {c.ref_len} chars)")
    print(f"WER: {100 * w.rate:.2f}%  ({w.edits} edits / {w.ref_len} words)")
    print(f"chrF (EN): {chrf([ref[i]['en'] for i in ids], [hyp[i]['en'] for i in ids]):.1f}")
    if ratings:
        counts = {k: sum(v == k for v in ratings.values()) for k in (1, 2, 3)}
        mean = sum(ratings.values()) / len(ratings)
        print(f"rating: mean {mean:.2f}  (1: {counts[1]}, 2: {counts[2]}, 3: {counts[3]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
