#!/usr/bin/env python3
"""Which of the two things broke the track: the encoder, or the memory bank.

The question this answers is the one that decides what to fix next, and it is
not answerable by watching the video:

* **The encoder could not read the frame.** A thermal frame at night uses sixty
  grey levels out of 255; divided by 255 and put through ImageNet's mean and
  standard deviation it lands in a band a tenth as wide as anything the
  backbone was trained on. The target is *in* the frame and the features are
  weak anyway. The fix is on the input side: `--policy prefilter`, or training
  with the photometric augmentation.

* **The bank was holding another exposure.** The frame is perfectly readable --
  its span is normal -- but it has *moved*: the gain changed, the sun went
  behind cloud, the sensor re-ranged. Memory attention is then matching
  appearances encoded under the old exposure against pixels that no longer sit
  where they did. The fix is on the memory side: gate what enters the bank
  (`--policy memgate`), shorten it, or train the memory path on exposure
  changes.

Both look identical in the mp4 -- the box leaves the target and settles on
something else -- so this joins the three files a run writes and says which of
the two moved when it broke.

    python3 tools/diagnose_break.py frame_output/vis3_deneme1/clip/crop768_pool_deep_memgate

Needs `photometry.json` (`--photometry` on run_records, or `--prefilter-log`
with `--prefilter 0` on cli.py) and `track.json` (`--track-chart`).
`memory.json` is used when it is there.

It reports rather than concludes: a break where neither number moved is a break
neither fix addresses, and saying so is the point of measuring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# A frame using less than this many grey levels is one the encoder is being
# asked to read out of a tenth of its input range. It is the same floor
# `--prefilter` fires on, so the two agree about what "dark" means.
DARK_SPAN = 70.0
# Exposure movement, in grey levels, big enough that the bank's contents were
# encoded somewhere else. Measured against the median of the frames the bank
# is holding, so a slow drift the bank followed does not count.
BIG_DRIFT = 25.0


def load(folder: Path, name: str) -> dict | None:
    path = folder / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def breaks(track: dict) -> list[dict]:
    """The frames where the track lost the target, jumped, or ballooned.

    Read off `track.json`'s per-frame rows, which is what the tracker itself
    returned -- no ground truth is involved and none is needed: a mask that
    vanishes, moves further than a target could, or covers a field is a failure
    whatever the answer was.
    """
    rows = track.get("rows") or []
    out = []
    for row in rows:
        why = []
        if not row.get("held"):
            why.append("lost")
        if row.get("jump") and row["jump"] > 1.0:
            why.append(f"jumped {row['jump']:.1f} widths")
        if row.get("grew") and row["grew"] > 2.5:
            why.append(f"grew x{row['grew']:.1f}")
        if why:
            out.append({"frame": row["frame"], "why": ", ".join(why)})
    return out


def episodes(rows: list[dict], gap: int = 5) -> list[tuple[int, int, str]]:
    """Consecutive bad frames as one event, so a ten-frame dropout is one line."""
    out: list[list] = []
    for row in rows:
        if out and row["frame"] - out[-1][1] <= gap:
            out[-1][1] = row["frame"]
        else:
            out.append([row["frame"], row["frame"], row["why"]])
    return [(int(a), int(b), why) for a, b, why in out]


def photometry_at(photometry: dict, first: int, last: int, before: int = 4):
    """The span and the exposure drift around one break."""
    rows = {int(r["frame"]): r for r in photometry.get("rows", [])}
    window = [rows[f] for f in range(max(first - before, 0), last + 1) if f in rows]
    if not window:
        return None, None
    spans = [r["span"] for r in window if r.get("span") is not None]
    drifts = [r["drift"] for r in window if r.get("drift") is not None]
    return (min(spans) if spans else None), (max(drifts) if drifts else None)


def verdict(span, drift) -> str:
    dark = span is not None and span < DARK_SPAN
    moved = drift is not None and drift >= BIG_DRIFT
    if dark and moved:
        return "BOTH — the frame is dark and the exposure moved"
    if dark:
        return "ENCODER — the frame itself is unreadable (small span)"
    if moved:
        return "MEMORY — span fine, exposure moved under the bank"
    return "NEITHER — the pixels did not change; look for a distractor"


def report(folder: Path) -> int:
    track = load(folder, "track.json")
    photometry = load(folder, "photometry.json") or load(folder, "prefilter.json")
    memory = load(folder, "memory.json")
    if track is None or not track.get("rows"):
        print(f"!! {folder}/track.json has no per-frame rows. Re-run with "
              f"--track-chart (run_records passes it).")
        return 1
    if photometry is None:
        print(f"!! no photometry.json or prefilter.json in {folder}. Re-run "
              f"with --photometry.")
        return 1

    print(f"== {folder}")
    print(f"   {track['held']}/{track['frames']} frames held, longest gap "
          f"{track['longest_gap']}, {track['jumps']} jumps, max share "
          f"{track['share_max']:.1%}")
    print(f"   span median {photometry.get('span_median')} levels "
          f"({photometry.get('dark_frames')}/{photometry.get('frames')} under "
          f"{DARK_SPAN:.0f}), exposure drift median "
          f"{photometry.get('drift_median')}, max {photometry.get('drift_max')}")
    if memory:
        gates = ", ".join(f"{k} x{v}" for k, v in memory.get("by_gate", {}).items())
        print(f"   memory gate refused {memory['refused']}/{memory['judged']}"
              + (f" ({gates})" if gates else ""))

    events = episodes(breaks(track))
    if not events:
        print("   no break to explain: nothing was lost, no jump, no balloon.")
        return 0

    # The cause runs a few frames ahead of the effect -- the exposure moves,
    # the bank fills with it, and the track comes off a frame or two later --
    # so both columns are the worst reading over the break and the four frames
    # before it, not the reading on the break frame alone.
    print(f"\n   {'frames':>13}  {'span-':>6} {'drift+':>6}  what happened "
          f"(worst over the break and the 4 frames before it)")
    counts: dict[str, int] = {}
    for first, last, why in events:
        span, drift = photometry_at(photometry, first, last)
        call = verdict(span, drift)
        counts[call.split(" ")[0]] = counts.get(call.split(" ")[0], 0) + 1
        window = f"{first}-{last}" if last > first else str(first)
        print(f"   {window:>13}  {span if span is not None else '-':>6} "
              f"{drift if drift is not None else '-':>6}  {why}")
        print(f"   {'':>13}  {call}")
    print("\n   " + ", ".join(f"{name} x{count}" for name, count
                              in sorted(counts.items(), key=lambda kv: -kv[1])))
    print("   ENCODER -> --policy prefilter, or photometric augmentation in "
          "training.\n   MEMORY  -> --policy memgate, a shorter bank, or "
          "training the memory\n             path across exposure changes.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path,
                        help="Run folders written by run_records.py.")
    args = parser.parse_args(argv)
    worst = 0
    for folder in args.folders:
        worst = max(worst, report(folder))
        print()
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
