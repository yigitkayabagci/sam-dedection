#!/usr/bin/env python3
"""Look at what `decompose` did to SegFly, frame by frame, with your own eyes.

Every number in `docs/segfly_decomposition.md` is an aggregate, and an aggregate
cannot tell you whether a particular mask sits on a car. This renders the three
things side by side -- **the thermal frame, the semantic map SegFly ships, and
the instances the training path reconstructs from it** -- so a person can page
through them and say "that one is wrong".

It runs the repository's own `decompose`, with the repository's own gates, so
what you are looking at is what stage B trains on and not a re-implementation
that might differ. Rejected components are drawn too, dashed, labelled with the
gate that stopped them: a target missing from the picture is as interesting as
a wrong one.

**Reads either layout.** A directory of parquet shards (the publisher's own
download) or the `images/`+`labels/` tree `tools/export_hf_dataset.py` writes.
Nothing needs exporting first.

The filters are the point. SegFly's `truck` is the class the report calls
broken, so seeing it mixed into every page makes the rest harder to judge:

    --filter vehicle-only   frames whose targets are all `vehicle`
    --filter has-truck      only frames carrying a `truck` target
    --filter merge-suspect  only frames holding an instance shaped like two
                            vehicles merged -- the case `fill` cannot see, and
                            the one worth a human eye more than any other

    python tools/segfly_eyeball.py --root ~/Downloads/SegFly --out ~/Desktop/kontrol
    python tools/segfly_eyeball.py --root ~/Downloads/SegFly --out ~/Desktop/kontrol \\
        --filter vehicle-only --limit 60
    python tools/segfly_eyeball.py --root ~/Downloads/SegFly --out ~/Desktop/kontrol \\
        --filter merge-suspect --drop-truck

Writes numbered pages plus `ozet.csv`, one row per component, so a frame you
flag by eye can be found again by name.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import (  # noqa: E402
    SPECS,
    Instance,
    InstanceGates,
    decompose,
    list_frames,
    read_mask,
    reject_reason,
)

# The publisher's own class table, from the SegFly dataset card. Display only --
# the annotation itself is a single-channel id map.
COLORS = {0: (0, 0, 0), 1: (128, 0, 128), 2: (204, 163, 72), 3: (128, 0, 0),
          4: (192, 192, 192), 6: (0, 255, 0), 7: (112, 148, 32), 8: (64, 64, 0),
          9: (255, 255, 0), 13: (0, 128, 128), 14: (0, 0, 255), 16: (255, 0, 0),
          17: (64, 160, 120), 33: (128, 64, 128), 34: (240, 120, 120),
          36: (128, 128, 64)}
# Instance identity, not a data encoding -- the numbers carry the identity and
# these only make neighbours easy to tell apart.
INSTANCE = [(230, 85, 60), (50, 140, 215), (245, 180, 40), (120, 190, 80),
            (175, 95, 205), (0, 175, 170), (240, 120, 175), (150, 110, 60),
            (90, 110, 230), (210, 205, 70)]


def load_source(root: Path, modality: str, spec):
    """`(name, thermal frame, semantic map)` for every frame under `root`.

    Yields lazily: the label maps are read twice (once to measure, once to
    draw) and the images only for the frames that survive the filter, so a
    15 000-frame set does not have to be decoded to look at forty of them.
    """
    shards = sorted(root.glob("*.parquet")) or sorted(root.glob("**/*.parquet"))
    if shards:
        return _parquet_reader(shards, modality), len(shards), "parquet"
    frames = list_frames(root, spec, modality)
    if not frames:
        raise SystemExit(
            f"{root} holds neither *.parquet nor the images/+labels/ tree "
            f"`{spec.glob(modality)}` and `{spec.mask_glob(modality)}` expect. "
            f"Point --root at the folder you unpacked, or at the parquet "
            f"shards as they were downloaded.")
    return _folder_reader(frames), len(frames), "klasör"


def _folder_reader(frames):
    import cv2

    def read(frame):
        raw = cv2.imread(str(frame.image), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        image = raw if raw.ndim == 2 else cv2.cvtColor(raw[..., :3],
                                                       cv2.COLOR_BGR2GRAY)
        return image
    for frame in frames:
        yield frame.name, (lambda f=frame: read(f)), read_mask(frame.mask)


def _parquet_reader(shards, modality):
    import PIL.Image
    import pyarrow.parquet as pq

    PIL.Image.MAX_IMAGE_PIXELS = None
    want = modality.lower()
    for shard in shards:
        handle = pq.ParquetFile(shard)
        columns = set(handle.schema_arrow.names)
        wanted = [c for c in ("image", "label", "scene", "altitude", "modality")
                  if c in columns]
        table = handle.read(columns=wanted)
        rows = table.num_rows
        modes = (table["modality"].to_pylist() if "modality" in wanted
                 else [want] * rows)
        if not any(m.lower() == want for m in modes):
            continue                                   # a shard of the other half
        scenes = (table["scene"].to_pylist() if "scene" in wanted
                  else ["?"] * rows)
        highs = (table["altitude"].to_pylist() if "altitude" in wanted
                 else ["?"] * rows)
        images, labels = table["image"].to_pylist(), table["label"].to_pylist()
        for i in range(rows):
            if modes[i].lower() != want:
                continue
            name = f"{scenes[i]}_{highs[i]}_{shard.stem}_{i:03d}"
            label = np.array(PIL.Image.open(io.BytesIO(labels[i]["bytes"])))
            cell = images[i]

            def read(cell=cell):
                frame = np.array(PIL.Image.open(io.BytesIO(cell["bytes"])))
                return frame if frame.ndim == 2 else frame[..., 0]
            yield name, read, label


def components_of(label, spec, gates):
    """Every raw component with the gate that stopped it, or None if it passed."""
    import cv2

    frame_area = float(label.shape[0] * label.shape[1])
    rows = []
    for class_id in spec.thing_ids:
        mask = (label == class_id).astype(np.uint8)
        if not mask.any():
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask,
                                                              connectivity=8)
        for i in range(1, count):
            x, y, w, h, area = (int(stats[i, k]) for k in range(5))
            instance = Instance(label=0, class_id=class_id,
                                box=(x, y, x + w, y + h), area=area)
            rows.append({"class": spec.name_of(class_id), "box": (x, y, w, h),
                         "area": area, "fill": area / max(w * h, 1),
                         "aspect": max(w, h) / max(min(w, h), 1),
                         "reject": reject_reason(instance, frame_area, gates)})
    return rows


def measure(reader, spec, gates, stride, scan_limit):
    """Pass one: what every frame holds, from the label maps alone."""
    seen = []
    for n, (name, read_image, label) in enumerate(reader):
        if n % stride:
            continue
        rows = components_of(label, spec, gates)
        kept = [r for r in rows if r["reject"] is None]
        seen.append({"name": name, "rows": rows, "kept": kept,
                     "read_image": read_image, "label": label})
        if scan_limit and len(seen) >= scan_limit:
            break
    reference = {}
    for name in spec.things:
        sizes = [r["area"] for f in seen for r in f["kept"] if r["class"] == name]
        aspects = [r["aspect"] for f in seen for r in f["kept"]
                   if r["class"] == name]
        if len(sizes) >= 50:
            reference[name] = (float(np.median(sizes)), float(np.median(aspects)))
    return seen, reference


def is_merge_shaped(row, reference, area_factor=1.6, square=0.72, stretch=1.5):
    """The two shapes a merge leaves behind -- see `aerial.drop_merge_profile`."""
    got = reference.get(row["class"])
    if got is None or row["area"] <= area_factor * got[0]:
        return None
    if row["aspect"] < square * got[1]:
        return "yan yana"
    if row["aspect"] > stretch * got[1]:
        return "uç uca"
    return None


def wanted(frame, mode, reference):
    kept = frame["kept"]
    classes = {r["class"] for r in kept}
    if mode == "all":
        return bool(frame["rows"])
    if mode == "vehicle-only":
        return bool(kept) and classes == {"vehicle"}
    if mode == "truck-only":
        return bool(kept) and classes == {"truck"}
    if mode == "has-truck":
        return "truck" in classes
    if mode == "merge-suspect":
        return any(is_merge_shaped(r, reference) for r in kept)
    if mode == "empty":
        return bool(frame["rows"]) and not kept
    raise SystemExit(f"unknown --filter {mode}")


def paint(label):
    out = np.zeros((*label.shape, 3), np.uint8)
    for value, colour in COLORS.items():
        out[label == value] = colour
    unknown = ~np.isin(label, list(COLORS))
    out[unknown] = (255, 0, 255)                       # loud on purpose
    return out


def render(page, frames, spec, gates, mode, reference, out, dpi):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = len(frames)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.6 * rows), squeeze=False)
    for r, frame in enumerate(frames):
        label = frame["label"]
        image = frame["read_image"]()
        if image is None:
            image = np.zeros_like(label, np.uint8)
        comp, kept, _ = decompose(label, spec, gates, mode)
        # A thermal frame's 8 bits rarely span 0-255; stretched, the scene is
        # visible under the masks instead of a black rectangle.
        low, high = float(image.min()), float(image.max())
        stretched = ((image.astype(float) - low) / max(high - low, 1) * 255)
        base = cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        axes[r][0].imshow(image, cmap="inferno")
        axes[r][0].set_title(f"{frame['name']}\norijinal termal",
                             fontsize=10.5, weight="bold", loc="left")
        axes[r][1].imshow(paint(label))
        axes[r][1].set_title("SegFly semantik maske", fontsize=10.5,
                             weight="bold", loc="left")

        canvas = base.astype(float) * 0.62
        for k, instance in enumerate(kept):
            canvas[comp == instance.label] = INSTANCE[k % len(INSTANCE)]
        axes[r][2].imshow(canvas.astype(np.uint8))
        axes[r][2].set_title(
            f"decompose() → {len(kept)} hedef, "
            f"{sum(1 for x in frame['rows'] if x['reject'])} red",
            fontsize=10.5, weight="bold", loc="left")

        for k, instance in enumerate(kept):
            x0, y0, x1, y1 = (int(v) for v in instance.box)
            colour = np.array(INSTANCE[k % len(INSTANCE)]) / 255
            name = spec.name_of(instance.class_id)
            row = next((x for x in frame["kept"]
                        if x["box"] == (x0, y0, x1 - x0, y1 - y0)), None)
            flag = is_merge_shaped(row, reference) if row else None
            axes[r][2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, lw=1.3,
                                           ec="red" if flag else "white"))
            axes[r][0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, ec=colour, lw=1.3))
            tag = f"#{k + 1} {name} {instance.area}px"
            if flag:
                tag += f"\nŞÜPHELİ: {flag}"
            # Anchor by the end nearer the edge, so a label on a target at the
            # right of the frame stays inside the picture instead of running
            # off it and out of the page.
            right = x0 > label.shape[1] * 0.6
            axes[r][2].text(x1 if right else x0, y0 - 4, tag, fontsize=7.5,
                            weight="bold", ha="right" if right else "left",
                            color="red" if flag else colour,
                            bbox=dict(fc="black", ec="none", alpha=.6, pad=1))
        for row in frame["rows"]:
            if not row["reject"]:
                continue
            x, y, w, h = row["box"]
            axes[r][2].add_patch(Rectangle((x, y), w, h, fill=False, lw=1.1,
                                           ec="deepskyblue", ls="--"))
            axes[r][2].text(x, y + h + 11, f"{row['reject']} ({row['area']}px)",
                            fontsize=7, color="deepskyblue", weight="bold")
        for c in range(3):
            axes[r][c].axis("off")
    fig.suptitle(f"sayfa {page}   ·   beyaz kutu = kabul · kırmızı = kaynaşma "
                 f"şüphesi · mavi kesikli = kapıda reddedildi", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = out / f"sayfa_{page:03d}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="parquet shards, or the images/+labels/ tree")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec", default="segfly")
    parser.add_argument("--modality", default="thermal")
    parser.add_argument("--mode", default="components",
                        choices=("components", "watershed"))
    parser.add_argument("--filter", default="all",
                        choices=("all", "vehicle-only", "truck-only",
                                 "has-truck", "merge-suspect", "empty"))
    parser.add_argument("--drop-truck", action="store_true",
                        help="take `truck` out of the thing classes entirely")
    parser.add_argument("--limit", type=int, default=48,
                        help="how many matching frames to draw (0 = all)")
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1,
                        help="look at every Nth frame -- SegFly is consecutive "
                             "flight frames, so neighbours are near-duplicates")
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="stop scanning after N frames (0 = the whole set)")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--min-area", type=int, default=48)
    parser.add_argument("--min-side", type=int, default=4)
    parser.add_argument("--max-area", type=float, default=0.9)
    parser.add_argument("--fill", type=float, default=0.25)
    args = parser.parse_args()

    spec = SPECS[args.spec]
    if args.drop_truck:
        spec = replace(spec, things=tuple(t for t in spec.things
                                          if t != "truck"))
    gates = InstanceGates(min_area=args.min_area, min_side=args.min_side,
                          max_area=args.max_area, fill=args.fill)
    args.out.mkdir(parents=True, exist_ok=True)

    reader, count, kind = load_source(args.root, args.modality, spec)
    print(f"kaynak: {args.root}  ({kind}, {count} birim)")
    print(f"sınıflar: {spec.things}   kapılar: {gates}")
    print("taranıyor (yalnız etiket haritaları okunuyor, görüntüler değil)...")

    seen, reference = measure(reader, spec, gates, max(args.stride, 1),
                              args.scan_limit)
    for name, (area, aspect) in reference.items():
        print(f"   {name}: tek nesne profili — alan medyanı {area:.0f} px, "
              f"en-boy medyanı {aspect:.2f}")
    for name in spec.things:
        if name not in reference:
            print(f"   {name}: profil çıkarmak için yeterli örnek yok, "
                  f"kaynaşma şüphesi işaretlenmeyecek")

    picked = [f for f in seen if wanted(f, args.filter, reference)]
    total_kept = sum(len(f["kept"]) for f in seen)
    suspect = sum(1 for f in seen for r in f["kept"]
                  if is_merge_shaped(r, reference))
    print(f"\ntaranan {len(seen)} kare · {total_kept} kabul edilen hedef · "
          f"{suspect} kaynaşma şüphelisi (%{suspect / max(total_kept, 1) * 100:.1f})")
    counts = {m: sum(1 for f in seen if wanted(f, m, reference))
              for m in ("vehicle-only", "truck-only", "has-truck",
                        "merge-suspect", "empty")}
    for key, value in counts.items():
        print(f"   {key:<15}{value:>7} kare")
    print(f"\n--filter {args.filter} → {len(picked)} kare eşleşti")

    with (args.out / "ozet.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kare", "sinif", "x", "y", "w", "h", "alan", "fill",
                         "en_boy", "sonuc", "kaynasma_suphesi"])
        for frame in seen:
            for row in frame["rows"]:
                x, y, w, h = row["box"]
                writer.writerow([frame["name"], row["class"], x, y, w, h,
                                 row["area"], f"{row['fill']:.3f}",
                                 f"{row['aspect']:.2f}",
                                 row["reject"] or "kabul",
                                 is_merge_shaped(row, reference) or ""])
    print(f"yazıldı: {args.out / 'ozet.csv'}  (her bileşen için bir satır)")

    if args.limit:
        picked = picked[:args.limit]
    pages = 0
    for start in range(0, len(picked), args.per_page):
        pages += 1
        path = render(pages, picked[start:start + args.per_page], spec, gates,
                      args.mode, reference, args.out, args.dpi)
        print(f"yazıldı: {path}")
    if not pages:
        print("çizilecek kare yok -- filtreyi gevşet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
