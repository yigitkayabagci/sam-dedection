#!/usr/bin/env python3
"""The aerial RGB-T sets, downloaded and laid out the way the specs glob them.

One command per dataset, no staging archives in Drive by hand:

    python tools/fetch_datasets.py kust4k    --dest /content/data/Kust4K
    python tools/fetch_datasets.py vtuav_vis --dest /content/data/VTUAV_VIS
    python tools/fetch_datasets.py segfly    --dest /content/data/SegFly
    python tools/fetch_datasets.py all       --root /content/data

Every URL below was checked against the live host rather than copied off a
paper, because each of the three is served in a way that breaks the obvious
approach:

**Kust4K.** The figshare link on the article page,
`ndownloader/articles/29476610/versions/3`, answers **HTTP 202 with an empty
body** -- it asks figshare to *build* a zip of the whole article and returns
before it exists, so a downloader sees a zero-byte file and a success code.
The per-file endpoints under `ndownloader.figshare.com/files/<id>` are built
already, answer 206 to a range request, and come with the publisher's md5, so
those are what this uses. It also means the 1.66 GB `RGB.zip` can be skipped
when only the thermal half is being trained on.

The three archives are **flat** -- `00001D.png` at the top level of each, no
directories -- so each is extracted into the folder its spec expects (`tir/`,
`label/`, `rgb/`). Extracting them side by side would collide: all three use
the same 4 024 filenames.

**VTUAV.** The mask split is a Google Drive *folder*, and the files inside are
8-17 GB each, which puts them past the size where Drive serves a file directly
and into the "cannot scan for viruses" interstitial. `gdown` replays that form;
plain `curl` gets an HTML page with a 200 on it. The folder holds eight zips --
`training/train_001..003` and `test/test_001..005` -- and only `train_001`
(8.5 GB) is fetched by default, because it already carries 14 sequences,
26 059 registered pairs and 875 RGB masks. `--parts` asks for more.

**SegFly** is parquet on the Hub, not files, so it goes through
`tools/export_hf_dataset.py`, which writes the PNG layout the spec globs.

Downloads resume (`Range`), verify against the publisher's md5 where there is
one, and delete the archive once it has been extracted, which halves the peak
disk a Colab session needs.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Part:
    """One archive: where it comes from, and where its contents belong."""

    name: str
    into: str = ""              # subdirectory to extract into; "" keeps the
                                # archive's own layout
    url: str | None = None      # a direct HTTP download
    drive: str | None = None    # a Google Drive file id
    size: int = 0               # bytes, for the "this will take a while" line
    md5: str | None = None
    default: bool = True        # fetched when --parts is not given


@dataclass(frozen=True)
class Recipe:
    name: str
    note: str
    parts: tuple[Part, ...] = ()
    hub: str | None = None      # a Hugging Face dataset id instead of archives
    modality: str | None = None

    def chosen(self, names: tuple[str, ...] | None) -> list[Part]:
        if not names:
            return [p for p in self.parts if p.default]
        known = {p.name: p for p in self.parts}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(
                f"{self.name}: no part named {unknown} -- have "
                f"{sorted(known)}")
        return [known[n] for n in names]


# Verified 2026-08: `GET https://api.figshare.com/v2/articles/29476610/files`.
# The md5s are the publisher's `computed_md5`.
KUST4K = Recipe(
    name="kust4k",
    note="4 024 registered RGB-TIR pairs at 640x512, 9 classes (Sci. Data 2025)",
    parts=(
        Part("tir", into="tir", size=1_032_806_526,
             url="https://ndownloader.figshare.com/files/55979165",
             md5="90562a17a4160600b2a12359d2c48391"),
        Part("labels", into="label", size=20_500_353,
             url="https://ndownloader.figshare.com/files/55979159",
             md5="c3111062f64dd06aeacdf16c2c5d1ee5"),
        # The RGB half is only needed for stage-A distillation, where it is the
        # teacher's input. Training the thermal encoder alone does not read it.
        Part("rgb", into="rgb", size=1_660_347_962, default=False,
             url="https://ndownloader.figshare.com/files/55979147",
             md5="d0bc9895100f339b1e13f40f2efe532f"),
    ),
)

# Verified 2026-08 by listing the Drive folder `11E-WPkCPVL49hOKRdCzfgQULmGU8pyz8`
# (the "Video instance segmentation" release on https://zhang-pengyu.github.io/DUT-VTUAV/)
# and reading each zip's central directory over range requests.
VTUAV_VIS = Recipe(
    name="vtuav_vis",
    note="VTUAV mask split: 1920x1080 RGB-T video with per-frame target masks",
    # Read off the three training archives' central directories. The target
    # kinds are the sequence-name prefixes, and they are **not** spread evenly
    # -- which matters more than the sizes do:
    #
    #   train_001   9.1 GB  14 seq    875 masks   bike 1, bus 4, c-vehicle 1, car 8
    #   train_002  16.1 GB  18 seq  1 408 masks   car 4, elebike 3, excavator 2,
    #                                             pedestrian 9
    #   train_003  17.9 GB  18 seq  1 778 masks   pedestrian 15, train 1,
    #                                             tricycle 1, truck 1
    #
    # **train_001 contains no pedestrians at all.** On its own it teaches an
    # encoder that a target is a vehicle, which for a drone tracker is a hole
    # rather than a bias. It is still the default, because it is the smallest
    # and it saturates stage A by itself (26 059 pairs against DISTILL_PAIRS =
    # 20 000) -- but stage B wants at least train_003 beside it.
    #
    # It also decides how much a held-out number is worth: 14 sequences split
    # 11/2/1, so the test set is a single flight and its quirks dominate any
    # comparison between two runs. All three give 50 sequences and a 40/5/5.
    parts=(
        Part("train_001", drive="1LW1jyldaHmFolmcNzbnWFtzGr4gdfTiX",
             size=9_080_000_000),
        Part("train_002", drive="1wKffvGkpALbtXibnj-F-erSuu2CvamYo",
             size=16_100_000_000, default=False),
        Part("train_003", drive="17h2zBfmOwHFllw40Zln7fuhSFs0vXvvf",
             size=17_900_000_000, default=False),
        # The authors' own held-out sequences. `split_frames` already holds
        # whole sequences out of `train_001`, so these are only worth the disk
        # when the canonical split is the point.
        Part("test_001", drive="1LLYlNlV-V1jUcjT4kbhHSMDoQGSBs8LU",
             size=16_449_400_000, default=False),
        Part("test_002", drive="1bUflxddDafrUZOweTt7jGP0icY41Afwg",
             size=16_000_000_000, default=False),
        Part("test_003", drive="15o5AM9Qo1kjnJq1PTcd2x4VOLT48dVWT",
             size=16_000_000_000, default=False),
        Part("test_004", drive="1UwA4mPpNjkbYtMMgI0gpzQQjkJ5QRkjX",
             size=14_000_000_000, default=False),
        Part("test_005", drive="1A5o4zm3sSqR-wFmqmxC7gdAngAQOkYAc",
             size=15_000_000_000, default=False),
    ),
)

# Verified 2026-08 against the mirror: anonymous HTTPS, HTTP 206 to a range
# request, `application/zip`, real zip magic. The authors distribute this
# through Baidu, which in practice needs an account and a Chinese phone number;
# the mirror is what makes it usable at all.
#
# 28 442 registered RGB-TIR pairs from a UAV, day and night. Its labels are
# oriented boxes in XML, so it is **stage-A data only** -- and stage A reads no
# labels, which is exactly why a set nobody can use for segmentation is worth
# 14 GB here.
#
# Layout inside the archives, read from their central directories:
#     train/trainimg/00001.jpg    RGB          17 991
#     train/trainimgr/00001.jpg   thermal      17 991
#     train/trainlabel/*.xml      oriented boxes, unused
# and the same shape for val/ and test/. Extracted as-is: the globs
# `**/*img/*.jpg` and `**/*imgr/*.jpg` pick the two halves apart, and they
# cannot cross-match because a glob component must match in full.
#
# Every image is 840x712 with a 100 px band of pure white on all four sides;
# `SPECS["dronevehicle"].border` discards it at read time. See that spec.
DRONEVEHICLE = Recipe(
    name="dronevehicle",
    note="28 442 registered RGB-TIR UAV pairs, day and night (RA-L 2022)",
    parts=(
        Part("train", size=8_880_000_000,
             url="https://huggingface.co/datasets/McCheng/DroneVehicle/"
                 "resolve/main/train.zip"),
        Part("test", size=4_430_000_000, default=False,
             url="https://huggingface.co/datasets/McCheng/DroneVehicle/"
                 "resolve/main/test.zip"),
        Part("val", size=720_000_000, default=False,
             url="https://huggingface.co/datasets/McCheng/DroneVehicle/"
                 "resolve/main/val.zip"),
    ),
)

SEGFLY = Recipe(
    name="segfly",
    note=">15 000 aligned RGB-T pairs over three altitudes (ECCV 2026)",
    hub="markus-42/SegFly",
    modality="thermal",
)

RECIPES: dict[str, Recipe] = {
    r.name: r for r in (KUST4K, VTUAV_VIS, DRONEVEHICLE, SEGFLY)}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GB"


def checksum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def http_download(url: str, path: Path, expect: int = 0, quiet: bool = False,
                  session=None, params: dict | None = None) -> Path:
    """Stream `url` to `path`, resuming a partial file rather than restarting.

    Resume matters more than it looks: these are gigabyte archives over a link
    that occasionally drops, and starting again from zero on a 1 GB file is how
    a fetch cell turns into an hour.
    """
    import requests

    session = session or requests.Session()
    params = params or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    have = path.stat().st_size if path.exists() else 0
    if expect and have == expect:
        if not quiet:
            print(f"   already complete ({human(have)})")
        return path

    headers = {"Range": f"bytes={have}-"} if have else {}
    response = session.get(url, params=params, headers=headers, stream=True,
                           timeout=120)
    if have and response.status_code == 200:
        have = 0                      # server ignored the range; start over
    elif response.status_code not in (200, 206):
        raise RuntimeError(f"{url}: HTTP {response.status_code}")
    response.raise_for_status()

    total = expect or (int(response.headers.get("content-length", 0)) + have)
    done = have
    with path.open("ab" if have else "wb") as handle:
        for block in response.iter_content(1 << 20):
            handle.write(block)
            done += len(block)
            if not quiet and total and done % (64 << 20) < (1 << 20):
                print(f"   {human(done)} / {human(total)} "
                      f"({100 * done / total:.0f}%)", flush=True)
    return path


QUOTA = "quota"


def staged(name: str, search: SequenceABC[str]) -> Path | None:
    """A copy of `<name>.zip` already sitting somewhere we can read.

    The escape hatch for Drive's download quota, and the reason it works: a
    file in *your own* Drive is not a widely-shared file, so reading it through
    the Colab mount is an ordinary authenticated read with no shared-file quota
    attached. Three clicks in the Drive web UI ("Make a copy") turn the one
    into the other.
    """
    for folder in search:
        candidate = Path(folder).expanduser() / f"{name}.zip"
        if candidate.is_file() and candidate.stat().st_size > 1 << 20:
            return candidate
    return None


def drive_confirm(file_id: str) -> tuple[str, object, dict]:
    """Resolve a Drive id past the "can't scan for viruses" form.

    Past a few hundred megabytes Drive stops serving the file and serves that
    form instead -- with a 200 on it, so a plain downloader saves the HTML and
    reports success. Replaying the form's hidden fields yields the real URL.
    Separate from gdown on purpose: it is a different code path against the
    same endpoint, and when one is refused the other sometimes is not.
    """
    import requests

    session = requests.Session()
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    response = session.get(url, stream=True, timeout=60,
                           headers={"Range": "bytes=0-1"})
    kind = response.headers.get("content-type", "")
    if "text/html" not in kind:
        response.close()
        return url + "&confirm=t", session, {}

    page = session.get(url, timeout=60).text
    if "Too many users" in page:
        raise RuntimeError(QUOTA)
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', page))
    action = re.search(r'action="([^"]+)"', page)
    if not action:
        raise RuntimeError(f"Drive served a page with no download form for {file_id}")
    return action.group(1).replace("&amp;", "&"), session, fields


def drive_download(file_id: str, path: Path, quiet: bool = False,
                   attempts: int = 3) -> Path:
    """A Drive file, by whichever of two routes answers.

    Quota refusals ("Too many users have viewed or downloaded this file
    recently") are not a property of the file so much as of who is asking:
    Colab shares its egress addresses with a great many people, so a file that
    serves fine elsewhere can be refused there. That makes it worth retrying,
    and worth trying the second route, before giving up on the day.
    """
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                       check=True)
        import gdown

    path.parent.mkdir(parents=True, exist_ok=True)
    trouble: list[str] = []

    for attempt in range(1, attempts + 1):
        try:
            out = gdown.download(id=file_id, output=str(path), quiet=quiet,
                                 resume=True)
            if out is not None:
                return Path(out)
            trouble.append(f"gdown #{attempt}: refused without saying why")
        except Exception as error:                       # gdown raises its own
            trouble.append(f"gdown #{attempt}: {str(error).strip()[:90]}")

        try:
            url, session, fields = drive_confirm(file_id)
            return http_download(url, path, quiet=quiet, session=session,
                                 params=fields)
        except Exception as error:
            trouble.append(f"direct #{attempt}: {str(error).strip()[:90]}")

        if attempt < attempts:
            pause = 15 * attempt
            print(f"   both routes refused; retrying in {pause}s "
                  f"({attempt}/{attempts})")
            time.sleep(pause)

    raise RuntimeError(
        f"Drive would not serve {file_id} after {attempts} attempts:\n"
        + "\n".join(f"    {line}" for line in trouble)
        + "\n\n  This is Drive's download quota, and it is about who is asking "
          "rather than\n  about the file -- Colab shares its addresses with a "
          "great many people, so a\n  file that downloads fine elsewhere can be "
          "refused there. Three ways out:\n\n"
          "  1. Copy it into your own Drive, which is the reliable one. Open\n"
          "       https://drive.google.com/drive/folders/"
          "11E-WPkCPVL49hOKRdCzfgQULmGU8pyz8\n"
          "     right-click the zip you want, `Make a copy`, and move the copy "
          "into\n     MyDrive/datasets/. Copying is a server-side operation and "
          "is not a\n     download, so the quota does not apply -- and reading "
          "your own file\n     through the Colab mount does not either. Then "
          "re-run this cell: it\n     looks in MyDrive/datasets/ before it "
          "touches the network at all.\n\n"
          "  2. Wait. The refusal usually clears within a few hours.\n\n"
          "  3. Train on Kust4K and SegFly alone for now -- figshare and the "
          "Hub have\n     no quota. Know what it costs first: VTUAV is the only "
          "set here whose\n     masks somebody drew, so without it the score is "
          "measured on instances\n     `decompose` reconstructed, which grades "
          "the reconstruction as much as\n     the model. It is also what stage "
          "A distils from (`DISTILL_ROOT`), so\n     turn `PRETRAIN` off too.")


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


def masked_members(archive: zipfile.ZipFile) -> list[str]:
    """Only the frames that carry a mask, plus the masks themselves.

    VTUAV annotates every 30th frame, so a full extraction spends 9 GB of disk
    to make 875 trainable frames. This keeps the annotated frames in both
    modalities and drops the other 96 %, which is the difference between a
    smoke run that fits anywhere and one that needs a Pro+ disk.

    It is the wrong choice for stage A: distillation reads no labels and wants
    every registered pair it can get. `--frames all` is the default for that
    reason.
    """
    wanted: set[tuple[str, str]] = set()
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) >= 4 and parts[-3] == "mask":
            wanted.add((parts[0], Path(parts[-1]).stem))

    keep = []
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) >= 4 and parts[-3] == "mask":
            keep.append(name)
        elif len(parts) >= 3 and (parts[0], Path(parts[-1]).stem) in wanted:
            keep.append(name)
    return keep


def extract(archive_path: Path, dest: Path, into: str = "",
            frames: str = "all", quiet: bool = False) -> int:
    """Unpack into `dest/into`, returning how many files were written."""
    target = dest / into if into else dest
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = (masked_members(archive) if frames == "masked"
                   else archive.namelist())
        if not quiet:
            print(f"   extracting {len(members)} entries -> {target}")
        archive.extractall(target, members=members)
    return len(members)


# Where a hand-staged archive is looked for before the network is touched.
# `Make a copy` in Drive puts a shared file into MyDrive, and reading your own
# file has no shared-file quota on it -- see `drive_download`.
STAGING = ("/content/drive/MyDrive/datasets",
           "/content/drive/MyDrive",
           "/content/staging")


def fetch_part(part: Part, dest: Path, work: Path, frames: str,
               keep: bool, quiet: bool,
               staging: SequenceABC[str] = STAGING) -> None:
    archive = work / f"{part.name}.zip"
    print(f"-- {part.name}"
          + (f"  ({human(part.size)})" if part.size else ""))

    already = staged(part.name, staging)
    if already is not None:
        print(f"   using the copy already at {already}")
        archive = already
        keep = True                      # never delete something we did not fetch
    elif part.drive:
        drive_download(part.drive, archive, quiet=quiet)
    else:
        http_download(part.url, archive, expect=part.size, quiet=quiet)

    if part.md5:
        got = checksum(archive)
        if got != part.md5:
            raise RuntimeError(
                f"{part.name}: md5 {got}, expected {part.md5}. The download is "
                f"corrupt or truncated -- delete {archive} and run again.")
        print("   md5 ok")

    extract(archive, dest, part.into, frames=frames, quiet=quiet)
    if not keep:
        archive.unlink(missing_ok=True)


def fetch(name: str, dest: Path, parts: tuple[str, ...] | None = None,
          frames: str = "all", keep: bool = False, quiet: bool = False,
          limit: int | None = None,
          staging: SequenceABC[str] = STAGING) -> Path:
    """One dataset into `dest`, in the layout `SPECS[name]` globs."""
    recipe = RECIPES[name]
    dest = Path(dest).expanduser()
    print(f"\n=== {recipe.name}: {recipe.note}")

    if recipe.hub:
        from tools.export_hf_dataset import export, verify
        result = export(recipe.hub, dest, recipe.modality, "train", limit,
                        streaming=True, quiet=quiet)
        print(f"{result['written']} rows written, {result['skipped']} skipped")
        print(verify(result["values"], recipe.name))
        return dest

    chosen = recipe.chosen(parts)
    # Report against what is actually going to be fetched. Announcing "40 GB to
    # download" and then reading three staged copies off Drive is a confusing
    # way to be right.
    ready = {p.name: staged(p.name, staging) for p in chosen}
    for name, where in ready.items():
        if where is not None:
            print(f"   {name}: staged at {where}, no download needed")
    wanted = [p for p in chosen if ready[p.name] is None]
    if wanted:
        print(f"{len(wanted)} archive(s), about {human(sum(p.size for p in wanted))} "
              f"to download")
    else:
        print("every archive is staged already; nothing to download")

    work = dest / "_archives"
    work.mkdir(parents=True, exist_ok=True)
    try:
        for part in chosen:
            fetch_part(part, dest, work, frames, keep, quiet, staging)
    finally:
        if not keep and work.exists() and not any(work.iterdir()):
            work.rmdir()
    return dest


def report(name: str, dest: Path, modality: str = "thermal") -> str:
    """What the reader actually finds there -- the only check that counts."""
    from src.training.aerial import SPECS, describe_layout, list_frames, list_pairs

    spec = SPECS[name]
    lines = [f"\n--- {dest}"]
    try:
        frames = list_frames(dest, spec, modality)
        sequences = {f.name.rsplit("/", 1)[0] for f in frames if "/" in f.name}
        lines.append(f"{len(frames)} labelled {modality} frames"
                     + (f" over {len(sequences)} sequences" if sequences else ""))
    except (FileNotFoundError, ValueError) as error:
        lines.append(f"no labelled frames: {error}".split("\n\n")[0])
        lines.append(describe_layout(dest))
    try:
        lines.append(f"{len(list_pairs(dest, spec))} registered thermal/RGB pairs")
    except (FileNotFoundError, ValueError):
        lines.append("no registered pairs (only one modality was fetched)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=[*sorted(RECIPES), "all"])
    p.add_argument("--dest", default=None,
                   help="Where this dataset goes. Defaults to <root>/<name>.")
    p.add_argument("--root", default="/content/data",
                   help="Parent directory, used when --dest is not given.")
    p.add_argument("--parts", nargs="*", default=None,
                   help="Which archives to fetch. Without it, the defaults: "
                        "Kust4K thermal+labels, VTUAV train_001.")
    p.add_argument("--frames", choices=("all", "masked"), default="all",
                   help="`masked` keeps only annotated frames and their twins "
                        "-- a twentieth of the disk, and too few pairs for "
                        "stage-A distillation.")
    p.add_argument("--limit", type=int, default=None,
                   help="Hub datasets only: stop after N rows.")
    p.add_argument("--keep", action="store_true",
                   help="Keep the archives after extracting them.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    names = sorted(RECIPES) if args.dataset == "all" else [args.dataset]
    folders = {"kust4k": "Kust4K", "vtuav_vis": "VTUAV_VIS",
               "dronevehicle": "DroneVehicle", "segfly": "SegFly"}
    for name in names:
        dest = Path(args.dest) if args.dest else Path(args.root) / folders[name]
        fetch(name, dest, tuple(args.parts) if args.parts else None,
              frames=args.frames, keep=args.keep, quiet=args.quiet,
              limit=args.limit)
        print(report(name, dest, "rgb" if name == "vtuav_vis" else "thermal"))

    free = shutil.disk_usage(args.root if not args.dest else args.dest).free
    print(f"\n{human(free)} of disk left.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
