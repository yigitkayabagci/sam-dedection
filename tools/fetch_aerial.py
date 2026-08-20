#!/usr/bin/env python3
"""Get an aerial RGB-T dataset onto the machine that will train on it.

Unlike Anti-UAV410 there is no single canonical URL here: Kust4K comes from
figshare, SegFly from Hugging Face, Caltech Aerial RGB-T from GitHub releases,
and every one of them is a click-through or a login away. So this takes
whatever you already have -- a folder on your Drive, an archive on your Drive,
a direct URL, a Google Drive file id -- and turns it into an extracted dataset
on **local disk**, verified against a `DatasetSpec`.

**Why the copy off Drive is not optional.** Training reads thousands of PNGs in
random order and decodes a semantic map for each. The Drive FUSE mount serves
small files at a few hundred a second, an order of magnitude under what the GPU
consumes, and the run ends up loader-bound with the accelerator idle. Copying is
a one-off sequential read, which is the access pattern Drive is actually good
at. An archive is better still: one file in, one extraction out, no FUSE round
trip per image.

Drive is the right place for exactly two things in this project, and neither is
the dataset: the checkpoint, and the instance index.

Usage:
    # an archive sitting on your mounted Drive -- the fast path
    python tools/fetch_aerial.py --spec kust4k \\
        --source /content/drive/MyDrive/datasets/Kust4K.zip --dest /content/data

    # a folder on your Drive
    python tools/fetch_aerial.py --spec kust4k \\
        --source /content/drive/MyDrive/datasets/Kust4K --dest /content/data

    # a direct URL, or a Google Drive file id
    python tools/fetch_aerial.py --spec segfly --source https://... --dest /content/data
    python tools/fetch_aerial.py --spec kust4k --source gdrive:1A2B3C... --dest /content/data
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS, list_frames, probe_classes  # noqa: E402

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


# --------------------------------------------------------------------------
# Getting the bytes here
# --------------------------------------------------------------------------


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def safe_target(member: str, dest: Path) -> Path | None:
    """Where `member` may be written, or None if it escapes `dest`.

    Archive members carry whatever path the writer put in them, `../../etc`
    very much included. Nothing here trusts an archive's own idea of where it
    goes. Same rule as `tools/fetch_antiuav410.py`, for the same reason.
    """
    if member.endswith("/"):
        return None
    path = PurePosixPath(member)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return None
    target = (dest / member).resolve()
    return target if target.is_relative_to(dest.resolve()) else None


def fetch(source: str, staging: Path, quiet: bool = False) -> Path:
    """Bring `source` to `staging` and return the local path.

    A local path is returned as-is -- copying an already-local dataset would
    double the disk for nothing. Only remote sources are downloaded.
    """
    if source.startswith("gdrive:"):
        try:
            import gdown
        except ImportError as exc:                       # pragma: no cover
            raise SystemExit("gdown is required for gdrive: sources "
                             "(pip install gdown)") from exc
        staging.mkdir(parents=True, exist_ok=True)
        target = staging / "download.zip"
        gdown.download(id=source[len("gdrive:"):], output=str(target), quiet=quiet)
        return target

    if source.startswith(("http://", "https://")):
        staging.mkdir(parents=True, exist_ok=True)
        name = PurePosixPath(source.split("?")[0]).name or "download.zip"
        target = staging / name
        if not quiet:
            print(f">> downloading {source} -> {target}")
        urlretrieve(source, target)                      # noqa: S310 - explicit user URL
        return target

    path = Path(source).expanduser()
    if not path.exists():
        raise SystemExit(f"{path}: no such file or directory.")
    return path


def unpack(archive: Path, dest: Path, workers: int = 8, quiet: bool = False) -> None:
    """Extract `archive` into `dest`, in parallel where the format allows it.

    Zip members are independently seekable so threads help -- decompression
    gives up the GIL and these archives are tens of thousands of small PNGs.
    Tar is a stream and is not seekable member-by-member, so it extracts
    serially; that is the format's property, not a shortcut.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(archive):
        if not quiet:
            print(f">> extracting {archive.name} (tar, serial) -> {dest}")
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and safe_target(m.name, dest) is not None]
            tf.extractall(dest, members=members)         # noqa: S202 - members filtered
        return

    if not zipfile.is_zipfile(archive):
        size = archive.stat().st_size
        head = archive.read_bytes()[:300].decode("utf-8", "replace") if size < 1_000_000 else ""
        raise SystemExit(
            f"{archive} is not an archive this understands ({size} bytes). "
            f"A few-kilobyte 'archive' is usually a login or quota page saved "
            f"under the name of the file you wanted.\n\n{head}")

    with zipfile.ZipFile(archive) as zf:
        wanted = [info for info in zf.infolist()
                  if not info.is_dir() and safe_target(info.filename, dest) is not None]
        if not quiet:
            total = sum(info.file_size for info in wanted)
            print(f">> extracting {len(wanted)} files ({total / 2**30:.2f} GiB) -> {dest}")

        # One handle per thread: ZipFile's shared file object is not safe to
        # read from concurrently, and the failure mode is interleaved bytes
        # rather than an exception.
        local = threading.local()

        def one(info: zipfile.ZipInfo) -> None:
            handle = getattr(local, "zf", None)
            if handle is None:
                handle = local.zf = zipfile.ZipFile(archive)
            target = safe_target(info.filename, dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, wanted))


def mirror(folder: Path, dest: Path, quiet: bool = False) -> None:
    """Copy a dataset folder to local disk, skipping what is already there."""
    if not quiet:
        print(f">> copying {folder} -> {dest} (this is the slow shape; an "
              f"archive on Drive is much faster)")
    shutil.copytree(folder, dest, dirs_exist_ok=True)


# --------------------------------------------------------------------------
# Saying what arrived
# --------------------------------------------------------------------------


def describe(root: Path, spec, modality: str, sample: int = 32) -> str:
    """What the download actually contains, checked against the spec.

    The palette in a `DatasetSpec` is an assumption about an archive, and a
    wrong one produces a training run with zero instances and no error to
    explain it. Reporting the values *present* next to the values *expected* is
    what turns that into a two-line fix before any GPU time is spent.
    """
    frames = list_frames(root, spec, modality)
    present = probe_classes(frames, limit=sample)
    expected = {value: name for name, value in spec.classes.items()}
    things = set(spec.thing_ids)

    lines = [f"{spec.name}: {len(frames)} {modality} image/mask pairs under {root}",
             f"  {sum(1 for f in frames if f.pair)} carry a registered twin",
             "",
             f"| value | in spec | thing | frames of {min(sample, len(frames))} |",
             "|---:|---|---|---:|"]
    for value, count in present.items():
        lines.append(f"| {value} | {expected.get(value, '**not in spec**')} | "
                     f"{'yes' if value in things else ''} | {count} |")

    missing = sorted(things - set(present))
    if missing:
        lines += ["", f"!! no pixels of thing class(es) {missing} in the sample. "
                      f"Either the palette differs from the spec (fix it with "
                      f"dataclasses.replace) or these classes are rare."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   help="Drive path, archive, http(s) URL, or gdrive:<file id>.")
    p.add_argument("--dest", required=True, help="Local directory to land it in.")
    p.add_argument("--spec", default="kust4k", choices=sorted(SPECS))
    p.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--keep-archive", action="store_true",
                   help="Do not delete a downloaded archive after extraction.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    dest = Path(args.dest).expanduser()
    spec = SPECS[args.spec]
    staging = dest.parent / "_staging"

    local = fetch(args.source, staging, args.quiet)
    if local.is_dir():
        if local.resolve() != dest.resolve():
            mirror(local, dest, args.quiet)
    elif is_archive(local) or zipfile.is_zipfile(local) or tarfile.is_tarfile(local):
        unpack(local, dest, args.workers, args.quiet)
        if not args.keep_archive and staging in local.parents:
            local.unlink()
            shutil.rmtree(staging, ignore_errors=True)
    else:
        raise SystemExit(f"{local}: neither a directory nor an archive.")

    print()
    print(describe(dest, spec, args.modality))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
