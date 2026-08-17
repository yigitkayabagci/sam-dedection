#!/usr/bin/env python3
"""Download Anti-UAV410 straight onto the machine that will train on it.

The dataset is one 8.7 GB zip on the authors' Google Drive
(https://github.com/HwangBo94/Anti-UAV410), holding

    Anti-UAV410/{train,val,test}/<sequence>/{0.jpg, 1.jpg, ..., IR_label.json}

410 sequences, 438K annotated boxes, 640x512 8-bit mono thermal.

**It goes on local disk, not on Drive.** Training reads a few hundred thousand
small JPEGs in random order; the Drive FUSE mount serves those at a few hundred
files a second, which is slower than the GPU can consume them by more than an
order of magnitude. Colab's local disk is NVMe. The zip is deleted after
extraction by default, so the peak requirement is ~19 GB and the resting one
~10 GB.

Three things this does that `gdown | unzip` does not:

* **Checks what it downloaded.** Drive answers a quota-exceeded request with an
  HTML page and HTTP 200. Written to `Anti-UAV410.zip` that is a 3 KB file that
  fails at extraction time, twenty minutes later, with an unrelated error.
* **Extracts in parallel.** zlib releases the GIL, so threads help, and there
  are ~440K members to get through.
* **Verifies the layout it produced**, rather than assuming the archive's
  top-level folder is named what the README says it is.

Usage:
    python tools/fetch_antiuav410.py --dest /content/data
    python tools/fetch_antiuav410.py --dest /content/data --splits train val
    python tools/fetch_antiuav410.py --dest /content/data --zip /content/drive/MyDrive/Anti-UAV410.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# https://github.com/HwangBo94/Anti-UAV410 -> Google Drive
FILE_ID = "1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR"
ARCHIVE = "Anti-UAV410.zip"
ARCHIVE_BYTES = 9_361_681_896
SPLITS = ("train", "val", "test")
LABEL_FILE = "IR_label.json"


# --------------------------------------------------------------------------
# Archive layout
# --------------------------------------------------------------------------


def split_of(member: str) -> str | None:
    """Which split a zip member belongs to, or None if it belongs to none.

    Read from the member's *directories* only. A sequence is a folder of frames
    named `0.jpg`, so matching against the whole path would put a hypothetical
    `.../test.jpg` in the test split.
    """
    parts = PurePosixPath(member).parts[:-1]
    for part in parts:
        if part.lower() in SPLITS:
            return part.lower()
    return None


def safe_target(member: str, dest: Path) -> Path | None:
    """Where `member` may be written, or None if it escapes `dest`.

    Zip members carry whatever path the writer put in them, `../../etc` very
    much included. Nothing here trusts the archive's own idea of where it goes.
    """
    if member.endswith("/"):
        return None
    path = PurePosixPath(member)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return None
    target = (dest / member).resolve()
    return target if target.is_relative_to(dest.resolve()) else None


def find_splits(root: Path) -> dict[str, Path]:
    """The `{split: directory}` map under `root`, wherever the archive put it.

    A split directory is one holding at least one sequence -- a subfolder with
    an `IR_label.json` in it. That test is what makes this independent of
    whether the archive's top level is `Anti-UAV410/`, `AntiUAV410/` or nothing
    at all, which is not worth finding out the hard way at 8.7 GB a try.
    """
    found: dict[str, Path] = {}
    candidates = [root, *(d for d in root.iterdir() if d.is_dir())] if root.is_dir() else []
    for base in candidates:
        for split in SPLITS:
            directory = base / split
            if split in found or not directory.is_dir():
                continue
            if any((d / LABEL_FILE).is_file() for d in directory.iterdir() if d.is_dir()):
                found[split] = directory
    return found


def dataset_root(splits: dict[str, Path]) -> Path:
    """The single directory holding the splits -- what `--data` wants.

    `list_sequences(root, split)` joins the two, so the splits have to share a
    parent. They do in the shipped archive; if a hand-assembled copy does not,
    saying so here beats a `FileNotFoundError` from inside a training run.
    """
    parents = {p.parent for p in splits.values()}
    if len(parents) != 1:
        raise ValueError(
            f"splits do not share a parent directory: "
            f"{ {k: str(v) for k, v in splits.items()} }"
        )
    return parents.pop()


def count_sequences(directory: Path) -> int:
    return sum(1 for d in directory.iterdir() if d.is_dir() and (d / LABEL_FILE).is_file())


def describe(splits: dict[str, Path]) -> str:
    lines = [f"{'split':<8}{'sequences':>11}  directory"]
    for split in SPLITS:
        if split in splits:
            lines.append(f"{split:<8}{count_sequences(splits[split]):>11}  {splits[split]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Download and extract
# --------------------------------------------------------------------------


def download(file_id: str, target: Path, quiet: bool = False) -> Path:
    """Fetch the archive from Google Drive, resuming a partial file if there is one."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise SystemExit(
            "gdown is required to download from Google Drive: pip install gdown"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(id=file_id, output=str(target), quiet=quiet, resume=True)
    except TypeError:  # older gdown has no resume=
        gdown.download(id=file_id, output=str(target), quiet=quiet)
    return target


def check_archive(path: Path, expect_bytes: int | None = ARCHIVE_BYTES) -> None:
    """Fail now, with the real reason, rather than at extraction time.

    A Drive download that hit the daily quota is an HTML page with a 200 next
    to it. It is a few kilobytes, and every symptom of it appears later and
    somewhere else.
    """
    if not path.is_file():
        raise SystemExit(f"{path}: nothing was downloaded.")
    size = path.stat().st_size
    if not zipfile.is_zipfile(path):
        if size < 1_000_000:
            head = path.read_bytes()[:400].decode("utf-8", "replace")
            raise SystemExit(
                f"{path} is only {size} bytes and is not a zip -- this is Google "
                f"Drive's error page, not the dataset. It usually means the "
                f"file's daily download quota is exhausted; try again later, or "
                f"add the file to your own Drive and pass --zip pointing at the "
                f"mounted copy.\n\n{head}"
            )
        raise SystemExit(f"{path} is not a zip archive ({size} bytes).")
    if expect_bytes and size != expect_bytes:
        print(f"!! {path.name} is {size} bytes, expected {expect_bytes}. "
              f"Continuing -- the archive opens -- but a resumed download that "
              f"was interrupted can look like this.")


def extract(
    archive: Path,
    dest: Path,
    splits: tuple[str, ...] = SPLITS,
    workers: int = 8,
    quiet: bool = False,
) -> None:
    """Unpack the members belonging to `splits`, in parallel.

    Extraction is decompress-then-write, and zlib gives up the GIL for the
    first half of that, so threads are worth roughly their count here. ~440K
    members makes the difference minutes rather than seconds.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        wanted = [
            info for info in zf.infolist()
            if not info.is_dir() and split_of(info.filename) in splits
            and safe_target(info.filename, dest) is not None
        ]
        if not wanted:
            raise SystemExit(
                f"{archive}: no members under {splits}. The archive's layout is "
                f"not the one this expects; first few members: "
                f"{[i.filename for i in zf.infolist()[:5]]}"
            )
        if not quiet:
            total = sum(info.file_size for info in wanted)
            print(f"extracting {len(wanted)} files ({total / 2**30:.1f} GiB) "
                  f"from {', '.join(sorted(splits))} -> {dest}")

        # One handle per thread: ZipFile's shared file object is not safe to
        # read from concurrently, and the failure is silently interleaved bytes.
        local = threading.local()

        def unpack(info: zipfile.ZipInfo) -> None:
            handle = getattr(local, "zf", None)
            if handle is None:
                handle = local.zf = zipfile.ZipFile(archive)
            target = safe_target(info.filename, dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(unpack, wanted):
                done += 1
                if not quiet and done % 20000 == 0:
                    print(f"  {done}/{len(wanted)}")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dest", required=True, help="Where the dataset is unpacked.")
    p.add_argument("--zip", default=None,
                   help="Use this archive instead of downloading (e.g. a copy on Drive).")
    p.add_argument("--file-id", default=FILE_ID, help="Google Drive file id.")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS),
                   help="Which splits to unpack. Fewer means less disk and less time.")
    p.add_argument("--workers", type=int, default=8, help="Extraction threads.")
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep the archive after extracting (needs ~9 GB more disk).")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if the splits are already there.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    wanted = tuple(args.splits)

    present = find_splits(dest)
    if not args.force and all(s in present for s in wanted):
        print(f"already present under {dest}:\n{describe(present)}")
        print(f"\n--data {dataset_root(present)}")
        return 0

    archive = Path(args.zip).expanduser().resolve() if args.zip else dest / ARCHIVE
    if args.zip:
        check_archive(archive, expect_bytes=None)
    else:
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            print(f"downloading Anti-UAV410 ({ARCHIVE_BYTES / 2**30:.1f} GiB) -> {archive}")
            download(args.file_id, archive, quiet=args.quiet)
        else:
            print(f"using the archive already at {archive}")
        check_archive(archive)

    extract(archive, dest, wanted, workers=args.workers, quiet=args.quiet)

    if not args.keep_zip and not args.zip and archive.is_file():
        archive.unlink()
        print(f"removed {archive}")

    found = find_splits(dest)
    missing = [s for s in wanted if s not in found]
    if missing:
        raise SystemExit(f"extracted, but {missing} are not there. Under {dest}: "
                         f"{sorted(d.name for d in dest.iterdir())}")
    print(f"\n{describe(found)}")
    print(f"\n--data {dataset_root(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
