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
`tools/export_hf_dataset.py`, which writes the PNG layout the spec globs. Only
half of it is worth fetching and the halves are mixed together, so that script
reads the `modality` column out of the shard footers first and downloads only
the shards that answer -- 22 GiB rather than 178 GiB. See the SEGFLY recipe.

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
    columns: tuple[str, ...] = ()   # for a `hub` recipe, the Image columns
                                # worth writing out; empty means all of them
    spec: str = ""              # the SPECS entry that reads the result; ""
                                # means the recipe's own name
    rows: int | None = None     # hub recipes: default row cap (CLI --limit
                                # overrides); None means every matching row
    passthrough: bool = False   # hub recipes: keep the publisher's bytes for
                                # photographic columns instead of PNG
    spread: bool = False        # hub recipes: a capped export samples evenly
                                # spaced shards instead of the first ones
    snapshot: str | None = None  # a Hub repo of *plain files* (not parquet):
                                # fetched with snapshot_download, each part's
                                # `into` used as its allow_pattern
    stream: bool = False        # the parts are consecutive slices of ONE
                                # tar.gz: never saved to disk, decompressed
                                # straight off the network (see LASHER)
    extras: tuple[tuple[str, str], ...] = ()   # (filename, source)
                                # small sidecar files written into the dataset
                                # root -- manifests the reader needs, not data.
                                # `source` is a full URL, a `drive:<file id>`,
                                # or a bare figshare file id (Kust4K's form,
                                # which predates the other two)

    def chosen(self, names: tuple[str, ...] | None) -> list[Part]:
        if not names:
            return [p for p in self.parts if p.default]
        known = {p.name: p for p in self.parts}
        unknown = [n for n in names if n not in known]
        if unknown:
            # This is the first thing `fetch` does, so nothing has been
            # downloaded yet and the whole command is refused. Say which name
            # is wrong, which ones are right, and where the caller's list
            # lives -- a bare `have [...]` sent the last reader looking for a
            # missing Drive id that was never missing.
            where = (" -- notebook 31 builds that list from VTUAV_VIS_PARTS"
                     if self.name == "vtuav_vis" else "")
            raise SystemExit(
                f"{self.name}: no part named "
                + ", ".join(repr(n) for n in unknown)
                + f".\n  Known parts: {', '.join(p.name for p in self.parts)}."
                + "\n  Nothing was downloaded. Either ask for a name off that "
                  "list, or drop\n  the unknown one from --parts"
                + where + ".")
        return [known[n] for n in names]


class PartsFailed(RuntimeError):
    """Some archives landed and some did not, with the reason for each.

    A dataset here is several multi-gigabyte archives fetched one after the
    other, and the failure that actually happens is one of them being refused
    -- Drive's quota, a mount that dropped -- long after the earlier ones
    extracted. Letting that exception escape reported it as a traceback and
    said nothing about the parts that *did* land, which is the whole of what
    the caller needs in order to decide what to run next. So the loop carries
    on to the remaining parts, and this carries both lists out.
    """

    def __init__(self, dataset: str, dest: Path, done: SequenceABC[str],
                 failed: SequenceABC[tuple[str, str]]) -> None:
        self.dataset = dataset
        self.dest = Path(dest)
        self.done = list(done)
        self.failed = [(name, reason) for name, reason in failed]
        super().__init__(self.summary())

    def summary(self) -> str:
        names = ", ".join(name for name, _ in self.failed)
        total = len(self.failed) + len(self.done)
        lines = [f"!! {self.dataset}: {len(self.failed)} of {total} part(s) "
                 f"did not land: {names}"]
        for name, reason in self.failed:
            lines.append(f"   {name}: {reason.strip().splitlines()[0]}")
        lines.append("   (the full reason for each was printed above, where "
                     "it happened)")
        if self.done:
            lines.append(
                f"   Kept: {', '.join(self.done)} -- extracted under "
                f"{self.dest} and not touched by this failure. Nothing here "
                f"skips an archive that is already unpacked, so ask for the "
                f"missing part(s) alone rather than re-running all of them.")
        else:
            lines.append("   Nothing landed: not one part of this set "
                         "extracted.")
        where = (" (notebook 31 builds it from VTUAV_VIS_PARTS)"
                 if self.dataset == "vtuav_vis" else "")
        lines.append(
            "   To continue: re-run with `--parts "
            + " ".join(name for name, _ in self.failed)
            + "` alone once the\n   reason above is gone, or drop those "
            + f"name(s) from the --parts list{where}\n   and train on what "
              "landed.")
        return "\n".join(lines)


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
    # Five plain-text manifests naming the 1 160 frames (29 % of the set!) that
    # have one modality deliberately corrupted to simulate a sensor failure.
    # A few kilobytes, always fetched, and `SPECS["kust4k"].exclude` reads them
    # so the reader drops those frames. Without them the corruption is
    # invisible: the images decode fine, they are just not the scene.
    extras=(
        ("broken_in_test_day_151.txt", "55979195"),
        ("broken_in_test_night_91.txt", "55979198"),
        ("broken_in_train_day_528.txt", "55979201"),
        ("broken_in_train_night_316.txt", "55979204"),
        ("broken_in_val_day_74.txt", "55979207"),
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
# Counted 2026-08 off `train.zip`'s ZIP64 central directory rather than the
# paper: 17 990 pairs in the train split, and **four** folders of 17 990 --
# `trainimg` / `trainimgr` for the two modalities and `trainlabel` /
# `trainlabelr` for their two separate annotation sets. 316 412 thermal boxes
# and 286 794 RGB ones; the paper's 953 087 is all three splits together.
#
# Reading all 35 980 XMLs says the two label sets are separate but not
# independent: 53.2 % of thermal boxes are byte-identical to an RGB one, the
# matched ones sit a median 0.00 px apart, and 10.6 % have no RGB counterpart
# at all -- vehicles only the thermal half can see. That last figure is why
# the pool harvests the thermal side. Full census: `docs/datasets.md`.
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

# Verified 2026-08 against CaltechDATA record `cks6g-ps927` (InvenioRDM):
# anonymous, 302 to a presigned S3 object on Open Storage Network, then 206
# with `accept-ranges: bytes`. 35.7 MB/s measured. No form, no quota, no Baidu.
#
# Two live details this download needs and others do not:
#   * **HEAD answers 403.** The redirect target is presigned for GET only, so
#     a size probe has to be a ranged GET.
#   * **The presigned URL carries `X-Amz-Expires=60`.** It must never be
#     cached; a resumed download re-requests `/content` and gets a fresh one.
#     `http_download` already does exactly that, since it only ever holds the
#     original URL.
#
# **Two archives that are not interchangeable.** Both ship a `thermal16/`, so
# extracting them under one root doubles it and every glob returns twice what
# it should. `into` gives each its own.
#
#   labeled_rgbt_pairs.zip     4.29 GB  2 282 registered pairs at 960x600
#   labeled_thermal_singles.zip 4.14 GB  3 076 masks at 640x512, no usable RGB
#
# The paired archive is the one worth the disk: stereo-rectified with the
# thermal projected into the EO frame, 2.5 cm baseline against a ~40 m
# altitude, measured residual 1-4 px -- the best-registered RGB-T on this list.
# Its 2 282 pairs are small next to VTUAV's, but it is the only natural
# terrain, water and night domain here.
#
# As an instance source it is poor and that is measured, not assumed: running
# this repo's own `decompose` over all 3 076 masks yields 1 357 instances
# (876 vehicles, 481 person) from 430 masks, median area 145 px = 0.04 % of
# frame, and 117 of the 430 come from one flight.
CALTECH = Recipe(
    name="caltech",
    note="Caltech Aerial RGB-T: 2 282 stereo-rectified pairs, natural terrain "
         "and water (ECCV 2024, CC BY-NC-SA)",
    parts=(
        Part("pairs", into="pairs", size=4_287_170_398,
             url="https://data.caltech.edu/api/records/cks6g-ps927/files/"
                 "labeled_rgbt_pairs.zip/content",
             md5="22f42923694e724d4b7e354bace12389"),
        Part("singles", into="singles", size=4_141_028_864, default=False,
             url="https://data.caltech.edu/api/records/cks6g-ps927/files/"
                 "labeled_thermal_singles.zip/content",
             md5="709f772dad92b50e345f5ff4def78615"),
    ),
)

# Verified 2026-08 by reading all 761 shard footers plus their `modality`
# column: 35 613 rows / 178 GiB, of which **15 007 rows are thermal** at
# 640x512 (~1.4 MB each) and 20 606 are RGB at 4000x3000 (~11 MB each). The two
# are sorted into different shards -- 311 thermal-only, 434 RGB-only, 16 mixed
# -- so the 327 shards worth having are 22 GiB and the 434 that are not are
# 156 GiB. 88 % of this repository is a modality nothing here trains on, and
# the first shard carrying a thermal row is number 57, so a streaming filter
# pays ~25 GiB before it can write its first row. `export_hf_dataset` therefore plans on the
# `modality` column and downloads only the shards that matter -- see its
# docstring.
#
# `columns` drops `RGB_aligned`. It is read only by `list_pairs`, i.e. by
# stage-A distillation, and no notebook here distils from SegFly (07, 08 and 10
# use VTUAV, 09 uses DroneVehicle). Writing it costs ~10 GB of PNGs -- a 640x512
# JPEG re-encodes to ~0.69 MB -- that nothing opens. Put "RGB_aligned" back if
# you point DISTILL_SPEC at segfly.
SEGFLY = Recipe(
    name="segfly",
    note="15 007 thermal 640x512 frames with semantic maps, over three "
         "altitudes (ECCV 2026); 22 GiB of a 178 GiB repository",
    hub="markus-42/SegFly",
    modality="thermal",
    columns=("image", "label"),
)

# The RGB half of the same repository, as a bounded slice. Full size is the
# problem: 20 606 frames at 4000x3000, ~156 GiB of shards that would become
# ~570 GB of PNGs -- so this takes 3 000 rows from evenly spaced shards
# (every scene and altitude represented, ~23 GB downloaded and written) and
# keeps the publisher's own JPEG bytes instead of re-encoding them. Native
# pixels are the point: at 4000x3000 from 30-50 m, a vehicle is genuinely
# small in a 512 window, which is the regime the deployment lives in and the
# regime ground-level RGB pretraining never showed the trunk.
#
# It shares `SPECS["segfly"]` with the thermal export -- same palette, same
# label directory -- and lands its frames in `rgb/`, which is the directory
# that spec's `rgb` glob reads. Train on it with
#     segfly:<dest>:rgb:components:train
SEGFLY_RGB = Recipe(
    name="segfly_rgb",
    note="a 3 000-frame slice of SegFly's 4000x3000 RGB half, every scene "
         "and altitude, publisher's JPEGs (~23 GB of a 156 GiB half)",
    hub="markus-42/SegFly",
    modality="RGB",
    columns=("image", "label"),
    spec="segfly",
    rows=3000,
    passthrough=True,
    spread=True,
)

# Verified 2026-08: the dataset *is* the git repository (annotations in
# normal_json/, images under normal_json/{train,val,test}), 418 MB by GitHub's
# own size field, CC-BY-4.0. The archive endpoint streams a zip it builds on
# the fly, so there is no content-length, no md5 and no resume -- acceptable at
# this size, and the alternative mirrors are worse: Kaggle needs an API token,
# and the Roboflow re-export on the Hub keeps only the person class of four.
HITUAV = Recipe(
    name="hituav",
    note="2 898 thermal 640x512 frames, 24 899 boxes over person/car/bicycle/"
         "other-vehicle at 80-130 m (CC-BY-4.0); the git repo is the dataset",
    parts=(
        Part("hituav",
             url="https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-"
                 "Dataset/archive/refs/heads/main.zip"),
    ),
)

# Verified 2026-08: the authors distribute RGBT234 through Baidu only
# (pan.baidu.com, password b05o), which a Colab runtime cannot answer. The Hub
# mirror `xche32/rgbt234` is one anonymous tar.gz behind an LFS endpoint that
# answers 206 to a range request, so it resumes. Third-party mirror, so the
# notebook counts what arrived -- 234 sequences, visible/ and infrared/ halves,
# a visible.txt *and* an infrared.txt per sequence -- before anything trusts it.
RGBT234 = Recipe(
    name="rgbt234",
    note="234 aligned RGB-T videos / 233.8 K frame pairs, boxes annotated per "
         "modality (visible.txt + infrared.txt)",
    parts=(
        Part("rgbt234", size=7_665_568_241,
             url="https://huggingface.co/datasets/xche32/rgbt234/resolve/"
                 "main/rgbt234.tar.gz"),
    ),
)

# LasHeR is the same layout scaled up -- 1 224 sequences, 730 K+ aligned pairs
# -- and the same Baidu-only problem. The Hub mirror `xche32/lasher` is one
# tar.gz cut into five ~50 GB slices, **~224 GB in all**, which is more than a
# Colab disk. So: every part defaults to off, and fetching them does not stage
# archives at all -- the slices are read off the network in order and
# decompressed as one stream, extracting only the sequences asked for
# (`--sequences`), so the disk cost is the frames kept and nothing else. The
# network cost is still reading the stream up to the last wanted sequence;
# there is no index to seek by in a .tar.gz. A full harvest belongs on a
# machine with a real disk, and `docs/mask_pool_plan.md` says so out loud.
LASHER = Recipe(
    name="lasher",
    note="1 224 aligned RGB-T videos / 730 K+ pairs, boxes per modality "
         "(~224 GB; streamed, sequence-selective, never fetched by default)",
    stream=True,
    parts=tuple(
        Part(f"part_{suffix}", default=False,
             size=53_687_091_200 if suffix != "ae" else 9_350_475_511,
             url=f"https://huggingface.co/datasets/xche32/lasher/resolve/"
                 f"main/lasher.tar.gz.part.{suffix}")
        for suffix in ("aa", "ab", "ac", "ad", "ae")),
)

# Verified 2026-08: plain files on the Hub (images/ + labels/ per split, the
# ultralytics YOLO conversion), not parquet -- so it goes through
# snapshot_download with each part's `into` as the allow_pattern. Third-party
# mirror of the VisDrone2019-DET release; the notebook probes the class
# histogram against `boxes.VISDRONE_NAMES` before labelling anything.
VISDRONE = Recipe(
    name="visdrone",
    note="VisDrone2019-DET: 6 471 train / 548 val aerial RGB frames, dense "
         "small-object boxes over 10 classes (YOLO-converted mirror)",
    snapshot="banu4prasad/VisDrone-Dataset",
    parts=(
        Part("train", into="VisDrone2019-DET-train/**", size=1_500_000_000),
        Part("val", into="VisDrone2019-DET-val/**", size=80_000_000),
        Part("test-dev", into="VisDrone2019-DET-test-dev/**",
             size=300_000_000, default=False),
    ),
)

# Verified 2026-08 against the live Drive folder
# (`drive.google.com/drive/folders/1Mi3NXQ-YG1iiIWkPbe3GQoDK68dARMN6`), not the
# paper: the zip is 1 573 829 771 B and Drive answers 206 to a range request on
# the confirmed URL, so it resumes. Its central directory says
# `train/{annotation,thermal,visible}` and `val/{...}` -- 4 900 + 1 225 pairs,
# 12 250 jpgs, 6 125 Pascal-VOC xmls -- so the archive keeps its own layout and
# no part needs an `into`.
#
# The xml is drawn on the **thermal** frame (`<folder>Thermal</folder>`,
# `<depth>1</depth>`, and box (330,344)-(346,372) is COCO `[329,343,16,28]`
# off by the inclusive-max convention). The four COCO jsons live beside the zip
# as separate Drive files, which is why they are `extras` rather than members.
#
# Two numbers the notebook should see before it trusts this set as a prompt
# source, both measured off the jsons: the median target is sqrt(area) = 11 px
# and 93 % of them are under 16 px, and matching the thermal boxes to the
# visible ones (same class, nearest centre, 40 px gate, 94.5 % matched) puts
# the median centre disagreement at **11.7 px** -- about one target diameter,
# with only 15 % inside 5 px. Whether that is registration error or two
# annotators drawing the same person twice does not matter downstream: a
# thermal box is not a usable prompt on the visible frame here.
RGBTDRONEPERSON = Recipe(
    name="rgbtdroneperson",
    note="RGBTDronePerson (WHU-DroneDual): 6 125 aerial RGB-T pairs at "
         "640x512, 70 880 person/rider/crowd/uncertain boxes drawn on the "
         "thermal half (CC BY 4.0)",
    parts=(
        Part("rgbtdroneperson", drive="18zm2CaJS73a6eUHXUQ6_cjjbrurSkm9C",
             size=1_573_829_771),
    ),
    extras=(
        ("train_thermal.json", "drive:1P5FNdig1_IEq9xGfL3zAzMxQf-Jeg7_2"),
        ("val_thermal.json", "drive:1H_hHqoqaVPBa3LBTEgveZLxE-eovaOmE"),
        ("sub_train_thermal.json", "drive:1Cxp6Hau20jRQowEAU7ST5wwI_kRUQq65"),
        ("sub_train_visible.json", "drive:1aAlgSY8tWA4RDIi88l83mFtrdvj0KAGa"),
    ),
)

# The detection re-annotation of VTUAV, from the same authors' Drive folder
# (`1kBomGd7bu-9MiUDGmViHqmV9baN739iG`). Verified the same way: 6 314 183 652 B,
# range-resumable, central directory `train/{anno,ir,rgb}` (11 392 each) plus
# `test/{anno,ir,rgb}` (5 378), so again no `into`.
#
# Worth knowing before using it as a *thermal* prompt source: the xml says
# `<folder>rgb</folder>` at 1920x1080x3, and its box for `train/00001.jpg` is
# (769,217)-(793,258) while `train_ir.json` gives `[768,216,24,41]` for the
# same frame -- the identical rectangle. **One box set is shared by both
# modalities**, so whatever residual misregistration VTUAV has (see
# `docs/encoder_mimari.md` -- the two halves are not pixel-exact) is inherited
# silently rather than annotated. Against that, the targets are big enough for
# a box prompt to mean something: median sqrt(area) 69 px on train, 48 px on
# val, versus 11 px in RGBTDronePerson.
VTUAVDET = Recipe(
    name="vtuavdet",
    note="VTUAV-det: 11 392 train / 5 378 test aligned RGB-T frames at "
         "1920x1080, 124 869 person boxes shared by both modalities",
    parts=(
        Part("vtuavdet", drive="1TLmMOQWE5otkjJQmCWFVdWgWv05V1BO-",
             size=6_314_183_652),
    ),
    extras=(
        ("train_ir.json", "drive:13dguRNo6Cb84jYMWaYOnyT1d_xjqAts0"),
        ("val_ir.json", "drive:1bpECNDRZ6enewHyASGJOBdqhSt66k2ES"),
    ),
)

# BIRDSAI, hosted by LILA as "Conservation Drones". Anonymous Azure blobs that
# answer 206, no form and no account, and -- unusually for this list --
# **CDLA-Permissive-1.0**, read off the page's schema.org block rather than
# assumed. Central directories read over a range request: TrainReal is 32
# sequences / 40 661 frames / 32 csv, TestReal is 16 / 21 336 / 16, which is the
# paper's 48 real videos and ~62 K frames.
#
# Layout is `TrainReal/{images,annotations}/<sequence>/`, one MOT csv per
# sequence: `frame, id, x, y, w, h, class, species, occlusion, noise`. The `id`
# column is what makes it interesting here -- it is the only thermal aerial set
# on this list that ships **track ids**, so a masklet is a group-by rather than
# a tracker run. The targets are tiny (9-14 px boxes in the sample sequence),
# which is the same caveat RGBTDronePerson carries.
#
# The 42 GB synthetic half is AirSim renders; off by default.
BIRDSAI = Recipe(
    name="birdsai",
    note="BIRDSAI: 48 night-time aerial TIR sequences, ~62 K frames, 166 K "
         "boxes with track ids over humans and animals (CDLA-Permissive-1.0)",
    parts=(
        Part("train_real", size=2_271_707_323,
             url="https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                 "conservationdrones/v01/conservation_drones_train_real.zip"),
        Part("test_real", size=1_761_788_520,
             url="https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                 "conservationdrones/v01/conservation_drones_test_real.zip"),
        Part("train_simulation", default=False, size=42_222_672_278,
             url="https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                 "conservationdrones/v01/"
                 "conservation_drones_train_simulation.zip"),
    ),
)

# AIResQ (Sci. Data 2026). Read the two Zenodo records rather than the paper's
# headline: the 9 788-image, 2048x1536 half is `access_right: restricted` --
# the API returns an empty `files` list and you have to ask the authors -- and
# the only open record is the benchmark, 1 609 176 235 B with a publisher md5.
# Its central directory is `Benchmark/{images,labels}`, 1 988 DJI thermal JPGs
# and 1 988 YOLO txts, one class (person). So: a small, clean, CC BY 4.0
# thermal SAR set, not the 9 788 images the abstract advertises.
AIRESQ = Recipe(
    name="airesq",
    note="AIResQ benchmark: 1 988 aerial thermal frames with YOLO person "
         "boxes (CC BY 4.0; the 2048x1536 half is access-restricted)",
    parts=(
        Part("airesq", size=1_609_176_235,
             md5="6a7e8920ee62c6c8234e0f08e631410a",
             url="https://zenodo.org/records/17405074/files/"
                 "AIResQ_Benchmark.zip?download=1"),
    ),
)

# VTUAV's *tracking* archives, which are a different download from the mask
# split above: 500 sequences of 1920x1080 RGB-T with one target per sequence
# and a box every tenth frame. Ids and sizes read off the project page's Drive
# links in 2026-08, then each one range-probed for its true length.
#
# **Only the RGB-T version is listed, because the RGB version is the same
# bytes.** Sampling `train_ST_001` from both folders, every RGB member matched
# on CRC32 *and* compressed size: the RGB archive is the RGB-T archive with
# `ir/` and `ir.txt` removed. Downloading both is 9 GiB per part of pure
# duplication, and `boxes.vtuav_frames(modality="rgb")` reads the RGB half of
# the RGB-T extraction directly.
#
# Every part defaults to off: the train half alone is **214.5 GiB** and no
# default should reach for that. What one part buys, measured on ST_001:
# 15.4 GiB, 20 sequences, 37 419 frame pairs, 3 750 annotated rows.
#
# The parts are ordered by sequence name, so each carries only two to four
# object kinds -- ST_001 is animal/bike/bus, ST_005 is car/elebike, ST_008 is
# pedestrian (24 of its 28 sequences), ST_011 is car/pedestrian/truck. A pool
# built from consecutive parts is a pool of two categories; spread them.
VTUAV_TRACK = Recipe(
    name="vtuav_track",
    note="VTUAV tracking split: 1920x1080 RGB-T sequences, one box per target "
         "every 10th frame, per-modality (rgb.txt + ir.txt). 214.5 GiB train; "
         "nothing is fetched by default",
    parts=(
        Part("train_ST_001", default=False, drive="1GQgLZ8kJo6ljR3k0tLUyvokp33ecwmM4",
             size=16553040666),
        Part("train_ST_002", default=False, drive="1hOsPq6umKb0FtV4royO2nuiwBmOZR5zf",
             size=18059604978),
        Part("train_ST_003", default=False, drive="1Mlj73qj4JZgcUVhlir4oqDSkeFFUZk__",
             size=17986937566),
        Part("train_ST_004", default=False, drive="1UdbUJlJaTB7loptBlBY2Jb47YVDIxrbj",
             size=18850569348),
        Part("train_ST_005", default=False, drive="1RqSkP_qZ_3hjLhxd5Hm7uL6nLj6PufdL",
             size=7856011518),
        Part("train_ST_006", default=False, drive="1cp5mQjxTy3GChEYW_zmZ_HwsvApGygKK",
             size=15388627743),
        Part("train_ST_007", default=False, drive="1i6U7Ld1oHKsm9XyquE2WakB3poHdLkg9",
             size=12371156732),
        Part("train_ST_008", default=False, drive="1cMM7TJ2yIUttKKyBLkZTDohB1t1acrrA",
             size=14032522966),
        Part("train_ST_009", default=False, drive="1MKcrP-LhGSXiNghWGk-h9i7FX3u0vGLd",
             size=13380718697),
        Part("train_ST_010", default=False, drive="1-h66icS6IYOo8F0thDYkZJ9LFUzvHAZ-",
             size=18926310599),
        Part("train_ST_011", default=False, drive="1YEOCsni7RxnOI3c7sOG32qFwgB7xSY-O",
             size=12422921973),
        Part("train_LT_001", default=False, drive="1JAQfW2ZnRyAJ2llfiayGHjv6x1A2iRbR",
             size=17705121999),
        Part("train_LT_002", default=False, drive="1HD0StT_aGROmrluEVr86hH9Akee5qA3K",
             size=17287125186),
        Part("train_LT_003", default=False, drive="1X8_OUstaGyoJclihwBAiTXJLPFZ1LbJN",
             size=14607972970),
        Part("train_LT_004", default=False, drive="1RilAMqJLKIAlmW44o5K_v3RD2mlywmCk",
             size=14937685115),
    ),
)

RECIPES: dict[str, Recipe] = {
    r.name: r for r in (KUST4K, VTUAV_VIS, DRONEVEHICLE, CALTECH, SEGFLY,
                        SEGFLY_RGB, HITUAV, RGBT234, LASHER, VISDRONE,
                        RGBTDRONEPERSON, VTUAVDET, BIRDSAI, AIRESQ,
                        VTUAV_TRACK)}


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
    """A copy of `<name>.zip` (or `.tar.gz`) already sitting somewhere we can read.

    The escape hatch for Drive's download quota, and the reason it works: a
    file in *your own* Drive is not a widely-shared file, so reading it through
    the Colab mount is an ordinary authenticated read with no shared-file quota
    attached. Three clicks in the Drive web UI ("Make a copy") turn the one
    into the other.

    Those three clicks do not produce `<name>.zip`. Drive names the copy in the
    account's own language -- `Copy of train_001.zip` in English,
    `train_001.zip adlı dosyanın kopyası` in Turkish -- and drops the extension
    off the end in the process, so an exact-name lookup misses the very file
    this function exists to find and the run goes back to the network it was
    trying to avoid. Anything holding `<name><suffix>` inside its own name is
    therefore accepted too, largest first: a locale that phrases the copy some
    third way still lands on the same rule, and the size ordering prefers the
    real archive over a truncated earlier attempt sitting beside it.
    """
    for folder in search:
        directory = Path(folder).expanduser()
        for suffix in (".zip", ".tar.gz", ".tgz"):
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file() and candidate.stat().st_size > 1 << 20:
                return candidate
        if not directory.is_dir():
            continue
        copies = [
            path for path in directory.iterdir()
            if path.is_file() and path.stat().st_size > 1 << 20
            and any(f"{name}{suffix}" in path.name
                    for suffix in (".zip", ".tar.gz", ".tgz"))
        ]
        if copies:
            found = max(copies, key=lambda path: path.stat().st_size)
            print(f"   using {found.name} -- a Drive copy of {name}")
            return found
    return None


def archive_to(dest: Path, name: str, folder: str | Path) -> Path | None:
    """Zip an exported tree into `<folder>/<name>.zip`, where `staged` finds it.

    Stored, not deflated. The tree is PNGs, which are already deflate streams,
    so compressing them again buys a percent or two for several minutes of CPU
    -- and the point of this file is to be read back quickly.

    Written to a `.part` first and renamed, because the failure this guards
    against is a runtime dying mid-zip and leaving a truncated archive that
    `staged` then happily prefers over the network on every later run.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        print(f"   !! {folder} does not exist -- not staging a copy")
        return None
    target = folder / f"{name}.zip"
    partial = folder / f"{name}.zip.part"
    files = [p for p in sorted(dest.rglob("*")) if p.is_file()]
    print(f"   staging {len(files)} files -> {target}")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in files:
            z.write(path, path.relative_to(dest))
    partial.replace(target)
    print(f"   staged ({human(target.stat().st_size)}); later runs reuse it")
    return target


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


def tracked_members(archive: zipfile.ZipFile, modality: str = "rgb",
                    stride: int = 10) -> list[str]:
    """Only the frames a VTUAV *tracking* archive actually annotates.

    Verified against `train_ST_001.zip`'s central directory: a sequence is
    `<name>/rgb/000000.jpg`, `<name>/ir/000000.jpg`, `<name>/rgb.txt`,
    `<name>/ir.txt`, the frame ids run 0..n-1 with no gaps, and the box files
    hold `ceil(n / 10)` lines on all 20 sequences -- so **line k is frame
    10k** and nine frames in ten carry no label at all. Extracting everything
    spends 15.4 GB of disk to make 3 750 usable pairs.

    `modality` is `rgb`, `ir`, or **`both`**. One modality halves it again,
    which is what makes the RGB and thermal pools runnable side by side on two
    ordinary runtimes: each unzips only the half it prompts on. `both` keeps
    that tenth in *both* halves, for staging one tree a harvest of either half
    can read -- `--frames tracked` on the command line. (Putting a finished
    pool's frames back is a different question with a better answer:
    `pool_reader.extract_frames` takes exactly the members the records name,
    and needs no stride at all.) Both `.txt` files are always kept: they are a
    few kilobytes and the other modality's boxes are what any later agreement
    check reads.

    **The stride is read out of each sequence, not taken on faith.** That 10
    was measured on one short-term part; the long-term parts are a separate
    download and this function is the first thing that touches them. Getting
    it wrong is silent -- extract every 9th frame and the harvest labels frame
    9 with frame 10's box -- so `boxes.annotated_stride` derives it from the
    sequence's own frame and row counts, `stride` is preferred wherever those
    counts allow it, and a sequence whose counts allow no single answer is
    **dropped** with its numbers printed. A dropped sequence costs a sequence;
    a guessed stride costs the pool.
    """
    from src.training.boxes import annotated_stride

    if modality not in ("rgb", "ir", "both"):
        raise ValueError(f"modality must be rgb, ir or both, got {modality!r}")
    kept = ("rgb", "ir") if modality == "both" else (modality,)

    counts: dict[str, int] = {}
    present: dict[str, int] = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 2 and parts[1] == f"{kept[0]}.txt":
            with archive.open(name) as handle:
                counts[parts[0]] = sum(1 for line in handle if line.strip())
        elif len(parts) == 3 and parts[1] == kept[0]:
            if Path(parts[-1]).stem.isdigit():
                present[parts[0]] = present.get(parts[0], 0) + 1

    wanted: set[tuple[str, int]] = set()
    strides: dict[int, int] = {}
    dropped = 0
    for sequence, lines in sorted(counts.items()):
        try:
            step = annotated_stride(present.get(sequence, 0), lines, stride)
        except ValueError as mismatch:
            print(f"   {sequence}: {mismatch} -- dropped, no frame of it is "
                  f"extracted; both .txt files are kept so the numbers can be "
                  f"read back on disk")
            dropped += 1
            continue
        strides[step] = strides.get(step, 0) + 1
        wanted.update((sequence, index * step) for index in range(lines))

    if set(strides) - {stride} or dropped:
        print(f"   stride -> sequences: {dict(sorted(strides.items()))}, "
              f"{dropped} dropped")

    keep = []
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 2 and parts[1].endswith(".txt"):
            keep.append(name)
        elif len(parts) == 3 and parts[1] in kept:
            stem = Path(parts[-1]).stem
            if stem.isdigit() and (parts[0], int(stem)) in wanted:
                keep.append(name)
    return keep


def extract(archive_path: Path, dest: Path, into: str = "",
            frames: str = "all", quiet: bool = False, workers: int = 1) -> int:
    """Unpack into `dest/into`, returning how many files were written.

    Zip or tar.gz, told apart by the file itself rather than its name -- a
    staged copy may have been renamed. Tars are read in **stream** mode
    (`r|gz`): `getmembers()` on a compressed tar decompresses the whole file
    once just to list it and a second time to extract, which on RGBT234's
    7.7 GB doubles a ten-minute step for nothing.

    `workers > 1` reads a **zip** on that many threads, each with its own
    handle on the file. It is off by default and it is not for speed on a
    local disk -- one thread already saturates that. It is for a `frames`
    filter over a Drive mount: `tracked_ir` keeps a twentieth of a 15.7 GiB
    part, so the read stops being a stream and becomes a few thousand random
    seeks, and a FUSE seek is latency, not bandwidth. Concurrency is the only
    thing that hides latency. Inflating releases the GIL, so threads are
    enough. A tar is a stream with no index and cannot be read this way.
    """
    target = dest / into if into else dest
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            if frames == "masked":
                members = masked_members(archive)
            elif frames == "tracked":
                members = tracked_members(archive, "both")
            elif frames.startswith("tracked_"):
                members = tracked_members(archive, frames.split("_", 1)[1])
            else:
                members = archive.namelist()
            if not quiet:
                print(f"   extracting {len(members)} entries -> {target}"
                      + (f" on {workers} threads" if workers > 1 else ""))
            if workers > 1:
                return _extract_parallel(archive_path, members, target,
                                         workers, quiet)
            try:
                archive.extractall(target, members=members)
            except Exception as failure:     # noqa: BLE001 - see below
                # `extractall` stops at the first member it cannot read and
                # leaves every member after it on the floor. One bad CRC in a
                # Drive copy of DroneVehicle's train.zip cost 9 859 of 13 098
                # thermal frames that way, and the run that lost them reported
                # only `no_image`. So a failure drops to a member-by-member
                # pass that keeps what the archive can still give and names
                # what it cannot.
                if not quiet:
                    print(f"   !! {failure} -- extracting member by member")
                return _extract_each(archive, members, target, quiet)
        return len(members)

    import tarfile

    if not tarfile.is_tarfile(archive_path):
        raise RuntimeError(
            f"{archive_path}: neither a zip nor a tar. A download this size "
            f"that is not an archive is usually an HTML error page -- delete "
            f"it and run the cell again.")
    count = 0
    mode = "r|gz" if archive_path.name.endswith(("gz", "tgz")) else "r|"
    with tarfile.open(archive_path, mode) as archive:
        for member in archive:
            _tar_extract(archive, member, target)
            count += 1
            if not quiet and count % 20_000 == 0:
                print(f"   {count} entries so far...", flush=True)
    if not quiet:
        print(f"   extracted {count} entries -> {target}")
    return count


def _extract_parallel(archive_path: Path, members: SequenceABC[str],
                      target: Path, workers: int, quiet: bool = False) -> int:
    """The same members, on `workers` threads, each with its own zip handle.

    A `ZipFile` holds one file position, so sharing one across threads
    interleaves seeks and returns garbage. Opening one per thread is the whole
    trick; the central directory is parsed per handle, which is milliseconds
    against the seeks this exists to overlap.

    A member that cannot be read is counted rather than raised, the way the
    serial fallback does it: one bad CRC in a Drive copy must not cost the
    other seventy thousand frames.

    Every directory is made **before** the pool starts. `ZipFile.extract`
    creates a member's parent with a `isdir` test followed by `makedirs`, and
    two threads landing in the same folder both pass the test and one of them
    raises -- which showed up as a single "unreadable" frame per folder, on
    an archive with nothing wrong with it.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    local = threading.local()
    opened: list[zipfile.ZipFile] = []
    counts = {"taken": 0, "already": 0}
    bad: list[str] = []
    lock = threading.Lock()

    def handle() -> zipfile.ZipFile:
        if not hasattr(local, "zip"):
            local.zip = zipfile.ZipFile(archive_path)
            with lock:
                opened.append(local.zip)
        return local.zip

    def one(member: str) -> None:
        landing = target / member
        if landing.is_file() and landing.stat().st_size:
            with lock:
                counts["already"] += 1
            return
        try:
            handle().extract(member, target)
        except Exception:                    # noqa: BLE001 - one bad member
            with lock:
                bad.append(member)
            return
        with lock:
            counts["taken"] += 1

    for folder in {(target / member).parent for member in members}:
        folder.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, members))
    finally:
        for opened_zip in opened:
            opened_zip.close()
    if not quiet:
        print(f"   {counts['taken']} extracted, {counts['already']} already "
              f"there, {len(bad)} unreadable")
        if bad:
            print("   unreadable:", bad[:5], "..." if len(bad) > 5 else "")
    if not counts["taken"] and not counts["already"]:
        raise RuntimeError(
            f"{target}: not one member of the archive could be read. The "
            f"staged copy is corrupt -- delete it and let the download run.")
    return counts["taken"] + counts["already"]


def _extract_each(archive, members: SequenceABC[str], target: Path,
                  quiet: bool = False) -> int:
    """Every member the archive can still give, one at a time.

    Members already on disk are skipped, so this is also the resume path: a
    second run over a partly extracted tree writes only what is missing.
    """
    taken, skipped, bad = 0, 0, []
    for member in members:
        landing = target / member
        if landing.exists() and (landing.is_dir() or landing.stat().st_size):
            skipped += 1
            continue
        try:
            archive.extract(member, target)
            taken += 1
        except Exception:                    # noqa: BLE001 - one bad member
            bad.append(member)
    if not quiet:
        print(f"   {taken} extracted, {skipped} already there, "
              f"{len(bad)} unreadable")
        if bad:
            print("   unreadable:", bad[:5], "..." if len(bad) > 5 else "")
    if not taken and not skipped:
        raise RuntimeError(
            f"{target}: not one member of the archive could be read. The "
            f"staged copy is corrupt -- delete it and let the download run.")
    return taken + skipped


def _tar_extract(archive, member, target: Path) -> None:
    """One member, through the stdlib's traversal filter where it exists."""
    try:
        archive.extract(member, target, filter="data")
    except TypeError:                     # Python without the filter kwarg
        archive.extract(member, target)


def stream_extract(urls: SequenceABC[str], dest: Path,
                   sequences: SequenceABC[str] | None = None,
                   quiet: bool = False) -> int:
    """Extract one tar.gz served as consecutive URL slices, saving no archive.

    Built for LasHeR's five ~50 GB slices: concatenating them first needs
    224 GB of disk *before* extraction starts, which no Colab has. This reads
    the slices in order as a single gzip stream and hands it to `tarfile` in
    stream mode, so the only disk spent is on the members kept.

    `sequences` filters by path component -- only members whose path contains
    one of the names are written. The stream still has to be *read* up to the
    last member wanted (a tar.gz has no index to seek by), so this trades
    network time for disk, never the reverse. Interruption loses the stream;
    there is no resume, and the caller is told so rather than surprised.
    """
    import io
    import tarfile

    import requests

    wanted = set(sequences) if sequences else None

    class _Slices(io.RawIOBase):
        """The slices, presented as one read-only byte stream."""

        def __init__(self) -> None:
            self.remaining = list(urls)
            self.session = requests.Session()
            self.current = None

        def readable(self) -> bool:
            return True

        def _next(self):
            if not self.remaining:
                return None
            url = self.remaining.pop(0)
            if not quiet:
                print(f"   streaming {url.rsplit('/', 1)[-1]}", flush=True)
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            return response.raw

        def readinto(self, buffer) -> int:
            while True:
                if self.current is None:
                    self.current = self._next()
                    if self.current is None:
                        return 0
                got = self.current.read(len(buffer))
                if got:
                    buffer[:len(got)] = got
                    return len(got)
                self.current = None       # this slice is spent; move on

    dest.mkdir(parents=True, exist_ok=True)
    kept = 0
    with tarfile.open(fileobj=io.BufferedReader(_Slices(), 1 << 22),
                      mode="r|gz") as archive:
        for member in archive:
            if wanted is not None and not (wanted & set(Path(member.name).parts)):
                continue
            _tar_extract(archive, member, dest)
            kept += 1
            if not quiet and kept % 5_000 == 0:
                print(f"   {kept} members kept so far...", flush=True)
    print(f"   kept {kept} members -> {dest}")
    return kept


# Where a hand-staged archive is looked for before the network is touched.
# `Make a copy` in Drive puts a shared file into MyDrive, and reading your own
# file has no shared-file quota on it -- see `drive_download`.
STAGING = ("/content/drive/MyDrive/datasets",
           "/content/drive/MyDrive",
           "/content/staging")


def fetch_extra(source: str, target: Path) -> Path:
    """One sidecar file -- a manifest or a COCO json -- next to the data.

    Three hosts, because the sets that need a sidecar do not share one:
    Kust4K's palette script is a bare figshare file id (the form this argument
    had first), RGBTDronePerson and VTUAV-det keep their COCO jsons as separate
    Drive files beside the image zip, and anything else is a plain URL.
    """
    if source.startswith("drive:"):
        return drive_download(source[len("drive:"):], target, quiet=True)
    if source.startswith(("http://", "https://")):
        return http_download(source, target, quiet=True)
    return http_download(f"https://ndownloader.figshare.com/files/{source}",
                         target, quiet=True)


def fetch_part(part: Part, dest: Path, work: Path, frames: str,
               keep: bool, quiet: bool,
               staging: SequenceABC[str] = STAGING) -> None:
    # The work file keeps the server's own basename where there is one: naming
    # a tar.gz `<part>.zip` would send `extract` down the wrong branch, and the
    # sniff-by-magic there is a backstop, not an invitation.
    basename = (Path(part.url.split("?")[0]).name if part.url
                else f"{part.name}.zip")
    archive = work / basename
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
          limit: int | None = None, stage: str | Path | None = None,
          sequences: SequenceABC[str] | None = None,
          staging: SequenceABC[str] = STAGING) -> Path:
    """One dataset into `dest`, in the layout its reader expects."""
    recipe = RECIPES[name]
    dest = Path(dest).expanduser()
    print(f"\n=== {recipe.name}: {recipe.note}")

    if recipe.snapshot:
        from huggingface_hub import snapshot_download

        chosen = recipe.chosen(parts)
        patterns = [p.into for p in chosen]
        print(f"snapshot of {recipe.snapshot}: {', '.join(p.name for p in chosen)}"
              f" (~{human(sum(p.size for p in chosen))})")
        snapshot_download(repo_id=recipe.snapshot, repo_type="dataset",
                          local_dir=dest, allow_patterns=patterns)
        return dest

    if recipe.stream:
        chosen = recipe.chosen(parts)
        if not chosen:
            print(
                "nothing fetched: every slice of this set defaults to OFF "
                f"because the whole of it is ~{human(sum(p.size for p in recipe.parts))}.\n"
                "Ask for it explicitly -- all five slices, ideally with a "
                "sequence filter:\n"
                "    python tools/fetch_datasets.py lasher --dest <dest> "
                "--parts part_aa part_ab part_ac part_ad part_ae "
                "--sequences <name> [...]\n"
                "The stream is read start to finish (a tar.gz has no index), "
                "so the network cost is the slices asked for even when few "
                "sequences are kept.")
            return dest
        stream_extract([p.url for p in chosen], dest, sequences=sequences,
                       quiet=quiet)
        return dest

    if recipe.hub:
        from tools.export_hf_dataset import COLUMNS, export, verify

        # Same escape hatch the archive datasets get, and it is worth more
        # here: the export is the expensive step, not the download, and its
        # output is a few gigabytes of small PNGs. One zip on Drive turns a
        # 30-minute cell into a 2-minute one on every later runtime.
        already = staged(recipe.name, staging)
        if already is not None:
            print(f"   using the export already at {already}")
            extract(already, dest, quiet=quiet)
            return dest

        result = export(recipe.hub, dest, recipe.modality, "train",
                        recipe.rows if limit is None else limit,
                        streaming=False, quiet=quiet,
                        columns=recipe.columns or tuple(COLUMNS),
                        passthrough=recipe.passthrough, spread=recipe.spread)
        print(f"{result['written']} rows written, {result['skipped']} skipped")
        print(verify(result["values"], recipe.spec or recipe.name))
        if stage:
            archive_to(dest, recipe.name, stage)
        return dest

    chosen = recipe.chosen(parts)
    # Report against what is actually going to be fetched. Announcing "40 GB to
    # download" and then reading three staged copies off Drive is a confusing
    # way to be right.
    ready = {p.name: staged(p.name, staging) for p in chosen}
    for name, where in ready.items():
        if where is not None:
            print(f"   {name}: staged at {where}, no download needed")
    if not chosen:
        print(f"nothing fetched: every part of this set defaults to OFF "
              f"because the whole of it is "
              f"~{human(sum(p.size for p in recipe.parts))}. Ask for the ones "
              f"you want:\n    python tools/fetch_datasets.py {recipe.name} "
              f"--dest <dest> --parts {recipe.parts[0].name} [...]")
        return dest
    wanted = [p for p in chosen if ready[p.name] is None]
    if wanted:
        print(f"{len(wanted)} archive(s), about {human(sum(p.size for p in wanted))} "
              f"to download")
    else:
        print("every archive is staged already; nothing to download")

    for name, source in recipe.extras:
        target = dest / name
        if not target.is_file():
            fetch_extra(source, target)
    if recipe.extras:
        print(f"{len(recipe.extras)} manifest(s) alongside the data")

    work = dest / "_archives"
    work.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for part in chosen:
            try:
                fetch_part(part, dest, work, frames, keep, quiet, staging)
            except Exception as failure:    # noqa: BLE001 - see PartsFailed
                # One refused archive must not cost the ones after it, and it
                # must not cost the ones before it either: those are already
                # extracted and usable. Say why here, where the context is,
                # and collect the name for the summary at the end.
                reason = str(failure).strip() or type(failure).__name__
                failed.append((part.name, reason))
                print(f"   !! {part.name} did not land:")
                for line in reason.splitlines():
                    print(f"      {line}")
                if part is not chosen[-1]:
                    print("   carrying on with the next part.", flush=True)
            else:
                done.append(part.name)
    finally:
        if not keep and work.exists() and not any(work.iterdir()):
            work.rmdir()
    if failed:
        raise PartsFailed(recipe.name, dest, done, failed)
    return dest


def report(name: str, dest: Path, modality: str = "thermal") -> str:
    """What the reader actually finds there -- the only check that counts."""
    from src.training.aerial import SPECS, describe_layout, list_frames, list_pairs

    recipe = RECIPES.get(name)
    spec_name = (recipe.spec or name) if recipe else name
    if spec_name not in SPECS:
        # A box dataset -- no mask spec to glob by. Say what landed, in the
        # shape its own reader (src/training/boxes.py) will look for it.
        return f"\n--- {dest}\n{describe_layout(dest)}"
    spec = SPECS[spec_name]
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
    p.add_argument("--frames",
                   choices=("all", "masked", "tracked", "tracked_rgb",
                            "tracked_ir"),
                   default="all",
                   help="`masked` keeps only annotated frames and their twins "
                        "-- a twentieth of the disk, and too few pairs for "
                        "stage-A distillation. `tracked` / `tracked_rgb` / "
                        "`tracked_ir` are the same idea for VTUAV's tracking "
                        "archives, in both halves or one -- only the frames "
                        "the box file names, at the stride each sequence's own "
                        "counts imply.")
    p.add_argument("--limit", type=int, default=None,
                   help="Hub datasets only: stop after N rows.")
    p.add_argument("--keep", action="store_true",
                   help="Keep the archives after extracting them.")
    p.add_argument("--stage", nargs="?", const=STAGING[0], default=None,
                   help="Hub datasets only: after exporting, zip the result "
                        f"into this folder (default {STAGING[0]}) so later "
                        "runtimes read it from there instead of re-exporting.")
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Streamed datasets (lasher): keep only members whose "
                        "path contains one of these sequence names.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    names = sorted(RECIPES) if args.dataset == "all" else [args.dataset]
    folders = {"kust4k": "Kust4K", "vtuav_vis": "VTUAV_VIS",
               "dronevehicle": "DroneVehicle", "caltech": "Caltech",
               "segfly": "SegFly", "segfly_rgb": "SegFly_RGB",
               "hituav": "HIT_UAV", "rgbt234": "RGBT234",
               "lasher": "LasHeR", "visdrone": "VisDrone",
               "rgbtdroneperson": "RGBTDronePerson", "vtuavdet": "VTUAV_det",
               "birdsai": "BIRDSAI", "airesq": "AIResQ",
               "vtuav_track": "VTUAV_track"}
    incomplete: list[PartsFailed] = []
    for name in names:
        dest = Path(args.dest) if args.dest else Path(args.root) / folders[name]
        try:
            fetch(name, dest, tuple(args.parts) if args.parts else None,
                  frames=args.frames, keep=args.keep, quiet=args.quiet,
                  limit=args.limit, stage=args.stage, sequences=args.sequences)
        except PartsFailed as partial:
            # Not a traceback: a named, printed refusal on stdout, followed by
            # the report of what *is* on disk. A caller that runs this through
            # `subprocess.run(..., check=True)` sees the same non-zero exit
            # either way, and this way it sees which part and why.
            incomplete.append(partial)
            print("\n" + partial.summary(), flush=True)
        print(report(name, dest,
                     "rgb" if name in ("vtuav_vis", "segfly_rgb", "visdrone")
                     else "thermal"))

    free = shutil.disk_usage(args.root if not args.dest else args.dest).free
    print(f"\n{human(free)} of disk left.")
    if incomplete:
        missing = ", ".join(f"{p.dataset}/{name}"
                            for p in incomplete for name, _ in p.failed)
        print(f"\nexiting 1: {missing} did not land. Everything else above is "
              f"on disk.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
