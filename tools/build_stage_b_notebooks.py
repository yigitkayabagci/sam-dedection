#!/usr/bin/env python3
"""Generate the stage-B pool notebooks, and nothing else.

Two arms of one experiment, generated from one file the way 07-11 and 16/17
are, and for the same reason: they differ in **one setting** and every other
byte is identical, which is the only thing that lets their two numbers be set
beside each other.

    19_thermal_stage_b_pool.ipynb       MODALITIES = ["thermal"]
    20_thermal_stage_b_pool_rgb.ipynb   MODALITIES = ["thermal", "rgb"]
    22_thermal_deep.ipynb               long thermal run
    23_thermal_deep_lora.ipynb          22 with LoRA
    27_thermal_deep_rgb_aerovis.ipynb   22 with RGB pools and AeroVIS
    28_rgb_deep_aerovis.ipynb           RGB-only long run, AeroVIS required

Both are the shape 15-18 were asked for -- **no markdown, no comments, as few
cells as possible** -- because this is a job to run, not a decision to read.
Everything the run needs to say, it prints.

## What these do that 07-11 do not

**No stage A.** `07` distils an unlabelled RGB teacher into the encoder before
it trains, and `12` then measured the anchor meant to protect that stage
*eating* it (0.8670 at weight 0 against 0.8536 at 0.5, with a plain fine-tune
at 0.8611). These notebooks skip the whole question: one stage B from stock
EdgeTAM, so what the pools bought is the only thing in the number.

**The data is the pools.** 07's stage B is Kust4K and SegFly's *reconstructed*
instances plus VTUAV's 875 drawn masks. These read what notebooks 13-18
harvested -- real detection boxes turned into teacher masks, tens of thousands
of them -- through `src/training/pool_reader.py`, which was the follow-up
`docs/mask_pool_plan.md` left open.

**Before and after, on the same instances.** The whole point of the run is
whether stage B on pools is worth anything, so both notebooks score **stock
EdgeTAM** and the trained checkpoint on the same held-out split, under two
prompts, and then draw the instances whose IoU moved most -- in both
directions. A table of means can hide a model that got better at trucks and
worse at people; the panel cannot.

## The two grades, and why there are two

`pool test` is a held-out slice of the pools themselves: the same distribution
the run trained on, so it is the sensitive number, and its "truth" is a
teacher's guess gated four ways.

`drawn test` is a dataset whose masks a human drew (Kust4K's semantic maps,
decomposed the way `07` decomposes them), held at `role=eval` so no window of
it is ever trained on. It is the honest number and the conservative one.

The two cannot overlap, so cell 2 **drops any pool built from the drawn set's
own frames** and says it did. Training on `kust4k_thermal` while grading on
Kust4K's drawn maps would be scoring on frames the run had seen, and the
stratified split cannot prevent it: the pool and the semantic set are separate
sources with separate permutations.

## Filling an 80 GB card

`--batch` is measured, not chosen, and the ladder now reaches 512 (image mode
holds no clip length and no memory bank, so it fits far more than the video
path). `--steps` is fixed, so a bigger batch means more samples behind the same
number of updates -- and the linear scaling rule says the step should grow with
it. The notebook sets `--lr-scale` from the measured batch against a 16-window
reference, capped, and prints both.

## The two archives HIT-UAV ships as

Its pool was harvested from the kagglehub copy, whose frames sit under
`hit-uav/images/{train,val,test}/`, and the run stages the GitHub archive,
whose frames sit under `normal_json/{train,val,test}/` inside a folder named
after a branch. `Relocator` re-roots a recorded path by matching its *suffix*,
and these two disagree on a component rather than on a depth -- `images/test/f`
against `normal_json/test/f` -- so no suffix of one is a suffix of the other
and every frame reports `no_image`.

The by-name fallback cannot rescue it either, and the reason is worth writing
down: **the archive ships every frame four times**, because its annotations
come in four formats -- `normal_json/<split>/f`, `rotate_json/<split>/f`,
`normal_xml/JPEGImages/f` and `rotate_xml/JPEGImages/f` (the first three are
the same bytes; the fourth is not). Against a record naming
`hit-uav/images/train/f`, the first two share the same two-component tail, and
a reader that must not pick the wrong modality's copy of a name refuses a tie
rather than guessing. Right per frame, and it costs the whole pool: 2 866
frames, none of them indexed.

Pointing the root at the tree that holds the splits settles all of them, which
is what `IMAGE_ROOTS["hituav_thermal"]` does with a glob. But a glob is read
where the plan is built, which is *before* the download cell has run: on a
fresh runtime nothing matches it and the root stays the pattern -- and handing
that pattern to `fetch` as a destination extracts the dataset into a directory
literally named `**`. `images_for` therefore returns two roots now, the plain
folder a download lands in and the pattern frames are read back from, and
cell 2 ends by asking the disk which is which. `resolve_images_root` looks a
few recorded file names up under the root and ranks the trees they land in: by
how much of the recorded path they agree with first, then by how many frames
they re-root; a tie is broken by comparing the bytes, so four copies of one
archive are one answer and DroneVehicle's two modalities are still a refusal.
It prints the re-rooting it did, or that a pool's frames are on no disk here,
which is the same `no_image` with a different fix.

## The failure the deep arms are aimed at

A tracker that holds a car crossing cold asphalt and drops the same car parked
on sun-warmed concrete has not learned "vehicle". It has learned "the bright
blob", and the training sets hand it that shortcut: almost every annotated
thermal target in them is hot against cold ground, so a model can score well on
all of them without ever separating a target from a background that looks like
it.

Two pieces answer it, and the first is the measurement.
`image_loop.instance_contrast` scores each instance by its own signal over the
clutter of the ground immediately around it -- the signal-to-clutter ratio the
thermal literature uses -- and `eval_instances` now reports mean IoU in three
bands of it. A model that reads targets off their brightness has a flat top
band and a collapsed bottom one, which no aggregate mean shows.

The second is `src/training/photometric.py`, which manufactures the bottom band
out of the top one: a window's contrast is collapsed toward its own mean, its
polarity is flipped (white-hot and black-hot are one scene under two sensor
conventions), its transfer curve is jittered, and read noise is added. The
noise is not decoration. Collapsing contrast alone divides the target's signal
and the background's clutter by the same number and leaves the ratio exactly
where it was -- measured, in `tests/test_photometric.py` -- so a collapse
without noise after it is a no-op dressed as an augmentation. Masks, boxes and
classes are untouched: the target stays exactly where it was and only the
evidence for it gets worse.

Training windows only. Validation keeps the plain stream (`Loop.val_stream`,
the same argument `val_loss` makes) because that number selects which epoch's
weights are kept, and a validation set augmented differently each epoch would
be choosing on the draw.

## A budget for a night, and a net under it

22 and 23 run `[2, 24]` epochs of 800 steps with `PATIENCE = 4`. The budget is
long on purpose and the patience is what makes that safe: a stage gives up
after four epochs with no improvement in the validation loss, so the run ends
when it has stopped learning rather than when a counter runs out.

Two things follow from the one-cycle schedule and are worth saying plainly.
`total_steps` is sized from the stage's **budget**, not from where patience
stops it, so a longer budget is a slower anneal -- which is the point of
raising it -- and a stage cut short never reaches the low-rate end of that
descent. Patience is a net, not a tuner: set it loose enough that it only fires
on a genuine plateau.

And the checkpoint reaches Drive on **every improvement**, not only when the
run finishes (`--mirror`). A Colab runtime reclaimed at 4am used to take the
weights with it, because the only copy was on `/content`. The copy goes to a
`.part` and is renamed, so a run killed mid-write leaves the previous good file
rather than a truncated one.

## Cutting a pool tighter than its harvest did

`MIN_BOX_IOU`, and `POOL_MIN_BOX_IOU` per pool, drops instances whose stored
box-IoU falls below it. The harvest already applied a gate; this is a
**second, stricter** one, applied when the index is built rather than when the
teacher ran, so trying 0.7 against 0.8 costs two indexing passes instead of two
harvests. It only ever removes: a pool harvested at 0.5 has no record of what
0.4 would have kept.

Two things to know before raising it. `box_iou` compares the mask's own box
with the annotation's, so it measures *extent*, not shape -- `pool.summarise_gates`
prints mean `area_ratio` beside it, and a pool whose masks all sit near 1.0 is
a pool of rectangles that no threshold repairs. And a threshold is a filter
nobody chose: the same table breaks the cut down by class and by target size,
because a few pixels of slack round a 20 px object costs far more IoU than the
same slack round a 200 px one, and small targets are the axis this project is
judged on. Read those two tables, then pick a number.

A cached index is rebuilt whenever a cut is asked for: the file on disk was
written under whatever cut the run before it used, and nothing in it says
which.

## What none of it measures

Tracking. Every number here scores one prompted frame with the memory path
frozen and never executed. A better encoder is a precondition for a better
tracker, not evidence of one -- that is stage C.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Only these move between the two arms. Everything else is shared source, so a
# difference anywhere else is a bug rather than a variant.
# The gates and the photometric augmentation, off by default: 19 and 20 exist
# to be compared with numbers taken before either knob did, and a run whose data
# and whose windows both changed cannot be read against them.
# How many frames a pool may contribute. AeroVIS is the only one that needs a
# cap in a stage-B run -- 39 943 frames of it beside 2 866 of HIT-UAV is not a
# mixture -- and the RGB pretrain, where it *is* the run, raises it.
CAPS = {"POOL_LIMITS": '{"aerovis_train": 10000, "aerovis_heldout": 1500}',
        # Empty = start from stock EdgeTAM. Point it at a pretrain's
        # output (34/35) to chain, and the run asserts the file is
        # there rather than silently restarting from stock.
        "BASE_CHECKPOINT": '""'}

PLAIN = {"MIN_AREA": "48", "MIN_SIDE": "4", "MAX_AREA": "0.9",
         "CONTRAST_COLLAPSE": "0.0", "POLARITY_FLIP": "0.0",
         "GAMMA_JITTER": "0.0", "SENSOR_NOISE": "0.0"}

# What the deep arms run instead. `MAX_AREA` at a quarter of the frame drops the
# targets that fill the picture -- nothing in this deployment is one -- and
# `MIN_SIDE`/`MIN_AREA` drop the handful of pixels below what a prompt can even
# name.
#
# The photometric numbers are **deliberately milder than the first version of
# this preset**, and the reason is a measurement on real footage rather than a
# preference. The aggressive setting (collapse 0.4 to 0.15x, noise 5) was
# chosen on a synthetic window whose target sat at a signal-to-clutter ratio of
# 20; HIT-UAV's real targets sit at a median of 0.91, so there was no easy end
# to collapse and the setting moved its median only 0.85 -> 0.79 while spending
# dynamic range everywhere. What still earns its place at any contrast is the
# polarity flip -- white-hot and black-hot are one scene -- and the gamma.
# `SENSOR_NOISE` stays non-zero because without it a collapse cannot move the
# ratio at all (`src/training/photometric.py` measures that too).
#
# The right value is per-pool and the run now prints what it needs to choose:
# read the per-band table in the evaluation before turning these up.
HARDER = {**CAPS, "MIN_AREA": "64", "MIN_SIDE": "6", "MAX_AREA": "0.2",
          "CONTRAST_COLLAPSE": "0.25", "POLARITY_FLIP": "0.25",
          "GAMMA_JITTER": "0.25", "SENSOR_NOISE": "2.0"}

# Where AeroVIS's 12.6 GiB of frames are. Both its pools travelled to Drive
# without them -- `write_pool` stores `image_rel` for exactly that reason -- so
# until this named an archive, every arm that plans them indexed 47 921 records
# against an empty `/content/data/AeroVIS` and reported both pools unusable.
# 27, 28 and 35 `REQUIRE_POOLS` AeroVIS, so for them it was not a missing pool
# but a failed assert.
#
# It is a `POOL_ARCHIVES` entry and not a `SOURCE_ZIPS` one because these pools
# are RGB and half these arms are not. `SOURCE_ZIPS` unpacks before the plan
# exists, so a thermal arm would spend 12.6 GiB of a mounted Drive on frames
# nothing in it will open; `POOL_ARCHIVES` runs per *planned* pool, after the
# modality filter, and takes only the members that pool's records name.
#
# Which is why it sits in `INERT` rather than on the arms that use it. Where a
# pool's frames live is a fact about the pool, and 19 and 20 have to differ in
# `MODALITIES` and nothing else or the gap between their numbers is not
# attributable to the RGB windows. 19 carries the same entry and the loop says
# so: dropped at the modality filter, archive never opened.
#
# The path is the archive itself rather than a folder to glob, so it works
# whether the Drive holds the file or a shortcut to the shared release --
# `1DMLagGZ...`, the id `aerovis_selection.json` already records as the one the
# harvest ran against. Naming it here is also what makes the gdown-and-unzip in
# `docs/calisma_plani_tr.md` unnecessary: the frames come out of Drive, capped
# by what the pool asks for, instead of 13.5 GiB off the network each session.
AEROVIS_ZIP = "/content/drive/MyDrive/edgetam-pool/AeroVIS.zip"
AEROVIS_ARCHIVE = (f'"aerovis_train": "{AEROVIS_ZIP}",\n'
                   f'                 "aerovis_heldout": "{AEROVIS_ZIP}"')

INERT = {**PLAIN, **CAPS, "REQUIRE_POOLS": "{}", "CLASS_WEIGHTS": "{}",
         "LR_HEAD": "0", "LR_NECK": "0", "LR_TRUNK": "0",
         "POOL_ARCHIVES": "{" + AEROVIS_ARCHIVE + "}",
         "METHOD": '"finetune"',
         "REFERENCE_CHECKPOINT": '""',
         "EXTRA_DATASETS": "[]", "SKIP_POOLS": "[]",
         "EPOCHS": "[1, 3]", "STEPS": "400", "MIN_BOX_IOU": "0.0",
         "PATIENCE": "0"}

# The three thermal-only pools that must be present for 22 to mean anything,
# with a floor rather than a flag: a pool that resolved twelve frames out of
# forty thousand has arrived in name only.
REQUIRED = ('{"dronevehicle_thermal": 20000, "vtuav_thermal": 20000,\n'
            '                 "hituav_thermal": 2000,\n'
            '                 "kaggle_uav_thermal": 10000}')

# This arm is useful only when it is genuinely 22 plus RGB, rather than a
# thermal rerun whose optional colour files happened not to arrive. AeroVIS is
# required on both sides of its sequence-level split: train supplies the extra
# supervision and heldout supplies the RGB grade. Other RGB pools stay additive
# because not every Drive has harvested all of them.
REQUIRED_RGB_AEROVIS = (
    '{"dronevehicle_thermal": 20000, "vtuav_thermal": 20000,\n'
    '                 "hituav_thermal": 2000,\n'
    '                 "kaggle_uav_thermal": 10000,\n'
    '                 "aerovis_train": 10000,\n'
    '                 "aerovis_heldout": 1500}')

REQUIRED_AEROVIS = ('{"aerovis_train": 10000,\n'
                    '                 "aerovis_heldout": 1500}')

# SegFly ships semantic maps, so its thermal frames come out of `decompose`
# with no teacher pass at all. **15 007 is the frame count, not the yield**: the
# pool harvested from the same labels (`decompose:watershed`, these gates) says
# 5 378 frames and 20 145 instances in its own manifest, because SegFly is
# scenery and only two of its sixteen classes are things -- two of its eight
# 2 000-frame shards contribute under 200 frames between them. The dataset flag
# reads the same maps in `components` mode and lands in the same place, so a run
# that shows ~5 400 SegFly frames is not short of anything; the staging cell
# prints the funnel that says so. It is preferred over the pool because it is
# the same frames under a decomposition this repo can re-run and audit.
SEGFLY = '["segfly:/content/data/SegFly:thermal:components:train"]'

# What each source is worth, per instance, to `aerial.rebalance`.
#
# **SegFly is not in it.** It is the one source here whose masks a human drew
# rather than a teacher guessed, and thinning it to 0.6 threw away a quarter of
# the most reliable supervision in the run to hold down a class balance that
# `car`/`truck` already hold down.
#
# DroneVehicle stays at 0.45 and is the reason the rest of this table exists:
# 155 000 instances, nearly all cars, from one sensor over one city. VTUAV
# joins at 0.8 -- a tracking set is one target followed for thousands of
# frames, so its instances are correlated in a way a detection set's are not,
# and counting each one as a full example over-weights a handful of scenes.
THERMAL_WEIGHTS = ('{"pool/dronevehicle_thermal": 0.45,\n'
                   '                 "pool/dronevehicle_thermal_only": 0.7,\n'
                   '                 "pool/vtuav_thermal": 0.8,\n'
                   '                 "pool/vtuav_rgb": 0.8,\n'
                   '                 "car": 0.7, "truck": 0.7}')

SOURCE_ZIPS_DEFAULT = (
    '[\n'
    '    ["/content/drive/MyDrive/edgetam-pool/segfly/segfly.zip", "SegFly"],\n'
    '    ["/content/drive/MyDrive/edgetam-pool/kust4k/29476610.zip", "Kust4K"],\n'
    ']')

ARMS = {
    "19_thermal_stage_b_pool.ipynb": {
        **INERT,
        "MODALITIES": '["thermal"]',
        "RUN": '"thermal"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal"',
    },
    "20_thermal_stage_b_pool_rgb.ipynb": {
        **INERT,
        "MODALITIES": '["thermal", "rgb"]',
        "RUN": '"thermal_rgb"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal_rgb"',
    },
    # Everything thermal this repo has harvested, the vehicle classes thinned
    # so they cannot outvote the rest, and the rate table inverted so the
    # trunk learns the modality instead of the decoder compensating for it.
    "22_thermal_deep.ipynb": {
        **HARDER,
        "EPOCHS": "[2, 24]",
        "PATIENCE": "4",
        "MIN_BOX_IOU": "0.8",
        "STEPS": "800",
        "EXTRA_DATASETS": SEGFLY,
        "SKIP_POOLS": '["segfly_thermal", "vtuav_lt"]',
        "MODALITIES": '["thermal"]',
        "RUN": '"thermal_deep"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal_deep"',
        "REQUIRE_POOLS": REQUIRED,
        "CLASS_WEIGHTS": THERMAL_WEIGHTS,
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        # VTUAV's eleven tracking archives are 154 GiB and the disk is not.
        # The pool names the ~40 000 frames it wants, so only those come out.
        "POOL_ARCHIVES": ('{"vtuav_thermal": "/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_lt_thermal": '
                          '"/content/drive/MyDrive/VTUAV"}'),
        "METHOD": '"finetune"',
        "REFERENCE_CHECKPOINT": '""',
    },
    # 22's third arm, and the only line that differs from it: LoRA instead of
    # a partial fine-tune. Same pools, same gate, same thinning, same rates,
    # same seed, same split file -- so the two numbers differ by the method
    # and by nothing else, which is the only thing that makes them comparable.
    "23_thermal_deep_lora.ipynb": {
        **HARDER,
        "EPOCHS": "[2, 24]",
        "PATIENCE": "4",
        "MIN_BOX_IOU": "0.8",
        "STEPS": "800",
        "EXTRA_DATASETS": SEGFLY,
        "SKIP_POOLS": '["segfly_thermal", "vtuav_lt"]',
        "MODALITIES": '["thermal"]',
        "RUN": '"thermal_deep"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal_deep"',
        "REQUIRE_POOLS": REQUIRED,
        "CLASS_WEIGHTS": THERMAL_WEIGHTS,
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        "POOL_ARCHIVES": ('{"vtuav_thermal": "/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_lt_thermal": '
                          '"/content/drive/MyDrive/VTUAV"}'),
        "METHOD": '"lora"',
        "REFERENCE_CHECKPOINT": '""',
    },
    # ----------------------------------------------------------------
    # The two pretrains: everything harvested, in the modality it belongs to.
    # ----------------------------------------------------------------
    #
    # 22 and 23 are experiments -- one question each, held-out grades, floors
    # that fail the run when a pool arrives in name only. These two are the
    # other job: take every pool of one modality and spend a long schedule on
    # them, to produce the checkpoint the experiments start from.
    #
    # **Is putting everything in sensible?** For one modality, with weights,
    # yes; blindly, no, and the numbers say why. The thermal side spans two
    # orders of magnitude between sources -- HIT-UAV is 2 866 frames, VTUAV is
    # about forty thousand -- so an unweighted pool is a VTUAV model with a
    # rounding error of HIT-UAV in it, and VTUAV is one sensor over one set of
    # scenes. `CLASS_WEIGHTS` is what makes "all of it" mean all of it rather
    # than the biggest of it.
    #
    # Mixing the two modalities is the thing not to do here. 27 exists to
    # measure that question honestly; a pretrain that quietly blends them
    # produces a checkpoint neither arm can be compared against.
    "34_pretrain_thermal_aerial.ipynb": {
        **HARDER,
        # Long and patient: this is the run whose output everything else
        # starts from, so it gets the budget and the early stop rather than a
        # fixed number of epochs.
        "EPOCHS": "[3, 40]",
        "PATIENCE": "5",
        "STEPS": "1000",
        # The harvests differ in how hard their gates were, so the run re-cuts
        # them to one standard rather than trusting six.
        "MIN_BOX_IOU": "0.75",
        "EXTRA_DATASETS": SEGFLY,
        # SegFly enters as its own drawn semantic maps, not as the pool
        # harvested from them; the VTUAV long-term pools are out by request.
        "SKIP_POOLS": '["segfly_thermal", "vtuav_lt"]',
        "MODALITIES": '["thermal"]',
        "RUN": '"pretrain_thermal_aerial"',
        "MIRROR_DIR": ('"/content/drive/MyDrive/edgetam-stage-b/'
                       'pretrain_thermal_aerial"'),
        "REQUIRE_POOLS": REQUIRED,
        "CLASS_WEIGHTS": THERMAL_WEIGHTS,
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        "POOL_ARCHIVES": ('{"vtuav_thermal": "/content/drive/MyDrive/VTUAV"}'),
        "METHOD": '"finetune"',
        "REFERENCE_CHECKPOINT": '""',
    },
    # The RGB pretrain, separate on purpose and not a variant of the one above.
    # AeroVIS alone is 39 943 frames and 1 095 567 instances (VisDrone, UAVDT
    # and SeaDronesSee re-labelled), held out **by sequence** -- 18 sequences
    # and 7 978 frames that no training window can reach -- which is the one
    # clean drawn grade in this repo that is not thermal.
    #
    # `POOL_LIMITS` in cell 1 caps AeroVIS at 10 000 frames for the stage-B
    # arms, where it is a side dish. Here it is the meal: raise that cap in the
    # settings cell if the disk and the schedule allow, and the run prints what
    # the cap cost it.
    "35_pretrain_rgb_aerial.ipynb": {
        **HARDER,
        "EPOCHS": "[3, 40]",
        "PATIENCE": "5",
        "STEPS": "1000",
        "MIN_BOX_IOU": "0.75",
        # No thermal drawn grade in an RGB run: AeroVIS's held-out sequences
        # are the grade, and they are pool frames rather than drawn maps.
        "EVAL_DRAWN": "None",
        "EXTRA_DATASETS": "[]",
        # VisDrone is inside AeroVIS; a VisDrone pool beside it would put the
        # held-out sequences back into training through another annotation.
        "SKIP_POOLS": '["visdrone", "vtuav_lt", "segfly_thermal"]',
        "MODALITIES": '["rgb"]',
        "RUN": '"pretrain_rgb_aerial"',
        "MIRROR_DIR": ('"/content/drive/MyDrive/edgetam-stage-b/'
                       'pretrain_rgb_aerial"'),
        "REQUIRE_POOLS": REQUIRED_AEROVIS,
        # The cap the stage-B arms put on AeroVIS is lifted here, which is the
        # whole difference between a side dish and the meal: 39 943 frames and
        # 1 095 567 instances are what the release actually holds, and 10 000
        # of them was a number chosen to keep it from swamping a thermal run.
        "POOL_LIMITS": '{"aerovis_train": 40000, "aerovis_heldout": 3000}',
        # A polarity flip is a *thermal* sensor convention. `harden` already
        # refuses to apply it to a colour window, so this is 0 for honesty
        # rather than for effect: a config that reads 0.25 would say the run
        # does something it cannot.
        "POLARITY_FLIP": "0.0",
        # Cars are 400 450 of AeroVIS's instances and `vehicle` another
        # 307 555; without thinning them the run learns "a target is a car"
        # before it learns anything else.
        "CLASS_WEIGHTS": ('{"car": 0.5, "vehicle": 0.5, "truck": 0.7,\n'
                          '                 "pool/aerovis_train": 0.9}'),
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        "POOL_ARCHIVES": ('{"vtuav_rgb": "/content/drive/MyDrive/VTUAV",\n'
                          '                 ' + AEROVIS_ARCHIVE + '}'),
        "METHOD": '"finetune"',
        "REFERENCE_CHECKPOINT": '""',
    },
    # The direct answer to 22's next question: keep its schedule, optimiser,
    # gates, thermal floors and drawn thermal grade, but let RGB pools into the
    # same batches and require AeroVIS to be present. VisDrone is excluded
    # because AeroVIS contains VisDrone frames; mixing the two would leak the
    # release's held-out sequences into training through another annotation.
    "27_thermal_deep_rgb_aerovis.ipynb": {
        **HARDER,
        "EPOCHS": "[2, 24]",
        "PATIENCE": "4",
        "MIN_BOX_IOU": "0.8",
        "STEPS": "800",
        "EXTRA_DATASETS": SEGFLY,
        "SKIP_POOLS": '["segfly_thermal", "visdrone", "vtuav_lt"]',
        "MODALITIES": '["thermal", "rgb"]',
        "RUN": '"thermal_deep_rgb_aerovis"',
        "MIRROR_DIR": ('"/content/drive/MyDrive/edgetam-stage-b/'
                       'thermal_deep_rgb_aerovis"'),
        "REQUIRE_POOLS": REQUIRED_RGB_AEROVIS,
        "CLASS_WEIGHTS": THERMAL_WEIGHTS,
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        "POOL_ARCHIVES": ('{"vtuav_thermal": "/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_lt_thermal": '
                          '"/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_rgb": '
                          '"/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_lt_rgb": '
                          '"/content/drive/MyDrive/VTUAV",\n'
                          '                 ' + AEROVIS_ARCHIVE + '}'),
        "METHOD": '"finetune"',
        # `_finetune` is not decoration: the settings cell appends METHOD to
        # MIRROR_DIR so a fine-tune and a LoRA of the same RUN cannot overwrite
        # each other's Drive folder, which means 22 mirrors to
        # `thermal_deep_finetune/` and never to `thermal_deep/`. Naming the
        # folder without it pointed this A/B at a path nothing ever writes, and
        # cell 3 only warns -- so the run would have gone ahead scoring against
        # stock EdgeTAM while the log said it was comparing against 22.
        "REFERENCE_CHECKPOINT": ('"/content/drive/MyDrive/edgetam-stage-b/'
                                 'thermal_deep_finetune/'
                                 'edgetam_pool_thermal_deep_512.pt"'),
    },
    # A clean RGB control for the thermal and mixed arms. AeroVIS is both the
    # required training source and the held-out RGB grade; VisDrone stays out
    # because AeroVIS contains it. No drawn thermal set or thermal source zip
    # is allowed to slip through the modality filter by a separate code path.
    "28_rgb_deep_aerovis.ipynb": {
        **HARDER,
        "EPOCHS": "[2, 24]",
        "PATIENCE": "4",
        "MIN_BOX_IOU": "0.8",
        "STEPS": "800",
        "EVAL_DRAWN": "None",
        "EXTRA_DATASETS": "[]",
        "SOURCE_ZIPS": "[]",
        "SKIP_POOLS": '["visdrone", "vtuav_lt"]',
        "MODALITIES": '["rgb"]',
        "RUN": '"rgb_deep_aerovis"',
        "MIRROR_DIR": ('"/content/drive/MyDrive/edgetam-stage-b/'
                       'rgb_deep_aerovis"'),
        "REQUIRE_POOLS": REQUIRED_AEROVIS,
        "CLASS_WEIGHTS": '{"car": 0.7, "truck": 0.7}',
        "LR_HEAD": "0",
        "LR_NECK": "1e-4",
        "LR_TRUNK": "1e-4",
        "POOL_ARCHIVES": ('{"vtuav_rgb": "/content/drive/MyDrive/VTUAV",\n'
                          '                 "vtuav_lt_rgb": '
                          '"/content/drive/MyDrive/VTUAV",\n'
                          '                 ' + AEROVIS_ARCHIVE + '}'),
        "METHOD": '"finetune"',
        "REFERENCE_CHECKPOINT": '""',
    },
}

BRANCH = "claude/thermal-stage-b-training-43ktcl"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Every knob, the repo, the card, and what the card can take
# --------------------------------------------------------------------------

code('''
MODALITIES  = {{MODALITIES}}
RUN         = {{RUN}}
MIRROR_DIR  = {{MIRROR_DIR}}
REFERENCE_CHECKPOINT = {{REFERENCE_CHECKPOINT}}
BASE_CHECKPOINT = {{BASE_CHECKPOINT}}
DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"
POOL_ROOT   = "/content/pool"
DATA_ROOT   = "/content/data"
WORK        = "/content/work"
STAGE_DIR   = "/content/drive/MyDrive/datasets"
DRIVE_MY    = "/content/drive/MyDrive"

EVAL_DRAWN  = {{EVAL_DRAWN}}
EXTRA_DATASETS = {{EXTRA_DATASETS}}
EVAL_SPEC   = "kust4k:{root}:thermal:components:all"
SKIP_POOLS  = {{SKIP_POOLS}}
REQUIRE_POOLS = {{REQUIRE_POOLS}}
CLASS_WEIGHTS = {{CLASS_WEIGHTS}}
POOL_ROLE   = "all"
POOL_ROLES  = {"kaggle_uav_thermal": "train", "aerovis_train": "train",
               "aerovis_heldout": "eval"}
POOL_MODALITIES = {"aerovis_train": "rgb", "aerovis_heldout": "rgb"}
POOL_LIMITS = {{POOL_LIMITS}}
POOL_ZIP_MAX_MB = 2048
MIN_BOX_IOU = {{MIN_BOX_IOU}}
POOL_MIN_BOX_IOU = {}

VTUAV_PARTS = []
VTUAV_VIS_PARTS = []

IMAGE_ROOTS = {"kaggle_uav_thermal": "/content/data/kaggle_uav_thermal",
               "aerovis_train": "/content/data/AeroVIS",
               "aerovis_heldout": "/content/data/AeroVIS",
               "hituav_thermal": "/content/data/HIT_UAV/**/normal_json",
               "vtuav_lt_thermal": "/content/data/VTUAV_lt_ir",
               "vtuav_lt_rgb": "/content/data/VTUAV_lt_rgb"}
KAGGLE_DATASETS = {
    "kaggle_uav_thermal": "umuttuygurr/aerial-uav-thermal-inferred-unified-dataset",
}

POOL_ARCHIVES = {{POOL_ARCHIVES}}

SOURCE_ZIPS = {{SOURCE_ZIPS}}

IMAGES = [
    ["hituav",       "hituav",       "HIT_UAV",      []],
    ["dronevehicle", "dronevehicle", "DroneVehicle", ["train"]],
    ["kust4k",       "kust4k",       "Kust4K",       ["tir", "labels", "rgb"]],
    ["visdrone",     "visdrone",     "VisDrone",     []],
    ["segfly_rgb",   "segfly_rgb",   "SegFly_RGB",   []],
    ["segfly",       "",             "SegFly",       []],
    ["vtuav",        "vtuav_track",  "VTUAV",        VTUAV_PARTS],
    ["vtuav_vis",    "vtuav_vis",    "VTUAV_VIS",    VTUAV_VIS_PARTS],
    ["kaggle",       "",             "kaggle_uav_thermal", []],
    ["aerovis",      "",             "AeroVIS",      []],
]

SIZE           = 512
PER_IMAGE      = 2
MAX_INSTANCES  = 8
MIN_AREA       = {{MIN_AREA}}
MIN_SIDE       = {{MIN_SIDE}}
MAX_AREA       = {{MAX_AREA}}
FILL           = 0.25
CONTRAST_COLLAPSE = {{CONTRAST_COLLAPSE}}
CONTRAST_FLOOR    = 0.15
POLARITY_FLIP     = {{POLARITY_FLIP}}
GAMMA_JITTER      = {{GAMMA_JITTER}}
SENSOR_NOISE      = {{SENSOR_NOISE}}
JITTER         = 32
PROMPT         = "mix"
PROMPT_JITTER  = 0.3
METHOD         = {{METHOD}}
LORA_R         = 16
LORA_ALPHA     = 0
LORA_DROPOUT   = 0.0
ANCHOR_WEIGHT  = 0.0
BLEND_ALPHAS   = []
MAX_REGRESSION = 0.15
EPOCHS         = {{EPOCHS}}
STEPS          = {{STEPS}}
PATIENCE       = {{PATIENCE}}
VAL_BATCHES    = 24
DEPTH          = 2
BATCH          = 0
BATCH_CEILING  = 512
BATCH_RESERVE  = 0.12
LR_REFERENCE   = 16
LR_SCALE_MAX   = 4.0
LR_HEAD        = {{LR_HEAD}}
LR_NECK        = {{LR_NECK}}
LR_TRUNK       = {{LR_TRUNK}}
SEED           = 0
SCORE_PROMPTS  = ["box", "point"]
PANEL_CASES    = 6
PANEL_WINDOWS  = 400
FETCH          = True

REPO_URL = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH   = "{{BRANCH}}"
REPO_DIR = "/content/sam-dedection"
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
else:
    for _git in (["fetch", "--depth", "1", "origin", BRANCH],
                 ["reset", "--hard", f"origin/{BRANCH}"]):
        subprocess.run(["git", "-C", REPO_DIR, *_git], check=False)
print("repo at", subprocess.run(["git", "-C", REPO_DIR, "log", "-1",
                                 "--format=%h %s"], capture_output=True,
                                text=True).stdout.strip())
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive")
except Exception as _mount_error:
    print("no Colab Drive mount:", _mount_error)

import torch
_TORCH_WAS = torch.__version__

subprocess.run(["bash", "scripts/setup_edgetam.sh"], check=False)

REQUIREMENTS = [_line.split("#")[0].strip() for _line in
                (Path(REPO_DIR) / "requirements.txt").read_text().splitlines()]
REQUIREMENTS = [_line for _line in REQUIREMENTS
                if _line and not _line.lower().startswith("opencv")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer",
                "tqdm", "matplotlib"], check=False)

import importlib.metadata as _md
assert _md.version("torch") == _TORCH_WAS, (
    f"pip replaced torch {_TORCH_WAS} with {_md.version('torch')} on disk. "
    f"Restore it before restarting: pip install --force-reinstall "
    f"torch=={_TORCH_WAS}")

import importlib

OPENCV = ["opencv-python", "opencv-python-headless", "opencv-contrib-python",
          "opencv-contrib-python-headless"]

def cv2_works():
    try:
        import cv2
        return hasattr(cv2, "imread")
    except Exception:
        return False

CV2_REPAIRED = False
if not cv2_works():
    print("cv2 is broken -- reinstalling one distribution over the several "
          "that share its directory")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    *OPENCV], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "opencv-python-headless>=4.8"], check=False)
    for _stale in [n for n in list(sys.modules) if n.split(".")[0] == "cv2"]:
        del sys.modules[_stale]
    importlib.invalidate_caches()
    CV2_REPAIRED = True
assert cv2_works(), (
    "cv2 imports but has no imread, and reinstalling did not fix it in this "
    "kernel. Runtime > Restart session (your /content files survive it) and "
    "run this cell again -- the repair is already on disk.")
if CV2_REPAIRED:
    print("cv2 repaired; restart the runtime if anything below still fails")

EDGETAM = str(Path(REPO_DIR) / "third_party" / "EdgeTAM")
if EDGETAM not in sys.path:
    sys.path.insert(sys.path.index(REPO_DIR) + 1, EDGETAM)
importlib.invalidate_caches()
import sam2
assert Path(sam2.__file__).resolve().parent.parent == Path(EDGETAM).resolve(), (
    f"sam2 imported from {Path(sam2.__file__).parent} and not the EdgeTAM "
    f"checkout. pip uninstall -y sam2 SAM-2, then re-run this cell.")
BASE_CKPT = str(Path(EDGETAM) / "checkpoints" / "edgetam.pt")
assert Path(BASE_CKPT).is_file(), "edgetam.pt did not download"
if BASE_CHECKPOINT:
    assert Path(BASE_CHECKPOINT).is_file(), (
        f"BASE_CHECKPOINT is set to {BASE_CHECKPOINT} and nothing is there. "
        f"Run the pretrain that writes it first, or clear the setting to "
        f"start from stock EdgeTAM.")
    BASE_CKPT = BASE_CHECKPOINT
    _from = (Path(BASE_CHECKPOINT).stem
             .replace("edgetam_pool_", "").replace(f"_{SIZE}", ""))[:24]
    RUN = f"{RUN}_from_{_from}"
    MIRROR_DIR = f"{MIRROR_DIR.rstrip('/')}_from_{_from}"
    print("starting from", BASE_CKPT, "-- not stock EdgeTAM.")
    print("   the run is named", RUN, "so it lands beside the arm that starts "
          "from stock rather than on top of it. MIRROR_DIR and the checkpoint "
          "name both carry it: `pretrain then this` and `this alone` are two "
          "measurements of the same recipe, and the whole reason to run the "
          "second is to find out whether the first was worth it.")
    print("   BASE_CHECKPOINT is the weights this run starts from, which is a "
          "different question from REFERENCE_CHECKPOINT: that one only names "
          "what the before/after is scored against. Chaining a pretrain into "
          "a narrower run needs this one -- without it every arm restarts "
          "from stock EdgeTAM and the pretrain's epochs buy the run below it "
          "nothing at all.")
    print("   the before/after below is measured against this checkpoint, so "
          "it reports what THIS run added on top of the pretrain rather than "
          "what the two together added to stock.")

assert METHOD in ("finetune", "lora"), f"METHOD is finetune or lora, not {METHOD!r}"
if METHOD != "finetune":
    MIRROR_DIR = f"{MIRROR_DIR.rstrip('/')}_{METHOD}"
for _dir in (POOL_ROOT, DATA_ROOT, WORK, MIRROR_DIR,
             str(Path(WORK) / "index"), str(Path(REPO_DIR) / "checkpoints")):
    Path(_dir).mkdir(parents=True, exist_ok=True)
INDEX_DIR = str(Path(WORK) / "index")
TAG = RUN if METHOD == "finetune" else f"{RUN}_{METHOD}"
CHECKPOINT = str(Path(REPO_DIR) / "checkpoints" / f"edgetam_pool_{TAG}_{SIZE}.pt")

_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
VRAM = round(_props.total_memory / 2 ** 30, 1) if _props else 0.0
if _props:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _probe = torch.randn(256, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (_probe @ _probe).sum().item()
    del _probe
WORKERS = min(2 * (os.cpu_count() or 4), 24)

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print(_props.name if _props else "no GPU", VRAM, "GiB |", WORKERS, "loader threads")
print("run:", RUN, "| modalities:", MODALITIES, "| checkpoint ->", CHECKPOINT)
print("method:", METHOD, f"(r={LORA_R})" if METHOD == "lora" else "",
      "| anchor weight:", ANCHOR_WEIGHT,
      "| blend sweep:", BLEND_ALPHAS or "off")
print("stage B only: no stage A, no distillation, base =", BASE_CKPT)
if ANCHOR_WEIGHT:
    print("   the anchor is the model this run starts from, which without a "
          "stage A is stock EdgeTAM: the loss gains a term for how far the "
          "encoder's features travel from it, which is what holds down the "
          "instances stock already handled. It costs one extra frozen encoder "
          "pass per batch.")
''')


# --------------------------------------------------------------------------
# 2. Unpack the pools, work out what each one is, and fetch its frames
# --------------------------------------------------------------------------
#
# Discovery rather than a hard-coded list, because the Drive folder's shape is
# whatever the harvest notebooks left: 15 writes four pools as four zips, 16/17
# write one per sequence, 18 writes four, and a sharded staging writes numbered
# parts whose members still carry the pool name. Unzipping everything into one
# root and then asking `pool_datasets` what is there is the only reading of
# that folder that cannot go stale.
#
# The drop rule is the measurement rule: a pool built from the drawn set's own
# frames cannot be trained on while that set is the grade.

code('''
from src.training.pool import RECORD_FILE
from src.training.pool_reader import (acceptance, discover_pools,
                                      extract_frames, group_records, link_pool,
                                      resolve_images_root)

_done = Path(POOL_ROOT) / ".unpacked"
_done.mkdir(parents=True, exist_ok=True)
assert Path(DRIVE_POOLS).is_dir(), f"{DRIVE_POOLS} is not there -- set DRIVE_POOLS"

for _zip in sorted(Path(DRIVE_POOLS).rglob("*.zip")):
    _size = round(_zip.stat().st_size / 2 ** 20, 1)
    _marker = _done / (str(_zip.relative_to(DRIVE_POOLS)).replace("/", "__")
                       + ".done")
    if _marker.is_file():
        continue
    if _size > POOL_ZIP_MAX_MB:
        print("skipped", _zip.name, _size, "MiB -- a pool holds masks only, "
              "so this is source data, not a pool")
        continue
    try:
        with zipfile.ZipFile(_zip) as _handle:
            _members = _handle.namelist()
            if not any(_m.endswith("record.json") for _m in _members):
                print("skipped", _zip.name, "-- no record.json in it")
                continue
            _handle.extractall(POOL_ROOT)
    except Exception as _unzip_error:
        print("!! could not read", _zip.name, "--", _unzip_error)
        continue
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.touch()
    print("unpacked", _zip.name, _size, "MiB,", len(_members), "files")

for _name, _folder in discover_pools(DRIVE_POOLS).items():
    _marker = _done / (_name + ".folder.done")
    if _marker.is_file() or (Path(POOL_ROOT) / _name).is_dir():
        continue
    shutil.copytree(_folder, Path(POOL_ROOT) / _name, dirs_exist_ok=True)
    _marker.touch()
    print("copied", _name, "from Drive (it was staged unzipped)")

print()
RAW = group_records(POOL_ROOT)
POOLS, TEACHERS, HARVEST = {}, {}, {}
print(f"{'pool':<28}{'frames':>8}{'boxes':>9}{'masks':>9}{'kept':>7}   "
      f"{'first gate that stopped the rest':<34}teacher")
for _name, _records in RAW.items():
    POOLS[_name] = str(link_pool(_records, Path(POOL_ROOT) / "_by_name" / _name))
    HARVEST[_name] = acceptance(_records)
    TEACHERS[_name] = "+".join(HARVEST[_name]["teachers"])
    _top = ", ".join(f"{k} {v}" for k, v in
                     list(HARVEST[_name]["rejected"].items())[:2]) or "nothing"
    print(f"{_name:<28}{HARVEST[_name]['frames']:>8}"
          f"{HARVEST[_name]['attempted']:>9}{HARVEST[_name]['accepted']:>9}"
          f"{HARVEST[_name]['rate']:>7.1%}   {_top:<34}{TEACHERS[_name]}")
print("\\na box the first gate stops is never measured against its own "
      "annotation: `reject_reason` returns at the first failure, and "
      "`teacher_iou` is the teacher's own confidence, which this repo "
      "measured as the weak gate. A pool whose rejects pile up there has "
      "not been shown to hold bad masks -- only unsure ones.")
_impossible = {_name: (HARVEST[_name]["accepted"], _least)
               for _name, _least in REQUIRE_POOLS.items()
               if _name in HARVEST and HARVEST[_name]["accepted"] < _least}
assert not _impossible, (
    f"REQUIRE_POOLS cannot be met by these pools: {_impossible} (harvested, "
    f"least). This is the harvest's own accepted count, before any frame is "
    f"downloaded and before the gates are re-cut -- so it is an upper bound "
    f"and cell 3's check can only come out lower. Failing here costs a "
    f"minute; failing there costs the download first. Either the pool zip on "
    f"Drive is not the one this run was written against, or REQUIRE_POOLS "
    f"asks for more than was ever harvested.")
if REQUIRE_POOLS:
    print("\\nrequired pools, at harvest:",
          ", ".join(f"{_n} {HARVEST[_n]['accepted']}/{_l}"
                    for _n, _l in REQUIRE_POOLS.items() if _n in HARVEST)
          or "none of them are here yet")
    _absent = [_n for _n in REQUIRE_POOLS if _n not in HARVEST]
    if _absent:
        print("   not unpacked at all:", ", ".join(_absent),
              "-- their zips are not under DRIVE_POOLS, so cell 3 will stop.")

FOUND = sorted(POOLS)
assert FOUND, f"no record.json under {POOL_ROOT} -- nothing was unpacked"

GUESSED = set()

def modality_of(pool):
    if pool in POOL_MODALITIES:
        return POOL_MODALITIES[pool]
    GUESSED.add(pool)
    lowered = pool.lower()
    if lowered.endswith("_rgb") or "_rgb_" in lowered or "rgb" in lowered.split("_"):
        return "rgb"
    return "thermal"

def images_for(pool):
    """`(key, recipe, images root, parts, the folder a download lands in)`.

    The last two roots are different things, and reading them as one is what
    put HIT-UAV's archive in a directory named `**`: IMAGE_ROOTS may hold a
    *pattern*, because that archive nests its frames under a folder named
    after a branch that nobody can spell in advance -- and the pattern is
    resolved here, where the plan is built, which is before the download cell
    has run. On a fresh runtime it matches nothing and stays a pattern, and
    `fetch` then extracts the dataset into a folder literally called `**`.
    A download always goes to the dataset's plain folder; a pattern only ever
    narrows where the frames are read back from.
    """
    lowered = pool.lower()
    for key, recipe, folder, parts in IMAGES:
        if key in lowered:
            root = IMAGE_ROOTS.get(pool) or str(Path(DATA_ROOT) / folder)
            plain = root
            if "*" in root:
                plain = str(Path(DATA_ROOT) / folder)
                _hits = [_p for _p in Path("/").glob(root.lstrip("/"))
                         if _p.is_dir()]
                _hits.sort(key=lambda _p: (len(_p.parts), str(_p)))
                root = str(_hits[-1]) if _hits else root
            return key, recipe, root, list(parts), plain
    return None, "", "", [], ""

PLAN, DROPPED = [], []
for _pool in FOUND:
    _key, _recipe, _root, _parts, _plain = images_for(_pool)
    _modality = modality_of(_pool)
    if any(skip in _pool.lower() for skip in SKIP_POOLS):
        DROPPED.append((_pool, "SKIP_POOLS"))
    elif _modality not in MODALITIES:
        DROPPED.append((_pool, f"modality {_modality} not in MODALITIES"))
    elif _key is None:
        DROPPED.append((_pool, "no entry in IMAGES -- add one"))
    else:
        PLAN.append({"pool": _pool, "dir": str(POOLS[_pool]), "key": _key,
                     "recipe": _recipe, "images": _root, "parts": _parts,
                     "fetch": _plain,
                     "modality": _modality,
                     "role": POOL_ROLES.get(_pool, POOL_ROLE)})

print(f"\\n{'pool':<28}{'modality':<10}{'role':<7}{'frames from':<40}")
for _row in PLAN:
    print(f"{_row['pool']:<28}{_row['modality']:<10}{_row['role']:<7}"
          f"{_row['images']:<40}")
for _pool, _role in POOL_ROLES.items():
    if _role == "train" and any(r["pool"] == _pool for r in PLAN):
        print(f"   {_pool} feeds training only -- it never reaches val or test, "
              f"so it cannot inflate the score with a domain the deployment "
              f"does not have.")
_guessed = sorted(_pool for _pool in GUESSED & {_r["pool"] for _r in PLAN}
                  if "rgb" not in _pool.lower()
                  and "thermal" not in _pool.lower()
                  and "tir" not in _pool.lower())
if _guessed:
    print(f"   !! modality guessed for {_guessed} -- their names say neither. "
          f"A pool not naming `rgb` is read as thermal, which converts colour "
          f"frames to grey and trains them as thermal. Put them in "
          f"POOL_MODALITIES if that is wrong.")
if {"visdrone", "aerovis"} <= {_row["key"] for _row in PLAN}:
    print("   !! a VisDrone pool and an AeroVIS pool are both in this run. "
          "AeroVIS *is* VisDrone re-labelled (plus UAVDT and SeaDronesSee), "
          "so they are the same frames and no AeroVIS grade is held out any "
          "more. Drop one.")
for _pool, _why in DROPPED:
    print(f"{_pool:<28}dropped   {_why}")
assert PLAN, "nothing left to train on -- read the dropped list above"

_wanted = {}
for _row in PLAN:
    _wanted.setdefault((_row["recipe"], _row["fetch"]), set()).update(_row["parts"])
if EVAL_DRAWN:
    _drawn = next((r for r in IMAGES if r[0] == EVAL_DRAWN), None)
    assert _drawn, f"EVAL_DRAWN={EVAL_DRAWN} has no IMAGES entry"
    EVAL_ROOT = str(Path(DATA_ROOT) / _drawn[2])
    _wanted.setdefault((_drawn[1], EVAL_ROOT), set()).update(_drawn[3])
else:
    EVAL_ROOT = ""

def unpack(archive, target, label=""):
    """Every member the archive can still give, one at a time.

    `extractall` stops at the first bad member and leaves the rest on the
    floor. A Drive copy read by two runtimes at once returns a bad CRC often
    enough that one corrupt JPEG out of seventy thousand must not cost the
    other sixty-nine thousand. Already-extracted members are skipped, so this
    is also the resume path.
    """
    bad, taken = [], 0
    Path(target).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
        for member in members:
            landing = Path(target) / member
            if landing.exists() and (landing.is_dir() or landing.stat().st_size):
                continue
            try:
                handle.extract(member, target)
                taken += 1
            except Exception:
                bad.append(member)
    print(f"   {label or Path(archive).name}: {taken} extracted, "
          f"{len(members) - taken - len(bad)} already there, "
          f"{len(bad)} unreadable")
    if bad:
        print("   unreadable:", bad[:5], "..." if len(bad) > 5 else "")
    return bad

_DONE = Path(DATA_ROOT) / ".unpacked"
_DONE.mkdir(parents=True, exist_ok=True)
_PICTURES = ("*.jpg", "*.jpeg", "*.png")

def on_disk(root):
    root = Path(root)
    if not root.is_dir():
        return 0
    return sum(1 for _glob in _PICTURES for _ in root.rglob(_glob))

def complete(archive, target, mark):
    """Whether `target` already holds every frame `archive` carries.

    Answered for a staged archive before anything is unpacked, and again for
    each part of a dataset whose archives are all on this disk: an archive
    the tree is short of is walked member by member, which finishes what an
    earlier run left behind instead of reading it as done.

    "Some jpgs are under the root" is the check this used to make, and it is
    the check that let a quarter-extracted DroneVehicle look finished to every
    run after the one a bad CRC stopped -- the pools then lost 9 859 of 13 098
    frames to `no_image` and nothing said why. Counting the archive's own
    image members against the tree answers the question the jpg probe was
    standing in for, and costs one central-directory read.
    """
    if (_DONE / mark).is_file():
        return True
    try:
        with zipfile.ZipFile(archive) as handle:
            want = sum(1 for _m in handle.namelist()
                       if _m.lower().endswith((".jpg", ".jpeg", ".png")))
    except Exception:
        return False
    if want and on_disk(target) >= want:
        (_DONE / mark).touch()
        return True
    return False

for _archive, _folder in SOURCE_ZIPS:
    _target = Path(DATA_ROOT) / _folder
    if not Path(_archive).is_file():
        print("not there, skipping:", _archive)
        continue
    if complete(_archive, _target, _folder + ".zip.done"):
        print("already on disk:", _target)
        continue
    print("unzipping", _archive,
          round(Path(_archive).stat().st_size / 2 ** 30, 2), "GiB ->", _target)
    if not unpack(_archive, _target, _folder):
        (_DONE / (_folder + ".zip.done")).touch()

_PLANNED = {_row["pool"] for _row in PLAN}
for _pool, _folder in POOL_ARCHIVES.items():
    if _pool not in POOLS:
        print(f"!! {_pool} is not a pool here -- known: {sorted(POOLS)}")
        continue
    if _pool not in _PLANNED:
        print(f"   {_pool}: dropped by SKIP_POOLS or by the modality filter, "
              f"so its frames stay inside the archive -- nothing in this run "
              f"will open one of them")
        continue
    _key, _recipe, _root, _parts, _plain = images_for(_pool)
    _root = _plain
    _shelf = (sorted(Path(_folder).glob("*.zip")) if Path(_folder).is_dir()
              else [Path(_folder)])
    if _parts:
        _shelf = [_a for _a in _shelf if _a.stem in _parts]
    _records = sorted(Path(POOLS[_pool]).rglob(RECORD_FILE))
    print(f"\\n{_pool}: {len(_records)} records against {len(_shelf)} archives "
          f"({sum(_a.stat().st_size for _a in _shelf) / 2 ** 30:.1f} GiB) "
          f"-> {_root}")
    _report = extract_frames(_records, _shelf, _root)
    print(f"   asked {_report['asked']}, extracted {_report['taken']}, "
          f"already there {_report['already']}, missing {_report['missing']}, "
          f"unreadable {len(_report['unreadable'])}")
    for _name, _took in sorted(_report["by_archive"].items()):
        if _took:
            print(f"      {_name:<24}{_took:>8} frames")
    if _report["missing"]:
        print(f"   !! {_report['missing']} frames are in no archive listed "
              f"here -- name the missing parts in POOL_ARCHIVES")

if FETCH:
    from tools.fetch_datasets import fetch, staged
    Path(STAGE_DIR).mkdir(parents=True, exist_ok=True)
    for (_recipe, _root), _parts in sorted(_wanted.items()):
        if not _recipe or (_recipe == "vtuav_track" and not _parts):
            print("not fetched:", _root, "-- no recipe, or no parts named. "
                  "SegFly is a 761-shard parquet plan, VTUAV is ~16 GiB per "
                  "part and Kaggle sets come through kagglehub below, so none "
                  "of them is downloaded on a whim. Stage it, name its parts "
                  "in cell 1, or let the pool drop.")
            continue
        _mark = Path(_root).name + ".fetch.done"
        if (_DONE / _mark).is_file():
            print("already on disk:", _root)
            continue
        _search = (STAGE_DIR, str(Path(DRIVE_MY) / Path(_root).name),
                   DRIVE_MY, "/content/staging")
        _staged = {_p: staged(_p, _search) for _p in (sorted(_parts) or [_recipe])}
        if all(_c is not None for _c in _staged.values()):
            print("staged already:", _root, "-- counting each archive against "
                  "the tree rather than downloading it")
            _bad, _short = [], False
            for _part, _copy in sorted(_staged.items()):
                if complete(_copy, _root, f"{Path(_root).name}.{_part}.done"):
                    print("already on disk:", _copy.name, "->", _root)
                    continue
                _short = True
                print("unpacking staged", _copy.name, "->", _root)
                _bad += unpack(_copy, Path(_root), _part)
            if not _bad and not _short:
                (_DONE / _mark).touch()
            continue
        if Path(_root).is_dir() and any(
                any(Path(_root).rglob(_glob)) for _glob in ("*.jpg", "*.png")):
            print("already on disk:", _root, "-- staged by hand, not checked")
            continue
        try:
            fetch(_recipe, Path(_root), tuple(sorted(_parts)) or None,
                  stage=STAGE_DIR,
                  staging=_search)
            (_DONE / _mark).touch()
        except Exception as _fetch_error:
            print("!!", _recipe, "->", _root, "failed:", _fetch_error)
            for _part in sorted(_parts) or [_recipe]:
                _copy = staged(_part, _search)
                if _copy is None:
                    continue
                print("   retrying", _copy.name, "member by member")
                unpack(_copy, Path(_root), _part)

for _pool, _slug in KAGGLE_DATASETS.items():
    if not any(_row["pool"] == _pool for _row in PLAN):
        continue
    _root = Path(IMAGE_ROOTS.get(_pool, ""))
    if _root and _root.is_dir() and any(
            any(_root.rglob(_glob)) for _glob in ("*.jpg", "*.png")):
        print("already on disk:", _root)
        continue
    try:
        import kagglehub
        _got = kagglehub.dataset_download(_slug)
        _root.parent.mkdir(parents=True, exist_ok=True)
        if not _root.exists():
            os.symlink(_got, _root)
        print("kaggle:", _slug, "->", _got)
    except Exception as _kaggle_error:
        print("!! could not fetch", _slug, "--", _kaggle_error)
        print("   put its frames under", _root, "or set IMAGE_ROOTS in cell 1")

print()
for _row in PLAN:
    _found = resolve_images_root(RAW[_row["pool"]], _row["images"])
    if _found is None and str(Path(_row["images"]).parent) != DATA_ROOT:
        _found = resolve_images_root(RAW[_row["pool"]], _row["images"],
                                     search_root=DATA_ROOT)
    if _found is None:
        print(f"!! {_row['pool']}: not one of its recorded frames is anywhere "
              f"under {_row['images']}, so every frame of it will read as "
              f"`no_image`. Its download is missing or incomplete -- or its "
              f"frames are elsewhere, which IMAGE_ROOTS in cell 1 is for.")
    elif str(_found) != _row["images"]:
        print(f"   {_row['pool']}: frames are under {_found}, not "
              f"{_row['images']} -- re-rooted. Which mirror a pool was "
              f"harvested from and how the archive unpacked here are two "
              f"independent facts, and only the disk holds the second.")
        _row["images"] = str(_found)

import numpy as np

from src.training.aerial import IMAGE_SUFFIXES, InstanceGates, list_frames, rebalance
from src.training.datasets import parse
from src.training.pool_reader import Relocator, why_no_image
from tools.train_encoder import build_indexes

AUDIT_GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE,
                            max_area=MAX_AREA, fill=FILL)
PROBES = 200
READY = {}

print(f"{'pool':<28}{'records':>9}{'probed':>8}{'found':>7}   images root")
for _row in PLAN:
    _records = sorted(Path(_row["dir"]).rglob(RECORD_FILE))
    _probe = _records[::max(len(_records) // PROBES, 1)][:PROBES]
    _relocate = Relocator(_row["images"])
    _recorded, _hit, _joined = "", 0, 0
    for _record in _probe:
        _body = json.loads(_record.read_text())
        _named = _body.get("image", "")
        _recorded = _recorded or _named
        _inside = _relocate.direct(_body.get("image_rel"))
        _joined += _inside is not None
        _hit += (_inside or _relocate(_named)) is not None
    READY[_row["pool"]] = {"records": len(_records), "probed": len(_probe),
                           "found": _hit, "images": _row["images"],
                           "joined": _joined, "recorded": _recorded}
    print(f"{_row['pool']:<28}{len(_records):>9}{len(_probe):>8}{_hit:>7}   "
          f"{_row['images']}")
    if _joined:
        print(f"   {_joined} of them by joining the path the archive itself "
              f"recorded, which cannot pick the wrong file -- AeroVIS names "
              f"frames vd_001/0000001.jpg and 91 of its 117 sequences hold a "
              f"file of that name, so a match by name would be a coin flip.")
    if _hit < len(_probe):
        _why = why_no_image(_row["dir"], _row["images"])
        print(f"   recorded as {_recorded}")
        print(f"   {_why['verdict']}")
        for _same in _why.get("same_name_here", [])[:3]:
            print(f"   that name is here: {_same}")
        if _relocate.ambiguous:
            print(f"   {_relocate.ambiguous} of the probes matched more than "
                  f"one file by name and were refused rather than guessed. An "
                  f"archive that ships a frame twice does that -- name the "
                  f"tree that holds the splits in IMAGE_ROOTS.")

print()
for _spec, _spec_root in ([(_s, DATA_ROOT) for _s in EXTRA_DATASETS]
                          + ([(EVAL_SPEC, EVAL_ROOT)] if EVAL_DRAWN else [])):
    _flag = _spec.format(root=_spec_root, data=DATA_ROOT)
    try:
        _request = parse(_flag, AUDIT_GATES)
        _dataset, _where = _request.source.spec, Path(_request.root)
        _frames = [_p for _p in _where.glob(_dataset.glob(_request.modality))
                   if _p.suffix.lower() in IMAGE_SUFFIXES]
        _maps = [_p for _p in _where.glob(_dataset.mask_glob(_request.modality))
                 if _p.suffix.lower() in IMAGE_SUFFIXES]
        _pairs = list_frames(_where, _dataset, _request.modality)
        _index = build_indexes([_request], Path(INDEX_DIR), WORKERS)
    except Exception as _audit_error:
        print(f"!! {_flag} -> {type(_audit_error).__name__}: "
              f"{str(_audit_error).splitlines()[0]}")
        continue
    _rejected = {}
    for _entry in _index:
        for _reason, _count in _entry.rejects.items():
            _rejected[_reason] = _rejected.get(_reason, 0) + _count
    print(f"{_dataset.name}: {len(_frames)} frames and {len(_maps)} maps on "
          f"disk -> {len(_pairs)} paired by stem -> {len(_index)} frames with "
          f"an instance ({sum(len(_e.instances) for _e in _index)} instances)")
    print(f"   {len(_pairs) - len(_index)} paired frames carry nothing of "
          f"{_dataset.things} that the gates kept; the gates stopped {_rejected}")
    if len(_frames) != len(_maps):
        print(f"   !! the two halves disagree by {abs(len(_frames) - len(_maps))} "
              f"files, so that many frames can never pair. A half-extracted "
              f"archive looks exactly like this -- cell 2 re-reads any archive "
              f"the tree is short of, so re-run it before reading anything below.")
    if CLASS_WEIGHTS:
        _kept, _thin = rebalance(_index, CLASS_WEIGHTS, seed=SEED)
        print(f"   CLASS_WEIGHTS thins it to {len(_kept)} frames / "
              f"{_thin['instances']['after']} instances before training sees it")

print()
print(f"{'pool':<28}{'inst':>8}{'p50 px':>8}{'p50':>8}{'p99.9':>8}{'max':>8}"
      f"{'<min':>6}{'>max':>6}   share of the frame each target covers")
for _row in PLAN:
    _areas, _shares = [], []
    _recs = sorted(Path(_row["dir"]).rglob(RECORD_FILE))
    for _record in _recs[::max(len(_recs) // 400, 1)][:400]:
        try:
            _body = json.loads(_record.read_text())
            _frame = float(int(_body["shape"][0]) * int(_body["shape"][1]))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        for _inst in _body.get("instances", []):
            _px = _inst.get("area")
            if _px and _frame:
                _areas.append(float(_px))
                _shares.append(float(_px) / _frame)
    if not _shares:
        continue
    _a, _s = np.array(_areas), np.array(_shares)
    print(f"{_row['pool']:<28}{len(_s):>8}{np.percentile(_a, 50):>8.0f}"
          f"{np.percentile(_s, 50) * 100:>7.3f}%{np.percentile(_s, 99.9) * 100:>7.2f}%"
          f"{_s.max() * 100:>7.2f}%{float((_a < MIN_AREA).mean()) * 100:>5.1f}%"
          f"{float((_s > MAX_AREA).mean()) * 100:>5.2f}%")
print("   MIN_AREA and MAX_AREA read against the pool's own annotations, so "
      "the two gates can be set from what the data holds rather than from a "
      "feel for how big a target looks. Measured on AeroVIS: the median "
      "instance is 967 px (~31x31) and covers 0.065% of its frame; the "
      "largest car ever annotated reaches 6.0%, the largest truck 13.6%, the "
      "largest bus 4.7%. Everything above 25% is UAVDT's coarse `vehicle` "
      "class -- 603 of 1 411 468 instances, 0.04% -- which is the class this "
      "gate exists to stop teaching the decoder that the whole frame is an "
      "answer. A `>max` column in the percents rather than the hundredths is "
      "the signal that MAX_AREA is cutting real targets on this pool.")

print()
try:
    import torch

    from src.training.aerial import sample_windows
    from src.training.image_loop import collate, instance_contrast
    from src.training.pool_reader import index_pool, parse_pool

    print(f"{'source':<28}{'windows':>8}{'median':>8}{'<1':>7}{'1-3':>7}{'>3':>7}"
          f"   how far its targets stand out")
    for _row in PLAN:
        try:
            _probe = parse_pool(f"{_row['dir']}:{_row['images']}:"
                                f"{_row['modality']}:train", AUDIT_GATES)
            _slice = index_pool(_probe.pool, _probe.images, _probe.modality,
                                "train", AUDIT_GATES, _row["pool"], limit=48,
                                workers=WORKERS)
            _windows = sample_windows(_slice, size=SIZE, per_image=1, seed=SEED)[:24]
            if not _windows:
                continue
            _scores = instance_contrast(collate(_windows, "cpu"))
        except Exception as _contrast_error:
            print(f"{_row['pool']:<28}   not measured: "
                  f"{type(_contrast_error).__name__}")
            continue
        if not len(_scores):
            continue
        _bands = [float((( _scores >= _lo) & (_scores < _hi)).mean())
                  for _lo, _hi in ((0, 1), (1, 3), (3, 1e9))]
        print(f"{_row['pool']:<28}{len(_scores):>8}{float(np.median(_scores)):>8.2f}"
              f"{_bands[0]:>7.0%}{_bands[1]:>7.0%}{_bands[2]:>7.0%}")
    print("   the signal-to-clutter ratio of each pool's own targets, on a "
          "sample of its windows. It is what decides whether the photometric "
          "knobs in cell 1 have anything to work on: a source already sitting "
          "under 1 has no easy end left to collapse, and one sitting above 3 "
          "is where `the bright blob` is still a free answer. HIT-UAV measures "
          "0.91 on its own annotations, so do not assume.")
except Exception as _audit_error:
    print("per-source contrast not measured:", _audit_error)

subprocess.run(["df", "-h", "/content"], check=False)
''')


# --------------------------------------------------------------------------
# 3. The index, the split, and the flags every later cell reuses
# --------------------------------------------------------------------------
#
# Built in-process and cached, so the training and scoring subprocesses below
# reuse the same file and therefore provably the same instances. A pool whose
# frames were never downloaded is dropped here with its reason rather than
# taking the run down: five pools that resolve are worth more than a traceback.

code('''
import numpy as np

from src.training.aerial import (InstanceGates, rebalance, sample_windows,
                                 save_splits, split_index)
from src.training.datasets import parse
from src.training.image_loop import ImageSplit
from src.training.pool_reader import (SKIP_REASONS, exclude_frames, frame_keys,
                                      index_pool, load_pool_index, parse_pool,
                                      save_pool_index, spread, why_no_image)
from tools.train_encoder import build_indexes

GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE, max_area=MAX_AREA,
                      fill=FILL)

INDEX, DATASET_FLAGS, DRAWN_HELD = [], [], set()
if EVAL_DRAWN:
    try:
        _drawn_flag = EVAL_SPEC.format(root=EVAL_ROOT)
        _drawn = build_indexes([parse(_drawn_flag, GATES)], Path(INDEX_DIR),
                               WORKERS)
        _parts = split_index(_drawn, seed=SEED)
        DRAWN_HELD = frame_keys(_parts["val"]) | frame_keys(_parts["test"])
        DATASET_FLAGS.append(_drawn_flag)
        INDEX.extend(_drawn)
        print(f"drawn grade: {EVAL_DRAWN}, {len(_drawn)} frames, "
              f"{sum(len(e.instances) for e in _drawn)} instances, "
              f"{len(DRAWN_HELD)} frames held out of it")
        print("   these are drawn *semantic* maps decomposed into instances, "
              "so where the decomposition fused two vehicles a model that "
              "separates them is scored wrong. It is the best drawn "
              "annotation here without VTUAV's archives; read it as a floor.")
    except (ValueError, FileNotFoundError, AssertionError) as _drawn_error:
        print(f"!! no drawn grade: {EVAL_DRAWN} -> "
              f"{str(_drawn_error).splitlines()[0]}")
        print("   the test split becomes the pools' own held-out slice, so "
              "its truth is the teacher's. Read the number knowing that.")

for _spec in EXTRA_DATASETS:
    _flag = _spec.format(root=DATA_ROOT, data=DATA_ROOT)
    try:
        _extra = build_indexes([parse(_flag, GATES)], Path(INDEX_DIR), WORKERS)
    except Exception as _extra_error:
        print(f"!! extra dataset {_flag} -> {type(_extra_error).__name__}: "
              f"{str(_extra_error).splitlines()[0]}")
        continue
    _name = _extra[0].source.spec.name if _extra else "?"
    _cut = 0
    if DRAWN_HELD and _name == EVAL_DRAWN:
        _before = len(_extra)
        _extra = exclude_frames(_extra, DRAWN_HELD)
        _cut = _before - len(_extra)
    DATASET_FLAGS.append(_flag)
    INDEX.extend(_extra)
    print(f"extra dataset {_name:<14}{len(_extra):>7} frames "
          f"{sum(len(e.instances) for e in _extra):>9} instances"
          f"{f'  (-{_cut} the drawn grade holds out)' if _cut else ''}")
    print("   these are drawn semantic maps decomposed into instances, so the "
          "target is the decomposition's and not a teacher's. That is a "
          "different kind of noise from a pool's, not a worse one -- but a "
          "run holding both a pool and the dataset it was harvested from is "
          "the same pixels twice under two targets. Put the pool in "
          "SKIP_POOLS when you add its dataset here.")

print()
POOL_FLAGS, FAILED, CUTS = [], [], {}
for _row in PLAN:
    _flag = f"{_row['dir']}:{_row['images']}:{_row['modality']}:{_row['role']}"
    _request = parse_pool(_flag, GATES)
    _cache = Path(INDEX_DIR) / f"{_request.cache_name}.json"
    try:
        _part = None
        if _cache.is_file():
            _part = load_pool_index(_cache, GATES, _row["role"])
            _short = {_k: _v for _k, _v in (_part[0].rejects if _part else {}).items()
                      if _k in ("no_image", "unreadable_image", "shape_mismatch")}
            if _short:
                print(f"   {_row['pool']}: the cached index was built while "
                      f"{sum(_short.values())} frame(s) were unreadable "
                      f"({_short}), so it is provisional -- indexing again in "
                      f"case the download it was waiting on has finished")
                _part = None
        _cut_at = POOL_MIN_BOX_IOU.get(_row["pool"], MIN_BOX_IOU)
        if _part is not None and _cut_at:
            print(f"   {_row['pool']}: re-indexing, the cached index predates "
                  f"a min_box_iou of {_cut_at}")
            _part = None
        if _part is None:
            _part = index_pool(_request.pool, _request.images, _request.modality,
                               _row["role"], GATES, _request.name,
                               workers=WORKERS, min_box_iou=_cut_at,
                               progress=progress, report=print)
            save_pool_index(_cache, _part)
    except Exception as _index_error:
        FAILED.append((_row["pool"], f"{type(_index_error).__name__}: "
                                     f"{str(_index_error).splitlines()[0]}"))
        continue
    _skips = {k: v for k, v in _part[0].rejects.items() if k in SKIP_REASONS}
    _capped = ""
    _cap = POOL_LIMITS.get(_row["pool"])
    if _cap and len(_part) > _cap:
        _was, _seqs = len(_part), len({e.frame.name.rsplit("/", 1)[0]
                                       for e in _part})
        _part = spread(_part, _cap, SEED)
        _kept = len({e.frame.name.rsplit("/", 1)[0] for e in _part})
        _capped = (f"  (spread to {_cap} of {_was} over {_kept}/{_seqs} "
                   f"sequences)")
    _leak = ""
    if DRAWN_HELD and _row["key"] == EVAL_DRAWN:
        _before = len(_part)
        _part = exclude_frames(_part, DRAWN_HELD)
        _cut = _before - len(_part)
        CUTS[_row["pool"]] = _cut
        _leak = (f"  (-{_cut} frames the drawn grade holds out)" if _cut
                 else "  (shares no held-out key)")
    if not _part:
        FAILED.append((_row["pool"], "nothing left after the overlap filter"))
        continue
    POOL_FLAGS.append(_flag)
    INDEX.extend(_part)
    print(f"{_row['pool']:<28}{len(_part):>7} frames "
          f"{sum(len(e.instances) for e in _part):>9} instances  "
          f"{_row['modality']:<8}{TEACHERS[_row['pool']]}"
          f"{('  skipped ' + str(_skips)) if _skips else ''}{_capped}{_leak}")

for _pool, _why in FAILED:
    print(f"{_pool:<28}unusable  {_why}")
    if "no_image" not in _why:
        continue
    _row = next(r for r in PLAN if r["pool"] == _pool)
    _report = why_no_image(_row["dir"], _row["images"])
    print(f"    recorded {_report['recorded'][0]['image']}")
    print(f"    -> {_report['verdict']}")
    if _report["extensions"]:
        print(f"    {_report['root']} holds {_report['files_under_root']} "
              f"files {_report['extensions']}")
assert POOL_FLAGS, "no pool resolved its frames -- check the IMAGES roots above"

_have = {}
for _row in PLAN:
    if any(_row["dir"] in _f for _f in POOL_FLAGS):
        _have[_row["pool"]] = sum(len(e.instances) for e in INDEX
                                  if e.source
                                  and e.source.spec.name == f"pool/{_row['pool']}")
_short = {_name: (_have.get(_name, 0), _least)
          for _name, _least in REQUIRE_POOLS.items()
          if _have.get(_name, 0) < _least}
if REQUIRE_POOLS:
    print(f"\\n{'required pool':<28}{'instances':>11}{'least':>9}  state")
    for _name, _least in REQUIRE_POOLS.items():
        _got = _have.get(_name, 0)
        print(f"{_name:<28}{_got:>11}{_least:>9}  "
              f"{'ok' if _got >= _least else 'MISSING'}")
assert not _short, (
    f"this run is defined by pools that did not arrive: {_short} (got, least). "
    f"Read the unusable list above -- a `no_image` there means the frames are "
    f"not on disk, and POOL_ARCHIVES in cell 1 is how a pool takes them out of "
    f"an archive Drive already holds. Training without them would answer a "
    f"different question than the one this notebook is for.")

assert not CUTS or any(CUTS.values()), (
    f"every pool built from {EVAL_DRAWN} ({sorted(CUTS)}) shares none of the "
    f"{len(DRAWN_HELD)} held-out keys, so the two readers name frames "
    f"differently and the overlap cannot be removed. Set EVAL_DRAWN = None, "
    f"or drop these pools.")
_blind = sorted(_p for _p, _c in CUTS.items() if not _c)
if _blind and any(CUTS.values()):
    print(f"\\n!! {', '.join(_blind)} shares no key with the held-out grade "
          f"while its siblings do, so it is a different set of frames rather "
          f"than a naming mismatch -- a broken-registration half, most "
          f"likely. It trains only; what it can still carry is a scene whose "
          f"other modality is graded, which is the registered-pair risk and "
          f"not the same pixels.")

_used = {TEACHERS[_row["pool"]] for _row in PLAN
         if any(_row["dir"] in _f for _f in POOL_FLAGS)}
if len(_used) > 1:
    print(f"\\n!! two teachers in one training set: {sorted(_used)}. The run "
          f"still works, but a per-pool difference now has two causes and "
          f"neither can be read off the result.")

COMMON = []
for _flag in POOL_FLAGS:
    COMMON += ["--pool", _flag]
for _flag in DATASET_FLAGS:
    COMMON += ["--dataset", _flag]
COMMON += ["--index", INDEX_DIR, "--size", str(SIZE),
           "--per-image", str(PER_IMAGE), "--max-instances", str(MAX_INSTANCES),
           "--min-area", str(MIN_AREA), "--min-side", str(MIN_SIDE),
           "--max-area", str(MAX_AREA), "--fill", str(FILL), "--seed", str(SEED)]

if CLASS_WEIGHTS:
    INDEX, _balance = rebalance(INDEX, CLASS_WEIGHTS, seed=SEED)
    print(f"\\nthinning: {_balance['frames']['before']} frames / "
          f"{_balance['instances']['before']} instances -> "
          f"{_balance['frames']['after']} / {_balance['instances']['after']}")
    _was_total = max(_balance["instances"]["before"], 1)
    _now_total = max(_balance["instances"]["after"], 1)
    print(f"{'class':<24}{'was':>10}{'share':>8}{'now':>10}{'share':>8}")
    for _name, (_was, _now) in list(_balance["by_class"].items())[:12]:
        print(f"{_name:<24}{_was:>10}{_was / _was_total:>8.1%}"
              f"{_now:>10}{_now / _now_total:>8.1%}")
    print(f"{'source':<34}{'was':>10}{'share':>8}{'now':>10}{'share':>8}")
    for _src, (_was, _now) in _balance["by_source"].items():
        print(f"{_src:<34}{_was:>10}{_was / _was_total:>8.1%}"
              f"{_now:>10}{_now / _now_total:>8.1%}")
    _matched = [_n for _n in CLASS_WEIGHTS if _n not in _balance["unmatched"]]
    assert _matched, (
        f"CLASS_WEIGHTS names {_balance['unmatched']} and nothing here matched "
        f"any of them, so the thinning did nothing. A key is a class name, a "
        f"source, or `source:class` -- read the two tables above and use those "
        f"names.")
    if _balance["unmatched"]:
        print(f"   !! no pool carries {_balance['unmatched']} -- those weights "
              f"thinned nothing. Not fatal, {_matched} did apply, but a "
              f"misspelt class looks exactly like a class that is already rare.")
    print("   thinned per instance, not per frame: dropping a frame that holds "
          "one pedestrian beside six cars would throw the pedestrian away "
          "too. A frame left with nothing is dropped.")

SPLITS = split_index(INDEX, seed=SEED)

DATASET_OF = {f"pool/{_row['pool']}": _row["key"] for _row in PLAN}
if EVAL_DRAWN:
    DATASET_OF[EVAL_DRAWN] = EVAL_DRAWN

def dataset_of(entry):
    _spec = entry.source.spec.name if entry.source else ""
    return DATASET_OF.get(_spec, _spec)

def stem_of(entry):
    if entry.source is not None and entry.source.mode == "pool":
        return entry.frame.name.lower()
    return Path(entry.frame.name).stem.lower()

GRADED = {}
for _name in ("val", "test"):
    for _entry in SPLITS[_name]:
        GRADED.setdefault(dataset_of(_entry), set()).add(stem_of(_entry))

_kept, _dropped = [], {}
for _entry in SPLITS["train"]:
    _key = dataset_of(_entry)
    if stem_of(_entry) in GRADED.get(_key, ()):
        _dropped[_key] = _dropped.get(_key, 0) + 1
    else:
        _kept.append(_entry)
SPLITS["train"] = _kept
if _dropped:
    print("dropped from training, graded elsewhere in the same dataset:")
    for _key, _count in sorted(_dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {_key:<24}{_count:>7} frames")
    print("   registered pairs are the same scene twice -- DroneVehicle's RGB "
          "and thermal halves, Kust4K's -- and each half is its own source "
          "with its own permutation, so a frame can train in one and be "
          "scored in the other. Same geometry, same objects, same mask.")
    print("   two pools match on the whole key a harvest recorded, not on its "
          "stem: VTUAV numbers every sequence from zero, so `000120` names one "
          "frame per sequence and a stem match would drop every sequence's "
          "frame because one of them is graded. The drawn grade is the case "
          "that needs a stem, and `exclude_frames` above has already done it.")

SPLIT_FILE = str(save_splits(Path(WORK) / "splits.json", SPLITS))
COMMON += ["--splits", SPLIT_FILE]

def windows(name, jitter):
    return ImageSplit(sample_windows(SPLITS[name], size=SIZE, per_image=PER_IMAGE,
                                     max_instances=MAX_INSTANCES, jitter=jitter,
                                     seed=SEED))
TRAIN, VAL, TEST = windows("train", JITTER), windows("val", 0), windows("test", 0)

print()
for _name, _split in (("train", TRAIN), ("val", VAL), ("test", TEST)):
    print(f"{_name:<6}{len(SPLITS[_name]):>7} frames{len(_split.samples):>8} windows"
          f"{sum(len(s.instances) for s in _split.samples):>9} instances")
print("\\ntrain windows by source:")
for _source, _count in TRAIN.sources.items():
    print(f"  {_source:<34}{_count:>8}")
_by_modality = {}
for _sample in TRAIN.samples:
    _key = "thermal" if (_sample.source is None or _sample.source.gray) else "rgb"
    _by_modality[_key] = _by_modality.get(_key, 0) + 1
_total = max(sum(_by_modality.values()), 1)
print("  " + "  ".join(f"{_k}: {_v} ({_v / _total:.0%})"
                       for _k, _v in sorted(_by_modality.items())))
print("\\ntest windows by source:")
for _source, _count in TEST.sources.items():
    print(f"  {_source:<34}{_count:>8}")
assert not ({id(e) for e in SPLITS["train"]} & {id(e) for e in SPLITS["test"]})
assert TEST.samples, ("the test split is empty -- every pool is role=train "
                      "and there is no drawn grade, so nothing can be scored")

_leaked = DRAWN_HELD & frame_keys(
    [e for e in SPLITS["train"] if e.source and e.source.mode == "pool"])
assert not _leaked, (
    f"{len(_leaked)} frames the drawn grade holds out are still in a pool's "
    f"training half, e.g. {sorted(_leaked)[:3]}")
print(f"\\nno pool trains on any of the {len(DRAWN_HELD)} frames the drawn "
      f"grade holds out" if DRAWN_HELD else "\\nno drawn grade to protect")
''')


# 4. The batch this card takes, and the before picture
# --------------------------------------------------------------------------
#
# Stock is scored first and deliberately: it is the floor everything after this
# is measured against, and taking it before the training run means a runtime
# that dies during training still leaves the baseline on Drive.
#
# Two prompts, because 12 measured that one of them cannot see an encoder
# change: an exact ground-truth box states most of the mask on an isolated
# target, and the same checkpoints separated by 0.086 under a centre point
# landed within 0.02 under a box.

code('''
from src.training.finetune import apply_freeze
from src.training.image_loop import auto_batch_size
from tools.train_encoder import build_model

def score_to(checkpoint, tag, prompt):
    out = Path(WORK) / f"score_{tag}_{prompt}.json"
    if out.is_file():
        return json.loads(out.read_text())
    subprocess.run(
        [sys.executable, "tools/eval_instances.py", *COMMON,
         "--checkpoint", checkpoint, "--split", "test", "--prompt", prompt,
         "--batch", str(max(BATCH // 2, 1)), "--device", "cuda",
         "--json", str(out)], check=True)
    return json.loads(out.read_text())

if BATCH <= 0:
    _model = build_model(SIZE, BASE_CKPT, "cuda")
    apply_freeze(_model, "encoder")
    BATCH = auto_batch_size(_model, TRAIN, device="cuda",
                            maximum=BATCH_CEILING, reserve=BATCH_RESERVE)
    del _model
    torch.cuda.empty_cache()
ACCUM = 1
LR_SCALE = round(min(max(BATCH / LR_REFERENCE, 1.0), LR_SCALE_MAX), 3)
print(f"batch {BATCH} x accum {ACCUM} on {VRAM} GiB | lr-scale {LR_SCALE} "
      f"(linear rule against {LR_REFERENCE} windows, capped at {LR_SCALE_MAX})")

COMPARE_CKPT = BASE_CKPT
COMPARE_TAG = "stock"
if REFERENCE_CHECKPOINT:
    if Path(REFERENCE_CHECKPOINT).is_file():
        COMPARE_CKPT = REFERENCE_CHECKPOINT
        COMPARE_TAG = "thermal_22"
        print("direct A/B reference:", REFERENCE_CHECKPOINT)
    else:
        print("!! 22 reference is not on Drive yet:", REFERENCE_CHECKPOINT)
        print("   scoring against stock for now; after 22 finishes, rerun this "
              "cell and the score/panel cells for a direct thermal-vs-mixed A/B")

BEFORE = {p: score_to(COMPARE_CKPT, COMPARE_TAG, p) for p in SCORE_PROMPTS}
for _prompt, _row in BEFORE.items():
    print(f"{COMPARE_TAG:<10}{_prompt:<6} mean {_row['mean_iou']:.4f}  "
          f">=.5 {_row['iou_50']:.3f}  small {_row['small_mean_iou']:.4f}  "
          f"n={_row['instances']}")
''')


# --------------------------------------------------------------------------
# 5. Stage B
# --------------------------------------------------------------------------

code('''
import time
_started = time.time()
_method_flags = ["--method", METHOD]
if METHOD == "lora":
    _method_flags += ["--lora-r", str(LORA_R), "--lora-dropout", str(LORA_DROPOUT)]
    if LORA_ALPHA:
        _method_flags += ["--lora-alpha", str(LORA_ALPHA)]
for _part, _rate in (("head", LR_HEAD), ("neck", LR_NECK), ("trunk", LR_TRUNK)):
    if _rate:
        _method_flags += [f"--lr-{_part}", str(_rate)]
if LR_HEAD or LR_NECK or LR_TRUNK:
    print(f"rates overridden in the encoder stage: head {LR_HEAD or 'table'}, "
          f"neck {LR_NECK or 'table'}, trunk {LR_TRUNK or 'table'} -- "
          f"absolute, so --lr-scale {LR_SCALE} does not touch the ones named. "
          f"The head stage ({EPOCHS[0]} epoch) keeps its own warmup rate: it "
          f"trains the decoder alone, and that is the stage that adapts it.")
    if LR_TRUNK and LR_HEAD and LR_TRUNK > LR_HEAD:
        print("   the trunk leads the head here, which inverts the shipped "
              "table. A modality shift lives in the trunk, and the reason the "
              "table holds it back -- a training set too small for the "
              "features it carries -- is weaker at this size. It is still a "
              "measurement, not a given.")
subprocess.run(
    [sys.executable, "tools/train_encoder.py", *COMMON, *_method_flags,
     "--base", BASE_CKPT, "--out", CHECKPOINT,
     "--prompt", PROMPT, "--prompt-jitter", str(PROMPT_JITTER),
     "--jitter", str(JITTER), "--batch", str(BATCH), "--accum", str(ACCUM),
     "--contrast-collapse", str(CONTRAST_COLLAPSE),
     "--contrast-floor", str(CONTRAST_FLOOR),
     "--polarity-flip", str(POLARITY_FLIP),
     "--gamma-jitter", str(GAMMA_JITTER),
     "--sensor-noise", str(SENSOR_NOISE),
     "--lr-scale", str(LR_SCALE), "--steps", str(STEPS),
     "--epochs", str(EPOCHS[0]), str(EPOCHS[1]),
     "--patience", str(PATIENCE), "--mirror", MIRROR_DIR,
     "--val-batches", str(VAL_BATCHES), "--workers", str(WORKERS),
     "--depth", str(DEPTH),
     "--anchor-weight", str(ANCHOR_WEIGHT), "--device", "cuda",
     "--json", str(Path(WORK) / "run.json")], check=True)

RUN_LOG = json.loads((Path(WORK) / "run.json").read_text())
assert Path(CHECKPOINT).is_file(), "training wrote no checkpoint"
shutil.copy(CHECKPOINT, Path(MIRROR_DIR) / Path(CHECKPOINT).name)
shutil.copy(Path(WORK) / "run.json", Path(MIRROR_DIR) / "run.json")
print(f"{RUN_LOG['best_val_loss']:.4f} best val loss, "
      f"{RUN_LOG['seconds'] / 60:.0f} min, peak {RUN_LOG['peak_gib']:.1f} GiB, "
      f"batch {RUN_LOG['batch']} -> {MIRROR_DIR}")
print("wall clock", round((time.time() - _started) / 60, 1), "min")
''')


# --------------------------------------------------------------------------
# 5b. Trading a little of the gain for the instances it lost
# --------------------------------------------------------------------------
#
# The first thermal run improved 1 222 of 1 707 held-out instances under a
# point prompt and made 281 worse. Those 281 are instances stock EdgeTAM
# already handled, and no amount of extra pool data removes the category: an
# encoder that moves at all moves some of them the wrong way.
#
# `theta = (1 - a) * base + a * tuned` is the cheapest answer -- no retraining,
# and the merged LoRA checkpoint has the same keys, so it applies to either
# method. The sweep is scored on **val** and paired per instance, because the
# question is not "is the mean higher" but "which instances did each alpha win
# and lose"; picking alpha on the test split would be fitting the grade.

code('''
def per_instance(checkpoint, tag, prompt, split):
    """`{instance key: IoU}` for one checkpoint, cached per tag."""
    rows = Path(WORK) / f"rows_{tag}_{prompt}_{split}.json"
    if not rows.is_file():
        subprocess.run(
            [sys.executable, "tools/eval_instances.py", *COMMON,
             "--checkpoint", checkpoint, "--split", split, "--prompt", prompt,
             "--batch", str(max(BATCH // 2, 1)), "--device", "cuda",
             "--json", str(Path(WORK) / f"score_{tag}_{prompt}_{split}.json"),
             "--per-instance", str(rows)], check=True)
    return {r["key"]: r["iou"] for r in json.loads(rows.read_text())["rows"]}

BLEND = {}
if not BLEND_ALPHAS:
    print("no blend sweep (BLEND_ALPHAS is empty). Set it to e.g. "
          "[1.0, 0.8, 0.6, 0.4] to trade a little of the gain for the "
          "held-out instances this run made worse; 1.0 is the trained "
          "checkpoint itself and belongs in the list as the baseline.")
else:
    _prompt = SCORE_PROMPTS[-1]
    _stock = per_instance(BASE_CKPT, "stock", _prompt, "val")
    _rows = []
    for _alpha in sorted({float(a) for a in BLEND_ALPHAS}, reverse=True):
        _name = f"a{int(round(_alpha * 100)):03d}"
        _path = CHECKPOINT
        if _alpha < 1.0:
            _path = str(Path(WORK) / f"blend_{_name}.pt")
            if not Path(_path).is_file():
                subprocess.run(
                    [sys.executable, "tools/blend_checkpoints.py",
                     "--base", BASE_CKPT, "--tuned", CHECKPOINT,
                     "--alpha", str(_alpha), "--out", _path], check=True)
        _scored = per_instance(_path, f"{TAG}_{_name}", _prompt, "val")
        _shared = sorted(set(_scored) & set(_stock))
        assert _shared, ("the two scorings share no instance key -- they were "
                         "run on different splits or different flags")
        _deltas = [_scored[_k] - _stock[_k] for _k in _shared]
        _lost = [_d for _d in _deltas if _d < -0.05]
        _rows.append({
            "alpha": _alpha, "path": _path, "instances": len(_shared),
            "mean_iou": sum(_scored[_k] for _k in _shared) / len(_shared),
            "delta": sum(_deltas) / len(_deltas),
            "better": sum(1 for _d in _deltas if _d > 0.05),
            "worse": len(_lost),
            "rate": len(_lost) / len(_deltas),
            "lost": -sum(_lost) / max(len(_lost), 1)})

    print(f"\\n{'alpha':>6}{'val mean IoU':>14}{'vs stock':>10}{'better':>9}"
          f"{'worse':>8}{'worse %':>9}{'when worse':>12}")
    for _row in _rows:
        print(f"{_row['alpha']:>6.2f}{_row['mean_iou']:>14.4f}"
              f"{_row['delta']:>+10.4f}{_row['better']:>9}{_row['worse']:>8}"
              f"{_row['rate']:>8.1%}{-_row['lost']:>+12.4f}")
    _capped = [_r for _r in _rows if _r["rate"] <= MAX_REGRESSION]
    BLEND = max(_capped or _rows, key=lambda _r: _r["mean_iou"])
    print(f"\\nkept alpha {BLEND['alpha']:.2f}: the highest val IoU among the "
          f"alphas whose regression rate is at or under {MAX_REGRESSION:.0%}"
          if _capped else
          f"\\nkept alpha {BLEND['alpha']:.2f}: no alpha in the sweep keeps "
          f"the regression rate at or under {MAX_REGRESSION:.0%}, so this is "
          f"the best val IoU of the whole sweep -- add a smaller alpha")
    print("   `worse` counts instances that lost more than 0.05 IoU against "
          "stock, and `when worse` is how much they lost on average. An alpha "
          "below 1 gives most of them back for a fraction of the gain, which "
          "is the trade a deployment usually wants.")
    if BLEND["alpha"] < 1.0:
        CHECKPOINT, TAG = BLEND["path"], f"{TAG}_a{int(round(BLEND['alpha'] * 100)):03d}"
        shutil.copy(CHECKPOINT, Path(MIRROR_DIR) / Path(CHECKPOINT).name)
        print("   everything below now scores the blend, which is what would "
              "ship:", CHECKPOINT)
''')


# --------------------------------------------------------------------------
# 6. The after picture, on the same instances
# --------------------------------------------------------------------------
#
# Split by modality and by source as well as in aggregate: a mixed run's mean
# blends two problems, and the thermal rows are the ones the two arms of this
# experiment can be compared on.

code('''
AFTER = {p: score_to(CHECKPOINT, TAG, p) for p in SCORE_PROMPTS}

print(f"{'prompt':<8}{'':<10}{'mean IoU':>10}{'>=.50':>8}{'>=.75':>8}"
      f"{'small':>10}{'larger':>9}")
for _prompt in SCORE_PROMPTS:
    for _label, _row in ((COMPARE_TAG, BEFORE[_prompt]),
                         ("stage B", AFTER[_prompt])):
        print(f"{_prompt:<8}{_label:<10}{_row['mean_iou']:>10.4f}"
              f"{_row['iou_50']:>8.3f}{_row['iou_75']:>8.3f}"
              f"{_row['small_mean_iou']:>10.4f}{_row['large_mean_iou']:>9.4f}")
    _d = AFTER[_prompt]["mean_iou"] - BEFORE[_prompt]["mean_iou"]
    _s = AFTER[_prompt]["small_mean_iou"] - BEFORE[_prompt]["small_mean_iou"]
    print(f"{_prompt:<8}{'delta':<10}{_d:>+10.4f}{'':>16}{_s:>+10.4f}\\n")

print(f"\\n{'prompt':<8}{'target against its ground':<28}{'inst':>7}"
      f"{'before':>9}{'after':>9}{'delta':>9}")
for _prompt in SCORE_PROMPTS:
    for _band in AFTER[_prompt].get("per_contrast", {}):
        _b = BEFORE[_prompt]["per_contrast"][_band]
        _a = AFTER[_prompt]["per_contrast"][_band]
        print(f"{_prompt:<8}{_band:<28}{_a['instances']:>7}"
              f"{_b['mean_iou']:>9.4f}{_a['mean_iou']:>9.4f}"
              f"{_a['mean_iou'] - _b['mean_iou']:>+9.4f}")
print("   contrast is the target's own signal over the clutter of the ground "
      "beside it (`image_loop.instance_contrast`). The bottom band is the "
      "case a tracker loses -- a parked car on warm concrete, a body at "
      "ambient temperature -- and a test set is mostly the top one, which is "
      "how a model that reads targets off their brightness keeps a good mean.")

for _prompt in SCORE_PROMPTS:
    for _modality in sorted(AFTER[_prompt].get("per_modality", {})):
        _b = BEFORE[_prompt]["per_modality"][_modality]
        _a = AFTER[_prompt]["per_modality"][_modality]
        print(f"{_prompt:<8}{_modality:<10}{_a['instances']:>8} inst  "
              f"mean {_b['mean_iou']:.4f} -> {_a['mean_iou']:.4f} "
              f"({_a['mean_iou'] - _b['mean_iou']:+.4f})  small "
              f"{_b['small_mean_iou']:.4f} -> {_a['small_mean_iou']:.4f} "
              f"({_a['small_mean_iou'] - _b['small_mean_iou']:+.4f})")

import numpy as np
import matplotlib.pyplot as plt

_classes = sorted(set(AFTER[SCORE_PROMPTS[0]]["per_class"])
                  & set(BEFORE[SCORE_PROMPTS[0]]["per_class"]))
_y = np.arange(len(_classes))
_fig, _ax = plt.subplots(figsize=(9.0, 0.42 * len(_classes) + 2.0))
for _i, _prompt in enumerate(SCORE_PROMPTS):
    _offset = (_i - (len(SCORE_PROMPTS) - 1) / 2) * 0.8 / len(SCORE_PROMPTS)
    _delta = [AFTER[_prompt]["per_class"][c]["mean_iou"]
              - BEFORE[_prompt]["per_class"][c]["mean_iou"] for c in _classes]
    _ax.barh(_y + _offset, _delta, height=0.8 / len(SCORE_PROMPTS),
             label=_prompt, color=["#4c72b0", "#dd8452", "#55a868"][_i % 3])
_ax.set_yticks(_y); _ax.set_yticklabels(_classes, fontsize=8)
_ax.axvline(0, color="k", lw=0.8)
_ax.set_xlabel(f"mean IoU after stage B, minus {COMPARE_TAG} (test split)")
_ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
''')


# --------------------------------------------------------------------------
# 7. The cases, before and after
# --------------------------------------------------------------------------
#
# A mean can move for reasons a table cannot show: a model that got better at
# trucks and worse at people reports the average of the two. This scores every
# held-out instance twice, ranks by the change, and draws both ends of the
# ranking -- the gains and the regressions, in one figure, so the second is not
# something a reader has to go looking for.
#
# Scored under the weakest prompt on the list, which is the one 12 found can
# see an encoder change at all.

code('''
from src.training.aerial import image_origin, load_image
from src.training.image_loop import collate, propagate_image
from src.training.loader import batch_clips

PANEL_PROMPT = SCORE_PROMPTS[-1]

def predict(model, samples, batch_size, want_masks=False):
    scored, drawn = [], []
    for chunk in batch_clips(samples, max(batch_size, 1), seed=0, drop_last=False):
        batch = collate(chunk, "cpu").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = propagate_image(model, batch.images, batch.boxes,
                                     batch.valid, PANEL_PROMPT)
        logits = result["pred_masks_high_res"]
        logits = logits[:, 0] if logits.dim() == 4 else logits
        predicted = (logits.float() > 0.0).cpu().numpy()
        targets = batch.masks.reshape(-1, *batch.masks.shape[-2:])
        targets = targets[result["rows"]].cpu().numpy()
        width = batch.boxes.shape[1]
        for _n, _flat in enumerate(result["rows"].cpu().numpy().tolist()):
            _b, _k = divmod(int(_flat), width)
            _union = np.logical_or(predicted[_n], targets[_n]).sum()
            _hit = np.logical_and(predicted[_n], targets[_n]).sum()
            scored.append((chunk[_b], _k, float(_hit / _union) if _union else 1.0))
            if want_masks:
                drawn.append((predicted[_n].copy(), targets[_n].copy()))
        del batch, result, predicted, targets
    torch.cuda.empty_cache()
    return scored, drawn

_shuffled = np.random.default_rng(SEED).permutation(len(TEST.samples))
PANEL_POOL = [TEST.samples[int(i)] for i in _shuffled[:PANEL_WINDOWS]]
_panel_sources = {}
for _sample in PANEL_POOL:
    _name = _sample.source.spec.name if _sample.source else "?"
    _panel_sources[_name] = _panel_sources.get(_name, 0) + 1
print("panel pool:", len(PANEL_POOL), "windows from", len(_panel_sources),
      "sources", dict(sorted(_panel_sources.items(), key=lambda kv: -kv[1])))
_model = build_model(SIZE, COMPARE_CKPT, "cuda")
_before, _ = predict(_model, PANEL_POOL, max(BATCH // 4, 1))
del _model; torch.cuda.empty_cache()
_model = build_model(SIZE, CHECKPOINT, "cuda")
_after, _ = predict(_model, PANEL_POOL, max(BATCH // 4, 1))
del _model; torch.cuda.empty_cache()

CASES = [{"sample": a[0], "k": a[1], "before": b[2], "after": a[2],
          "delta": a[2] - b[2]}
         for a, b in zip(_after, _before) if (a[0], a[1]) == (b[0], b[1])]
CASES.sort(key=lambda c: c["delta"])
_half = min(PANEL_CASES, len(CASES) // 2)
SHOWN = CASES[-_half:][::-1] + CASES[:_half]
print(f"{len(CASES)} held-out instances scored twice under `{PANEL_PROMPT}`: "
      f"{sum(1 for c in CASES if c['delta'] > 0.01)} better, "
      f"{sum(1 for c in CASES if c['delta'] < -0.01)} worse, "
      f"{sum(1 for c in CASES if abs(c['delta']) <= 0.01)} unchanged")
_tally = {}
for _case in CASES:
    _name = (_case["sample"].source.spec.name if _case["sample"].source else "?")
    _cell = _tally.setdefault(_name, [0, 0, 0.0])
    _cell[0 if _case["delta"] > 0.01 else 1] += 1 if abs(_case["delta"]) > 0.01 else 0
    _cell[2] += _case["delta"]
print(f"{'source':<34}{'better':>8}{'worse':>8}{'mean delta':>12}")
for _name, (_up, _down, _sum) in sorted(_tally.items(), key=lambda kv: -kv[1][2]):
    _n = sum(1 for c in CASES
             if (c["sample"].source.spec.name if c["sample"].source else "?") == _name)
    print(f"{_name:<34}{_up:>8}{_down:>8}{_sum / max(_n, 1):>+12.4f}")
assert SHOWN, "no instance was scored by both checkpoints"

PICKED = sorted({id(c["sample"]): c["sample"] for c in SHOWN}.values(),
                key=lambda s: s.frame.name)
MASKS = {}
for _tag, _path in (("before", COMPARE_CKPT), ("after", CHECKPOINT)):
    _model = build_model(SIZE, _path, "cuda")
    _rows, _drawn = predict(_model, PICKED, 1, want_masks=True)
    for (_sample, _k, _), (_mask, _target) in zip(_rows, _drawn):
        MASKS[(_tag, id(_sample), _k)] = _mask
        MASKS[("target", id(_sample), _k)] = _target
    del _model; torch.cuda.empty_cache()

_fig, _axes = plt.subplots(2, _half, figsize=(3.1 * _half, 7.0), squeeze=False)
for _ax, _case in zip(_axes.ravel(), SHOWN):
    _sample, _k = _case["sample"], _case["k"]
    _stock = MASKS[("before", id(_sample), _k)]
    _trained = MASKS[("after", id(_sample), _k)]
    _target = MASKS[("target", id(_sample), _k)]
    _canvas = load_image(_sample.frame.image, image_origin(_sample),
                         _sample.window, _sample.size,
                         _sample.source.gray if _sample.source else True
                         ).astype(np.float32)
    _both = _stock & _trained
    _canvas[_both] = 0.5 * _canvas[_both] + np.array([70, 110, 210])
    _canvas[_stock & ~_trained] = [225, 70, 70]
    _canvas[_trained & ~_stock] = [60, 215, 95]
    _canvas[_target ^ np.roll(_target, 1, axis=0)] = [255, 235, 0]
    _ax.imshow(_canvas.clip(0, 255).astype(np.uint8))
    _box = _sample.boxes[_k]
    _ax.add_patch(plt.Rectangle((_box[0], _box[1]), _box[2] - _box[0],
                                _box[3] - _box[1], fill=False, ec="w", lw=0.8))
    _ax.set_title(f"{_sample.source.spec.name.split('/')[-1]} "
                  f"{_sample.frame.name}\\n{_case['before']:.2f} -> "
                  f"{_case['after']:.2f}  ({_case['delta']:+.2f})", fontsize=8)
    _ax.axis("off")
plt.suptitle(f"top row: stage B gained   |   bottom row: stage B lost   "
             f"(prompt: {PANEL_PROMPT})", y=1.0)
plt.tight_layout(); plt.show()
print("yellow = the target's outline | green = only stage B found it | "
      f"red = only {COMPARE_TAG} found it | blue = both agreed")
''')


# 8. Everything worth keeping, and the verdict in one block
# --------------------------------------------------------------------------

code('''
VERDICT = {
    "run": RUN, "modalities": MODALITIES, "image_size": SIZE,
    "pools": POOL_FLAGS, "datasets": DATASET_FLAGS, "teachers": TEACHERS,
    "harvest": HARVEST,
    "dropped": DROPPED, "unusable": FAILED,
    "frames": {k: len(v) for k, v in SPLITS.items()},
    "train_windows_by_source": TRAIN.sources,
    "test_windows_by_source": TEST.sources,
    "gates": GATES.__dict__, "batch": BATCH, "lr_scale": LR_SCALE,
    "method": METHOD, "anchor_weight": ANCHOR_WEIGHT,
    "rates": {"head": LR_HEAD, "neck": LR_NECK, "trunk": LR_TRUNK,
              "scale": LR_SCALE},
    "class_weights": CLASS_WEIGHTS, "require_pools": REQUIRE_POOLS,
    "min_box_iou": MIN_BOX_IOU, "pool_min_box_iou": POOL_MIN_BOX_IOU,
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT}
            if METHOD == "lora" else {},
    "blend": BLEND, "checkpoint": CHECKPOINT,
    "comparison": {"tag": COMPARE_TAG, "checkpoint": COMPARE_CKPT},
    "roles": {_row["pool"]: _row["role"] for _row in PLAN},
    "modalities": {_row["pool"]: _row["modality"] for _row in PLAN},
    "limits": POOL_LIMITS,
    "prompt": PROMPT, "prompt_jitter": PROMPT_JITTER,
    "photometric": {"collapse": CONTRAST_COLLAPSE, "floor": CONTRAST_FLOOR,
                    "invert": POLARITY_FLIP, "gamma": GAMMA_JITTER,
                    "noise": SENSOR_NOISE},
    "epochs": EPOCHS, "steps": STEPS, "seed": SEED,
    "run_log": RUN_LOG, "before": BEFORE, "after": AFTER,
    "panel_prompt": PANEL_PROMPT,
    "cases": [{"frame": c["sample"].frame.name,
               "source": c["sample"].source.spec.name, "k": c["k"],
               "before": c["before"], "after": c["after"], "delta": c["delta"]}
              for c in SHOWN],
}
(Path(WORK) / "verdict.json").write_text(json.dumps(VERDICT, indent=2) + "\\n")
for _name in ("verdict.json", "run.json"):
    shutil.copy(Path(WORK) / _name, Path(MIRROR_DIR) / _name)
for _file in sorted(Path(WORK).glob("score_*.json")):
    shutil.copy(_file, Path(MIRROR_DIR) / _file.name)

_line = "-" * 74
print(_line)
print("WHAT THIS RUN TRAINED ON")
print(f"  {sum(len(v) for v in SPLITS.values())} frames, "
      f"{len(TRAIN.samples)} train windows, "
      f"{sum(len(s.instances) for s in TRAIN.samples)} instances")
for _source, _count in TRAIN.sources.items():
    print(f"    {_source:<36}{_count:>8} windows")
_how = [f"method {METHOD}"]
if METHOD == "lora":
    _how.append(f"r={LORA_R}")
if ANCHOR_WEIGHT:
    _how.append(f"anchor {ANCHOR_WEIGHT}")
if BLEND:
    _how.append(f"blended at alpha {BLEND['alpha']:.2f}")
print(f"  base {Path(BASE_CKPT).name}, stage A: none, " + ", ".join(_how))
print(_line)
print("DID IT HELP")
print("  comparison baseline:", COMPARE_TAG, "->", COMPARE_CKPT)
for _prompt in SCORE_PROMPTS:
    _d = AFTER[_prompt]["mean_iou"] - BEFORE[_prompt]["mean_iou"]
    _s = AFTER[_prompt]["small_mean_iou"] - BEFORE[_prompt]["small_mean_iou"]
    print(f"  prompt {_prompt:<6} mean IoU {_d:+.4f}   small (<32 px) {_s:+.4f}")
print(f"  {sum(1 for c in CASES if c['delta'] > 0.01)} of {len(CASES)} held-out "
      f"instances improved, {sum(1 for c in CASES if c['delta'] < -0.01)} got worse")
print("  read the `point` row, not the `box` one: an exact box states most of")
print("  the mask on an isolated target, which is why 12 could not tell two")
print("  visibly different encoders apart under it.")
print(_line)
print("WHAT IT DID NOT MEASURE")
print("  Tracking. Every number above is one prompted frame with the memory")
print("  path frozen and never run. A better encoder is a precondition for a")
print("  better tracker, not evidence of one -- that is stage C.")
print(_line)
print("ON YOUR DRIVE:", MIRROR_DIR)
for _file in sorted(Path(MIRROR_DIR).iterdir()):
    print(f"  {_file.name:<40}{round(_file.stat().st_size / 2 ** 20, 1):>9} MiB")
''')


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def render(notebook: str) -> tuple[list[str], str]:
    """One arm's cell texts and the stamp over them.

    The stamp folds in the variant, not just the shared source: two arms built
    from one file share most of their bytes, and a stamp that ignored the
    difference would call a swapped pair correct.
    """
    fields = {"EVAL_DRAWN": '"kust4k"',
              "SOURCE_ZIPS": SOURCE_ZIPS_DEFAULT,
              **ARMS[notebook], "BRANCH": BRANCH, "NOTEBOOK": notebook}
    cells = []
    for text in CELLS:
        for key, value in fields.items():
            text = text.replace("{{" + key + "}}", value)
        cells.append(text)
    return cells, hashlib.sha256("\n".join(cells).encode()).hexdigest()[:10]


def build(notebook: str) -> dict:
    cells, stamp = render(notebook)
    return {
        "cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [],
             "execution_count": None,
             "source": (text.replace("{{STAMP}}", stamp) + "\n")
                       .splitlines(keepends=True)}
            for text in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100",
                      "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    stamps_file = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps_file.read_text()) if stamps_file.is_file() else {}
    for name in ARMS:
        document = build(name)
        (repo / "notebooks" / name).write_text(
            json.dumps(document, indent=1, ensure_ascii=False) + "\n")
        known[name] = render(name)[1]
        print(f"wrote notebooks/{name} with {len(document['cells'])} cells "
              f"[{known[name]}]", file=sys.stderr)
    stamps_file.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
