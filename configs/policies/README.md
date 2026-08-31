# Inference policy overlays

Each file here holds **only** the runtime-policy blocks a backend config can
carry — `samurai:`, `ego_motion:`, `guard:`. `tools/run_records.py --policy`
merges one onto whichever backend YAML the `(--weights, mode)` pair chose and
writes the merged result beside the run as `config.yaml`.

They are overlays rather than whole configs because a policy is **orthogonal**
to the two axes that were already there. `--modes` varies the input, `--weights`
varies the model, and everything in this directory varies neither: no engine
changes shape, no checkpoint is touched, nothing is re-exported. Shipping the
combinations as whole files would be `sizes x weights x policies` copies of the
same engine paths, and the day one of those paths moved most of them would be
quietly stale.

Nothing here needs training. That is the point of the directory: it is the set
of things that can be measured on the weights already on disk.

## Where each one acts

The order the frame is touched in, which is the thing that decides what a
policy *can* fix:

    JPEG -> [prefilter] -> image encoder -> memory attention (reads the bank)
         -> mask decoder -> [memgate] -> memory write -> yielded -> [guard]

`prefilter` is the only one before the encoder. `memgate` is the only one
between the mask and the bank. The guard runs after the frame is already
remembered, which is why it can relabel an output but cannot stop a bad frame
becoming the target's remembered appearance.

## The gate between the mask and the memory bank

`memgate` is the one to read against `plain` when the failure is *the track
jumps to something that looks like the target, and stays there*. Two gates,
both arithmetic on the box, no filter and no re-scoring:

| gate | what it refuses | measured |
|---|---|---|
| `memory_jump: 2.5` | a mask whose centre lands more than 2.5 of the last accepted box's own lengths away | a 90 px jump on a 26 px target: 16 of the 16 following frames enter the bank without it, 8 with it |
| `memory_area_ratio: 3.0` | a mask that stays put and swells past 3x the running median of the areas already accepted | x2, x3, x4 balloons: 10 of 10 frames in without it, 0 of 10 with it |

Both are safe on the honest cases in the same measurements: a real manoeuvre to
16 px a frame keeps all 16, a real 2.6x approach keeps 40 of 40, and
`memory_patience: 8` re-seeds the history after eight refusals in a row so a
target that really did move is not locked out for the rest of the clip.

`kf_weight: 0` means SAM 2's own argmax still picks the mask, so a frame this
gate *accepts* is bit-identical to `plain` — and with nothing left to read the
filter is not run at all (33 us a frame against SAMURAI's 144). It needs no
`--all-pointers` re-export for the same reason.

`memory.json` beside the run is the account of what it did. Read `refused`
first: a gate that refused nothing means the run **is** `plain`.

## The ladder

| policy | adds | costs |
|---|---|---|
| `plain` (default, no file) | — | — |
| `prefilter` | stretches a frame whose used span is under 70 grey levels back to the full range, before the encoder sees it | one lookup per frame |
| `memgate` | the jump and balloon gates above, on the memory write | 33 us a frame |
| `samurai` | motion-aware memory: a Kalman filter re-scores the three candidate masks, and a frame enters the memory bank only if IoU, object score and motion all agree it was a good one | an 8-state filter in numpy, one sync per frame |
| `ego` | `samurai` + the camera's own displacement as a control input to that filter | one reduced-size greyscale decode and one sparse flow per frame (~1 ms) |
| `guard` | `ego` + the classical guard: area, aspect and travel plausibility, hysteresis, and template re-acquisition. A refused mask is reported **empty** | a second decode and a template match on the frames it searches |

Run them in that order and each row of the summary differs from the one above
it by one thing. `plain` vs `guard` alone answers "is the whole stack worth it"
and nothing else.

## Why `ego_motion` never ships without `samurai`

`ego_motion` measures the background's displacement and hands it to
`samurai.KalmanFilter.predict` as a control input — `EdgeTAMTracker._shift_for`
is passed to `samurai.install`, and that is its only consumer. With no
`samurai:` block the measurement is taken and then dropped: a decode and a
sparse flow per frame buying nothing. So `ego.yaml` carries both, and there is
deliberately no ego-only overlay to pick by mistake.

The guard is the other way round: it is independent of SAMURAI and would work
under `plain`. It ships on top of `ego` here so the ladder stays one change per
rung, and because both then share one `FrameMotion` instead of opening two.

## `reduce` is the guard's resolution too

`ego_motion.reduce` sets how small the extra decode is, and the guard reads its
frames from the same `FrameMotion`. At `reduce: 4` a 512-pixel input is judged
on a 128-pixel frame, where a 20-pixel target is 5 pixels across — enough for a
flow field, thin for the template match that re-acquires a lost track. `guard`
therefore drops to `reduce: 2`; `ego`, which only needs the translation, keeps
4. That difference is a deliberate part of the ladder, not an oversight: if
`guard` were run at 4 the rung would be measuring the decode as much as the
policy.
