# sam-dedection — working notes for Claude

## The target domain, and what it is not

**The goal is AERIAL imagery: a camera on a drone looking down.** Thermal
first; RGB is a second model beside it, not the main line. The RGB line is
35 (stage B in all but name) -> 37 (stage C on VTUAV-VIS), and it never shares
a checkpoint, a config or an output folder with the thermal one.

**Do not reach for Anti-UAV410.** It is a ground camera looking *up* at a
drone -- the opposite perspective -- so it answers a different problem, and
proposing it as the data, the evaluation, or the source of `exist` labels has
been wrong every time. Notebook 31 keeps it default-**off** for this reason and
notebooks 29/30 use it only because they were written before this was settled.

When a video-identity source is needed -- `exist` labels, memory-path
training, a real tracking sequence -- use these instead:

| need | source |
|---|---|
| thermal UAV tracking, real disappearances | VTUAV ST/LT (`ir.txt` carries `exist`) |
| aligned RGB-T identities | RGBT234, LasHeR |
| night TIR from a UAV, real track ids | BIRDSAI |
| long-interval instance masks | VTUAV-VIS |
| free negatives | crop windows with no target in them |

## The deployment target: **1280x768** in, **768** into the model

**768 is the size, not a midpoint under evaluation.** The recordings are
1280x768 -- confirmed by the user, not inferred, and not the 720p this file
first said. The short side being exactly 768 is the whole point: `crop768`
takes a **768x768 window in native pixels with no resampling at all**, where a
720-tall source would be clamped to 768x720 and stretched 6.7% vertically. It
is the case `configs/edgetam_768.yaml` was written for and what
`docs/EXPERIMENT_LOG.md` §3.9 means by 768 being for the wide recordings --
there is more than 512x512 of real detail in the frame, so 768's 2304 tokens
are spent on pixels the sensor actually produced. The argument against 768 in
that section is about 640x512 thermal sources, where it upsamples; it does not
apply here. `src/trackers/adaptive.py` is the tool for
those sources, not for this one.

What follows from it:

- **Build, measure and compare at 768.** A 768 run is scored against stock at
  768 (`configs/edgetam_768.yaml` / `configs/edgetam_trt_768.yaml`). A 512
  number is a different measurement and never the baseline for it.
- **768 is the RGB line's size too.** The second model beside the thermal one
  deploys at the same resolution, so notebook 37 trains at `SIZE = 768` rather
  than training at 512 and being run at 768. It can: VTUAV-VIS frames are
  1920x1080, so `_window_for` takes a **native 768 crop with no resampling**.
  That is exactly what the thermal sources cannot do -- most of them are
  640x512 -- which is why 31 and 32 are still 512 and this is not. The
  asymmetry is about the data, not about a preference.
- **`models768*/` is the product**; a 512 engine set is a comparison point.
- **Stage B checkpoints so far were trained at 512** and run at 768 with no
  error -- EdgeTAM keeps no resolution in any parameter, so the mismatch is a
  printed warning from `src/checkpoint_meta.py` and never a crash. That is a
  real cost and an unmeasured one. A stage B run meant for deployment should
  train at `--size 768` (notebook 32's `SIZE`), which removes the variable;
  whether it pays is still a measurement, taken against the same weights at 512.
- **Engines are not portable.** They are built for one GPU architecture and one
  TensorRT version, so they are built on the machine that will run them. A
  desktop build is for measuring, not for shipping to the Orin.

## Ground rules this repo already holds itself to

- **Measure before claiming.** Every threshold in `docs/` names the number it
  came from. "HIT-UAV targets are hot against cold ground" was assumed, then
  measured at a median signal-to-clutter ratio of **0.91**, and the docstrings
  were corrected. Do the same rather than reasoning from what sounds right.
- **A gate that fires on a real target is worse than no gate.** `stabiliser.py`
  refuses the impossible, not the unusual.
- Generated notebooks are **comment-free and capped at 9 cells**; anything
  worth saying gets printed, not commented. They come from `tools/build_*.py`
  -- edit the builder, never the `.ipynb`.
- `22_thermal_deep_3_fixed.ipynb` at the repo root is the user's own preserved
  file and the template notebook 32 is generated from. Do not regenerate it.
- Turkish is the working language for `docs/*_tr.md` and for replies.

## Where things are

| | |
|---|---|
| run plan, notebook order | `docs/termal_yol_haritasi_tr.md` |
| what changed and why | `docs/son_degisiklikler_tr.md` |
| how to verify each change | `docs/calisma_plani_tr.md` |
