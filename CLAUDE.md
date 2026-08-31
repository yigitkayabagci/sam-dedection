# sam-dedection — working notes for Claude

## The target domain, and what it is not

**The goal is AERIAL imagery: a camera on a drone looking down.** Thermal
first; RGB is a second model beside it, not the main line.

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
