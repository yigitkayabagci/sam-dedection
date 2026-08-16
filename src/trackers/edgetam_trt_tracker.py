"""EdgeTAM with its whole per-frame graph running on TensorRT.

Four engines cover everything that executes on every tracked frame:

    image_encoder     RepViT trunk + FPN neck (+ the mask decoder's two
                      high-res 1x1 projections, folded in at export time)
    memory_attention  2 layers of self + cross attention over the memory bank
    memory_encoder    memory encoder + 2D spatial Perceiver
    sam_head          SAM mask decoder on the propagation path

What stays in PyTorch, deliberately:

  * Frames that carry a user prompt. `add_new_points_or_box` runs the SAM head
    with real clicks and a variable point count -- a different graph, executed
    once or twice per video. Not worth an engine.
  * The memory bank bookkeeping in `_prepare_memory_conditioned_features`:
    which past frames to attend to, temporal position encodings, object
    pointer collection. Pure Python over small tensors.
  * The sigmoid/scale/bias applied to the mask before memory encoding, and the
    output mask resize. Single cheap kernels.

Each engine is optional and independently replaceable. A missing file, a
tensorrt import error, or an object count outside the engine's optimisation
profile all fall back to stock PyTorch for that module -- so a partial build
still runs, just slower. `strict=True` turns those into errors instead.

Integration is by patching methods on the built predictor rather than by
subclassing SAM2VideoPredictor: EdgeTAM's `track_step` is one long method and
we want to replace four leaves of it, not fork it.
"""
from __future__ import annotations

from pathlib import Path

from ._trt_layout import MASKED_LOGIT, MemoryLayout
from .edgetam_tracker import EdgeTAMTracker
from .registry import register

# Engine name -> the config key that points at its .engine file.
ENGINE_KEYS = {
    "image_encoder": "image_encoder_engine",
    "memory_attention": "memory_attention_engine",
    "memory_encoder": "memory_encoder_engine",
    "sam_head": "sam_head_engine",
}


def _connected_components_available() -> bool:
    try:
        from sam2 import _C  # noqa: F401
    except Exception:
        return False
    return True


def _to_bchw(seq_first, height: int, width: int):
    """[HW, B, C] token view -> [B, C, H, W], without copying.

    Upstream's `current_vision_feats` are `x.flatten(2).permute(2, 0, 1)` views
    of a BCHW tensor, so permuting back and splitting the (stride-1) token axis
    lands on the original layout. This holds for the broadcast views
    multi-object tracking produces too, where dim 0 has stride 0.
    """
    return seq_first.permute(1, 2, 0).unflatten(-1, (height, width))


@register("edgetam_trt")
class EdgeTAMTRTTracker(EdgeTAMTracker):
    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        image_encoder_engine: str | None = None,
        memory_attention_engine: str | None = None,
        memory_encoder_engine: str | None = None,
        sam_head_engine: str | None = None,
        engine_path: str | None = None,
        device: str = "cuda",
        precision: str = "float16",
        mask_threshold: float = 0.0,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        strict: bool = False,
        use_cuda_graph: bool = True,
        fill_hole_area: int | None = None,
        image_size: int | None = None,
        calibration_dir: str | None = None,
        calibration_limit: int = 512,
        calibration_stride: int = 1,
        samurai: dict | None = None,
        sam2long: dict | None = None,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            mask_threshold=mask_threshold,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            image_size=image_size,
            samurai=samurai,
            sam2long=sam2long,
        )
        self.engine_paths = {
            # `engine_path` is the pre-multi-engine config key for the image encoder.
            "image_encoder": image_encoder_engine or engine_path,
            "memory_attention": memory_attention_engine,
            "memory_encoder": memory_encoder_engine,
            "sam_head": sam_head_engine,
        }
        self.strict = strict
        self.use_cuda_graph = use_cuda_graph
        self.fill_hole_area = fill_hole_area
        # Set to a directory to record what every engine is fed, for INT8
        # calibration. Three of the four take internal activations whose
        # distributions cannot be guessed, so they have to come from a real run.
        self.calibration_dir = calibration_dir
        self.calibration_limit = calibration_limit
        self.calibration_stride = calibration_stride
        self._calibration: dict[str, object] = {}
        self._engines: dict[str, object] = {}
        self._layout: MemoryLayout | None = None
        self._patched = False
        self._warned: set[str] = set()

    # ------------------------------------------------------------ loading

    def _warn(self, key: str, message: str) -> None:
        """Print `message` once per distinct `key` (these fire per frame)."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[edgetam_trt] {message}")

    def _load_engines(self) -> None:
        if self._engines:
            return
        try:
            from ._trt_runtime import TRTEngine
        except ImportError as exc:
            if self.strict:
                raise
            print(f"[edgetam_trt] tensorrt unavailable ({exc}); running on PyTorch.")
            return

        for name, path in self.engine_paths.items():
            if not path:
                if self.strict:
                    raise ValueError(f"strict=True but no engine configured for {name}.")
                continue
            if not Path(path).exists():
                if self.strict:
                    raise FileNotFoundError(f"{name} engine not found: {path}")
                print(f"[edgetam_trt] {name}: no engine at {path}; using PyTorch.")
                continue
            try:
                self._engines[name] = TRTEngine(
                    path, name=name, use_cuda_graph=self.use_cuda_graph
                )
            except Exception as exc:
                if self.strict:
                    raise
                print(f"[edgetam_trt] {name}: engine load failed ({exc}); using PyTorch.")

    def _start_calibration(self) -> None:
        """Wrap the loaded engines so every input they see is kept.

        Wrapping happens before patching, so the tracker's own packing code --
        padded memory slots, the additive mask, the fp16 buffers -- is what
        produces the recorded tensors. Calibrating on anything reconstructed
        afterwards would be calibrating a different network.
        """
        if not self.calibration_dir or not self._engines:
            return
        from .calibration import wrap_engines

        self._calibration = wrap_engines(
            self._engines, self.calibration_limit, self.calibration_stride
        )
        print(f"[edgetam_trt] recording calibration data for "
              f"{', '.join(sorted(self._calibration))} -> {self.calibration_dir}")

    def save_calibration(self) -> list[Path]:
        """Write one `.npz` per engine. Call after tracking, before `reset`."""
        written = []
        for name, store in self._calibration.items():
            written.append(store.save(Path(self.calibration_dir) / f"{name}.npz"))
            print(f"[edgetam_trt] {name}: {store.summary()}")
        return written

    # ----------------------------------------------------------- patching

    def _patch(self) -> None:
        if self._patched:
            return
        for name, patch in (
            ("image_encoder", self._patch_image_encoder),
            ("memory_attention", self._patch_memory_attention),
            ("memory_encoder", self._patch_memory_encoder),
            ("sam_head", self._patch_sam_head),
        ):
            if name in self._engines:
                patch(self._engines[name])
        accelerated = sorted(self._engines)
        print(
            "[edgetam_trt] TensorRT: "
            + (", ".join(accelerated) if accelerated else "none")
            + (f" | CUDA graphs: {'on' if self.use_cuda_graph else 'off'}")
        )
        self._patched = True

    def _patch_image_encoder(self, engine) -> None:
        predictor = self._predictor
        encoder = predictor.image_encoder
        neck_pos = encoder.neck.position_encoding
        names = ("backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2")
        missing = [n for n in names if n not in engine.output_names]
        if missing:
            raise RuntimeError(
                f"image_encoder engine is missing output(s) {missing}; it was built "
                "from a different ONNX graph. Re-export with "
                "tools/export_edgetam_onnx.py."
            )

        # A fused engine has already applied the mask decoder's conv_s0/conv_s1,
        # which is visible in the channel count of the highest-resolution level.
        channels = engine.engine_shape(names[0])[1]
        if channels == predictor.hidden_dim // 8:
            fused = True
        elif channels == predictor.hidden_dim:
            fused = False
        else:
            raise RuntimeError(
                f"image_encoder engine emits {channels} channels on {names[0]}; "
                f"expected {predictor.hidden_dim // 8} (conv_s0 fused) or "
                f"{predictor.hidden_dim} (not fused). The engine does not match "
                "this model -- re-export with tools/export_edgetam_onnx.py."
            )
        # Positional encodings are a pure function of (H, W): compute once and
        # keep. At 1024x1024 that saves regenerating ~44 MB of fp16 sine tables
        # per frame, of which EdgeTAM only ever reads the smallest level.
        pos_cache: dict[tuple, list] = {}

        def forward(image):
            # Outputs are read within this frame only -- `cached_features`
            # holds a single frame and is refilled before it is read again --
            # so hand back the engine's buffers rather than copies.
            out = engine.run({"image": image}, clone=())
            feats = [out[n] for n in names]
            key = tuple(tuple(f.shape) for f in feats)
            pos = pos_cache.get(key)
            if pos is None:
                pos = [neck_pos(f).to(f.dtype) for f in feats]
                pos_cache[key] = pos
            return {
                "vision_features": feats[-1],
                "vision_pos_enc": pos,
                "backbone_fpn": feats,
            }

        encoder.forward = forward
        if fused:
            # Stop `forward_image` re-running the projections the engine did.
            predictor.forward_image = lambda img_batch: predictor.image_encoder(img_batch)

    def _patch_memory_attention(self, engine) -> None:
        predictor = self._predictor
        module = predictor.memory_attention
        fallback = module.forward

        total_tokens = engine.engine_shape("memory")[1]
        layout = MemoryLayout.from_model_and_engine(predictor, total_tokens)
        self._layout = layout
        side = predictor.image_size // predictor.backbone_stride
        dim, mem_dim = predictor.hidden_dim, predictor.mem_dim
        # Per-binding memo of the last mask written, so the mask is rebuilt only
        # while the memory bank is still filling up (~7 frames) and never after.
        mask_state: dict[object, tuple] = {}

        def forward(
            curr,
            memory,
            curr_pos=None,
            memory_pos=None,
            num_obj_ptr_tokens: int = 0,
            num_spatial_mem: int = -1,
        ):
            if isinstance(curr, list):
                curr, curr_pos = curr[0], curr_pos[0]
            tokens, batch, _ = curr.shape
            live_spatial = num_spatial_mem * layout.tokens_per_slot

            reason = None
            if not 0 <= num_spatial_mem <= layout.num_slots:
                reason = (
                    f"num_spatial_mem={num_spatial_mem} is outside the engine's "
                    f"0..{layout.num_slots} slots"
                )
            elif num_obj_ptr_tokens > layout.ptr_tokens:
                reason = (
                    f"{num_obj_ptr_tokens} object-pointer tokens > "
                    f"{layout.ptr_tokens} reserved (re-export with a larger "
                    "--max-ptr-tokens if this is your steady state)"
                )
            elif memory.shape[0] != live_spatial + num_obj_ptr_tokens:
                reason = (
                    f"memory length {memory.shape[0]} does not match "
                    f"{num_spatial_mem} slots + {num_obj_ptr_tokens} pointer tokens"
                )
            elif tokens != side * side or memory.shape[-1] != mem_dim:
                reason = f"unexpected shapes curr={tuple(curr.shape)} memory={tuple(memory.shape)}"

            shapes = {
                "curr": (batch, dim, side, side),
                "curr_pos": (batch, dim, side, side),
                "memory": (batch, layout.total_tokens, mem_dim),
                "memory_pos": (batch, layout.total_tokens, mem_dim),
                "memory_mask": (batch, 1, 1, layout.total_tokens),
            }
            if reason is None and not engine.accepts(shapes):
                reason = (
                    f"batch {batch} is outside the engine's profile "
                    "(rebuild with --max-batch >= object count)"
                )
            if reason is not None:
                if self.strict:
                    raise RuntimeError(f"memory_attention: {reason}")
                self._warn(f"mem_attn:{reason}", f"memory_attention on PyTorch: {reason}")
                return fallback(
                    curr=curr,
                    curr_pos=curr_pos,
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                    num_spatial_mem=num_spatial_mem,
                )

            binding = engine.binding(shapes)
            ins = binding.inputs
            ins["curr"].copy_(_to_bchw(curr, side, side), non_blocking=True)
            # curr_pos is the same sine table every frame; re-copying it costs
            # ~2 MB and removes any need to reason about who owns it.
            ins["curr_pos"].copy_(_to_bchw(curr_pos, side, side), non_blocking=True)

            ptr_end = layout.spatial_tokens + num_obj_ptr_tokens
            for buf, src in (
                (ins["memory"], memory),
                (ins["memory_pos"], memory_pos),
            ):
                buf[:, :live_spatial].copy_(src[:live_spatial].transpose(0, 1), non_blocking=True)
                if num_obj_ptr_tokens:
                    buf[:, layout.spatial_tokens : ptr_end].copy_(
                        src[live_spatial:].transpose(0, 1), non_blocking=True
                    )
                # Masked positions contribute zero weight, but keeping their
                # values at zero means a stray inf/NaN can never turn "weight 0"
                # into "0 * inf".
                buf[:, live_spatial : layout.spatial_tokens].zero_()
                buf[:, ptr_end:].zero_()

            mask_key = (num_spatial_mem, num_obj_ptr_tokens)
            if mask_state.get(binding) != mask_key:
                mask = ins["memory_mask"]
                mask.fill_(MASKED_LOGIT)
                mask[..., :live_spatial] = 0.0
                if num_obj_ptr_tokens:
                    mask[..., layout.spatial_tokens : ptr_end] = 0.0
                mask_state[binding] = mask_key

            binding.run()
            # The caller immediately does `.permute(1, 2, 0).view(B, C, H, W)`,
            # which turns this straight back into the buffer's own layout.
            return binding.outputs["pix_feat"].flatten(2).permute(2, 0, 1)

        module.forward = forward

    def _patch_memory_encoder(self, engine) -> None:
        import torch

        predictor = self._predictor
        memory_encoder = predictor.memory_encoder
        perceiver = predictor.spatial_perceiver
        encoder_fallback = memory_encoder.forward
        perceiver_fallback = perceiver.forward

        def forward(pix_feat, masks, skip_mask_sigmoid: bool = False):
            if not skip_mask_sigmoid:
                masks = torch.sigmoid(masks)
            shapes = {"pix_feat": tuple(pix_feat.shape), "mask_for_mem": tuple(masks.shape)}
            if not engine.accepts(shapes):
                reason = f"shapes {shapes} outside the engine's profile"
                if self.strict:
                    raise RuntimeError(f"memory_encoder: {reason}")
                self._warn(f"mem_enc:{reason}", f"memory_encoder on PyTorch: {reason}")
                out = encoder_fallback(pix_feat, masks, skip_mask_sigmoid=True)
                feats, pos = perceiver_fallback(
                    out["vision_features"], out["vision_pos_enc"][0]
                )
                return {"vision_features": feats, "vision_pos_enc": [pos]}

            # Both outputs are stored in the inference state and read on later
            # frames, so they must be copies rather than buffer views.
            out = engine.run({"pix_feat": pix_feat, "mask_for_mem": masks})
            return {"vision_features": out["maskmem"], "vision_pos_enc": [out["maskmem_pos"]]}

        memory_encoder.forward = forward
        # The engine already ran the Perceiver; `_encode_new_memory` calls it
        # next, so make that a no-op. The fallback above runs it itself.
        perceiver.forward = lambda x, pos=None: (x, pos)

    def _reselect(self, engine, out):
        """Let SAMURAI override the engine's own `argmax(ious)` mask choice.

        The engine picks a candidate internally and derives the mask, the
        high-res mask and the object pointer from it. Overriding the choice on
        the Python side means all three have to be re-derived from the chosen
        candidate -- especially the pointer, which is what gets written to the
        memory bank and therefore has to describe the mask that was kept.

        Needs an engine exported with `--all-pointers`. Without it the choice
        cannot be made consistently, so it is not made at all: the memory gate
        (the half that stops a bad frame being remembered) still applies.
        """
        import torch
        import torch.nn.functional as F

        if "obj_ptr_all" not in engine.output_names:
            self._warn("samurai:no_ptrs",
                       "sam_head engine has no obj_ptr_all output, so SAMURAI's "
                       "mask re-selection is off (its memory gate still applies). "
                       "Re-export with tools/export_edgetam_onnx.py --all-pointers.")
            return out

        scores = self._samurai.rescore(
            out["low_res_multimasks"], out["ious"], out["object_score_logits"]
        )
        best = scores.argmax(dim=-1)
        if bool((best == out["ious"].argmax(dim=-1)).all()):
            return out  # same candidate the engine chose; nothing to redo

        masks = out["low_res_multimasks"]
        index = best.reshape(-1, 1, 1, 1).expand(-1, 1, *masks.shape[-2:])
        low_res = torch.gather(masks, 1, index).clone()
        return {
            **out,
            "ious": scores,
            "low_res_masks": low_res,
            # One bilinear upsample of the chosen candidate, ~0.2 ms at 512.
            # Cheaper than making the engine emit all three at full size.
            "high_res_masks": F.interpolate(
                low_res, size=(self._predictor.image_size,) * 2,
                mode="bilinear", align_corners=False),
            "obj_ptr": torch.gather(
                out["obj_ptr_all"], 1,
                best.reshape(-1, 1, 1).expand(-1, 1, out["obj_ptr_all"].shape[-1]),
            )[:, 0].clone(),
        }

    def _patch_sam_head(self, engine) -> None:
        predictor = self._predictor
        fallback = predictor._forward_sam_heads

        def forward_sam_heads(
            backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
        ):
            def run_fallback():
                return fallback(
                    backbone_features=backbone_features,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )

            # The engine bakes in "no prompt, multimask" -- the propagation
            # path. Prompted frames go to PyTorch.
            if (
                point_inputs is not None
                or mask_inputs is not None
                or not multimask_output
                or high_res_features is None
                or len(high_res_features) != 2
            ):
                return run_fallback()

            shapes = {
                "pix_feat": tuple(backbone_features.shape),
                "high_res_0": tuple(high_res_features[0].shape),
                "high_res_1": tuple(high_res_features[1].shape),
            }
            if not engine.accepts(shapes):
                reason = f"shapes {shapes} outside the engine's profile"
                if self.strict:
                    raise RuntimeError(f"sam_head: {reason}")
                self._warn(f"sam_head:{reason}", f"sam_head on PyTorch: {reason}")
                return run_fallback()

            out = engine.run(
                {
                    "pix_feat": backbone_features,
                    "high_res_0": high_res_features[0],
                    "high_res_1": high_res_features[1],
                },
                # These three outlive the frame: they go into `output_dict` and
                # are read again when later frames build their memory bank.
                clone=("low_res_masks", "obj_ptr", "object_score_logits"),
            )
            if self._samurai is not None:
                out = self._reselect(engine, out)
            # Position 1 is `high_res_multimasks`. `track_step` and
            # `_use_mask_as_output` are the only callers and both discard it,
            # so materialising [B, 3, 1024, 1024] would be pure waste. A future
            # caller that wants it gets a TypeError, not a wrong answer.
            return (
                out["low_res_multimasks"],
                None,
                out["ious"],
                out["low_res_masks"],
                out["high_res_masks"],
                out["obj_ptr"],
                out["object_score_logits"],
            )

        predictor._forward_sam_heads = forward_sam_heads

    # ------------------------------------------------------------- policy

    def _apply_fill_hole_policy(self) -> None:
        """Decide whether to keep EdgeTAM's small-hole post-processing.

        `fill_holes_in_mask_scores` needs the compiled `sam2._C` extension. When
        it is missing, upstream catches the ImportError *per frame* and warns --
        the holes go unfilled either way, so the only thing the attempt buys is
        an exception and a warning on every frame.
        """
        predictor = self._predictor
        if self.fill_hole_area is not None:
            predictor.fill_hole_area = int(self.fill_hole_area)
            return
        if predictor.fill_hole_area > 0 and not _connected_components_available():
            print(
                "[edgetam_trt] sam2._C is not built; disabling fill_hole_area "
                "(it would raise and be skipped on every frame anyway)."
            )
            predictor.fill_hole_area = 0

    # ----------------------------------------------------------- lifecycle

    def prepare(self, frames_dir: str | Path) -> None:
        self._ensure_predictor()
        self._apply_fill_hole_policy()
        self._load_engines()
        self._start_calibration()
        # Before patching: `install` wraps the mask decoder, which the sam_head
        # engine replaces. When that engine is present `_reselect` applies the
        # same policy at the engine's boundary instead; when it is not, the
        # decoder wrapper is what runs. Either way the memory gate is the same
        # patch, because the memory bookkeeping never left PyTorch.
        self._install_samurai()
        self._patch()
        with self._inference_ctx():
            self._state = self._predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )

    # set_prompts / propagate / reset inherited from EdgeTAMTracker.
