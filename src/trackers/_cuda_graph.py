"""Capture a module's forward into a CUDA graph and replay it.

Why
    On Orin the CPU cannot issue kernels as fast as the GPU retires them.
    Measured on the image encoder at 1024: 29.6 ms of GPU work needed ~33 ms
    of CPU launch time, so the GPU idled between kernels. A captured graph
    replays the whole sequence from a single launch, which closes that gap
    without changing a single kernel -- the output is bit-identical.

    The same argument applies to every block in EdgeTAM's per-frame step, and
    the measurement says it applies hardest to the small ones. At image_size
    512 the TensorRT encoder runs in 4.70 ms while the mask decoder needs
    10.97 ms for a fraction of the arithmetic. That gap is launch overhead,
    not compute, and this is the tool for it.

What it handles
    Arbitrary nested args/kwargs, one graph per distinct input signature, and
    a fall back to eager for anything it cannot capture. A slower tracker
    beats a broken one.

What it deliberately does not handle
    Modules whose forward has host-side control flow driven by tensor values
    (anything that calls .item() or .cpu() mid-forward). Capture raises there
    and the wrapper drops to eager for the rest of the run.
"""
from __future__ import annotations

# One graph is normally enough: after the memory bank saturates every frame
# hits the same signature. The second slot exists for pipelines that alternate
# between two stable shapes. Each captured graph pins its own input buffers,
# output buffers and private memory pool, so this is a memory knob.
MAX_GRAPHS = 2
# Capture a signature only once it has been seen this many times. EdgeTAM's
# memory bank grows for the first few frames and each of those shapes appears
# exactly once; capturing them would spend the budget before the steady-state
# shape ever shows up.
MIN_HITS = 2
WARMUP_CALLS = 3


def flatten(obj, tensors, torch):
    """Split a nested arg structure into a flat tensor list plus a rebuild spec.

    Tensors become ("t", index) so the same structure can later be rebuilt
    around a *different* set of tensors, which is how the static capture
    buffers get substituted for the caller's live ones.
    """
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return ("t", len(tensors) - 1)
    if isinstance(obj, list):
        return ("l", [flatten(o, tensors, torch) for o in obj])
    if isinstance(obj, tuple):
        return ("p", [flatten(o, tensors, torch) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, flatten(v, tensors, torch)) for k, v in obj.items()])
    return ("c", obj)


def rebuild(spec, tensors):
    kind, payload = spec
    if kind == "t":
        return tensors[payload]
    if kind == "l":
        return [rebuild(s, tensors) for s in payload]
    if kind == "p":
        return tuple(rebuild(s, tensors) for s in payload)
    if kind == "d":
        return {k: rebuild(s, tensors) for k, s in payload}
    return payload


def signature(spec_args, spec_kwargs, tensors) -> tuple:
    """A hashable key that changes whenever the graph would have to change.

    Shapes and dtypes obviously matter. So do the non-tensor arguments: a
    bool like `multimask_output` selects a different code path, and a graph
    captured under one value is simply wrong under the other.
    """
    meta = tuple((tuple(t.shape), str(t.dtype), str(t.device)) for t in tensors)
    return (repr(spec_args), repr(spec_kwargs), meta)


def clone_outputs(out, torch):
    """Deep-copy the graph's output buffers before handing them to the caller.

    The graph writes into the same memory every replay, and EdgeTAM holds a
    frame's features while the next frame is computed, so returning the
    buffers themselves would silently alias frames together.

    Tensors that were the *same object* in the captured output stay the same
    object here (SAM2 relies on vision_features being backbone_fpn[-1]), and
    it avoids cloning a large tensor twice.
    """
    seen: dict[int, object] = {}

    def walk(o):
        if isinstance(o, torch.Tensor):
            if id(o) not in seen:
                seen[id(o)] = o.clone()
            return seen[id(o)]
        if isinstance(o, tuple) and hasattr(o, "_fields"):     # namedtuple
            return type(o)(*(walk(x) for x in o))
        if isinstance(o, list):
            return [walk(x) for x in o]
        if isinstance(o, tuple):
            return tuple(walk(x) for x in o)
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        return o

    return walk(out)


def graph_module(module, label, autocast_factory, max_graphs: int = MAX_GRAPHS,
                 min_hits: int = MIN_HITS, warmup: int = WARMUP_CALLS,
                 state: dict | None = None):
    """Replace `module.forward` with a lazily-captured, replayed CUDA graph.

    `autocast_factory` must return a *fresh* autocast context with weight
    caching disabled -- see EdgeTAMTracker._uncached_autocast for why that is
    not optional.

    Capture is lazy on purpose: it happens on a real call, so the graph sees
    exactly the shapes and the autocast state the pipeline actually uses.
    """
    import torch

    original = module.forward
    graphs: dict = {}
    hits: dict = {}
    st = state if state is not None else {}
    st.setdefault("failed", False)
    st.setdefault("captured", 0)

    def capture(sig, tensors, spec_args, spec_kwargs):
        static = [t.clone() for t in tensors]
        # Warm up on a side stream first: capture must not race with lazy
        # cuDNN/cuBLAS init or an allocator first-touch, either of which would
        # otherwise get recorded into the graph.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), autocast_factory():
            for _ in range(warmup):
                original(*rebuild(spec_args, static), **rebuild(spec_kwargs, static))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with autocast_factory(), torch.cuda.graph(graph):
            out = original(*rebuild(spec_args, static), **rebuild(spec_kwargs, static))
        torch.cuda.synchronize()
        graphs[sig] = (graph, static, out)
        st["captured"] += 1
        print(f"[edgetam] {label}: CUDA graph captured "
              f"({st['captured']}/{max_graphs} signatures)")

    def graphed(*args, **kwargs):
        if st["failed"]:
            return original(*args, **kwargs)
        tensors: list = []
        spec_args = flatten(list(args), tensors, torch)
        spec_kwargs = flatten(kwargs, tensors, torch)
        if not tensors or any(t.device.type != "cuda" for t in tensors):
            return original(*args, **kwargs)

        sig = signature(spec_args, spec_kwargs, tensors)
        entry = graphs.get(sig)
        if entry is None:
            hits[sig] = hits.get(sig, 0) + 1
            if hits[sig] < min_hits or st["captured"] >= max_graphs:
                return original(*args, **kwargs)
            try:
                capture(sig, tensors, spec_args, spec_kwargs)
            except Exception as exc:
                st["failed"] = True
                print(f"[edgetam] {label}: CUDA graph capture failed ({exc}); "
                      "running it eagerly for the rest of the run.")
                return original(*args, **kwargs)
            entry = graphs[sig]

        graph, static, out = entry
        for dst, src in zip(static, tensors):
            dst.copy_(src)
        graph.replay()
        return clone_outputs(out, torch)

    module.forward = graphed          # type: ignore[assignment]
    return st
