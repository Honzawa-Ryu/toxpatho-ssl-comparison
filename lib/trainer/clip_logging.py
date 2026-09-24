"""Read-only clipping telemetry; no model/optimizer dependencies."""
import json
import math
import time
from pathlib import Path


def step_record(norms, threshold, epoch, step, global_norm):
    """Norms are per parameter, before clipping AND last-layer cancellation.

    Coefficients reproduce the clipping formula in Python precision. Nonfinite
    steps are explicitly invalid, never silently counted as unclipped.
    """
    valid = math.isfinite(global_norm) and all(math.isfinite(n) for n in norms)
    row = dict(epoch=epoch, step=step, pre_clip_norm=global_norm if valid else None,
               threshold=threshold, n_params=len(norms), finite=valid)
    if not valid:
        return row
    coefs = [min(1.0, threshold / (n + 1e-6)) if threshold > 0 else 1.0
             for n in norms]
    row.update(
        n_above_threshold=sum(n > threshold for n in norms) if threshold > 0 else 0,
        n_clipped=sum(c < 1.0 for c in coefs),
        coef_min=min(coefs, default=1.0),
        coef_mean=sum(coefs) / len(coefs) if coefs else 1.0,
        post_clip_norm=math.sqrt(sum((n * c) ** 2 for n, c in zip(norms, coefs))),
    )
    return row


def write_epoch(directory, epoch, threshold, norms_by_step, global_norms):
    """One device-to-host transfer per completed epoch, rank 0 caller only.

    Atomic publication prevents partial epochs being used by the analysis.
    Timestamp suffix separates resumed attempts; analysis selects the latest.
    """
    import torch

    sizes = [len(norms) for norms in norms_by_step]
    nonempty = [norms for norms in norms_by_step if len(norms)]
    values = torch.cat(nonempty).cpu().tolist() if nonempty else []
    path = Path(directory) / f"clip_steps_ep{epoch:04d}_{time.time_ns()}.jsonl"
    temporary = path.with_suffix(".tmp")
    offset = 0
    with temporary.open("w") as handle:
        for step, (size, norm) in enumerate(zip(sizes, global_norms)):
            row = step_record(values[offset:offset + size], threshold, epoch, step, norm)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            offset += size
    temporary.replace(path)
