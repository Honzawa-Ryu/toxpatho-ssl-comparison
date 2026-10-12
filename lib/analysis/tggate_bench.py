"""DINO 0028 checkpoints on the tggate-bench severity benchmark.

tggate-bench is a separate repository under revision, so it is only read.
Its experiments run unmodified from an overlay root: `lib/` and the label
table are symlinks into tggate-bench, and the DINO embeddings are written
there in the layout of its Exp 0010, so Exp 0011 / 0002 resolve them like any
other encoder. All results stay under the overlay's `outputs/`.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

LABELS_EXP = "0008_20261009_select_benchmark_slides"
TILES_EXP = "0009_20261009_extract_tiles"
EMB_EXP = "0010_20261009_extract_embeddings"
ANOMALY_EXP = "0011_20261011_eval_severity_anomaly"
POOLED_EXP = "0002_20261007_eval_severity_pooled"
SUMMARY_EXP = "0005_20261007_summarize_report"
POOLERS = ("mean", "max", "topk")
BASELINES = ("uni", "conch", "virchow2", "h-optimus-0", "resnet50-imagenet", "imgstats")


def model_key(epoch):
    return f"dino0028-ep{epoch:03d}"


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_overlay(bench_root, overlay, dataset):
    """Symlink tggate-bench's lib/ and label table into the overlay root."""
    bench_root, overlay = Path(bench_root).resolve(), Path(overlay)
    links = {overlay / "lib": bench_root / "lib",
             overlay / "outputs" / LABELS_EXP / dataset: bench_root / "outputs" / LABELS_EXP / dataset}
    for link, target in links.items():
        if not target.exists():
            raise FileNotFoundError(target)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"{link} exists and does not point to {target}")
        link.symlink_to(target)
    return overlay / "outputs" / LABELS_EXP / dataset / "labels.parquet"


def tile_transform():
    """Same preprocessing as tggate-bench lib/encoders._imagenet_transform
    (UNI / ResNet50): 256 px tiles -> 224 px, ImageNet mean/std."""
    from torchvision import transforms

    return transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


def _write_npy(path, array):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def extract(bench_root, overlay, dataset, checkpoint_dir, epochs, batch_size=256, workers=6,
            device_name="cuda", precision="bf16"):
    """Embed every benchmark tile with all `epochs` at once (each slide's tiles
    are read once). Per-slide shards make the run resumable; the per-model
    .npz is assembled at the end with the keys of tggate-bench Exp 0010."""
    import torch
    import timm
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from lib.analysis.dino_epoch_analysis import load_student_backbone

    bench_root, overlay = Path(bench_root), Path(overlay)
    labels_path = prepare_overlay(bench_root, overlay, dataset)
    slide_ids = pd.read_parquet(labels_path)["slide_id"].astype(str).to_numpy()
    tile_files = {p.stem: p for p in (bench_root / "outputs" / TILES_EXP).glob(f"{dataset}__chunk*/*.npy")}
    missing = [s for s in slide_ids if s not in tile_files]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slides have no tiles, e.g. {missing[:5]}")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoints = {e: Path(checkpoint_dir) / f"model_ep{e}.pt" for e in epochs}
    run_dirs = {e: overlay / "outputs" / EMB_EXP / f"{dataset}__{model_key(e)}" for e in epochs}
    for e, ckpt in checkpoints.items():
        if not ckpt.is_file() or ckpt.stat().st_size == 0:
            raise FileNotFoundError(ckpt)
        (run_dirs[e] / "_shards").mkdir(parents=True, exist_ok=True)
    models = {e: load_student_backbone(ckpt).to(device).eval() for e, ckpt in checkpoints.items()}

    transform = tile_transform()

    class Tiles(Dataset):
        def __init__(self, arr):
            self.arr = arr

        def __len__(self):
            return len(self.arr)

        def __getitem__(self, i):
            return transform(Image.fromarray(self.arr[i]))

    t0, n_new = time.time(), 0
    for i, sid in enumerate(slide_ids, 1):
        shards = {e: run_dirs[e] / "_shards" / f"{sid}.npy" for e in epochs}
        todo = [e for e in epochs if not shards[e].exists()]
        if not todo:
            continue
        tiles = np.load(tile_files[sid])
        out = {e: [] for e in todo}
        loader = DataLoader(Tiles(tiles), batch_size=batch_size, num_workers=workers)
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and precision == "bf16"):
                    for e in todo:
                        out[e].append(models[e](batch).float().cpu().numpy())
        for e in todo:
            emb = np.concatenate(out[e])
            if emb.shape != (len(tiles), 768) or not np.isfinite(emb).all():
                raise ValueError(f"Invalid features for {sid} ep{e}: {emb.shape}")
            _write_npy(shards[e], emb)
        n_new += 1
        if n_new % 20 == 0 or i == len(slide_ids):
            print(f"[{i}/{len(slide_ids)}] {sid}: {len(tiles)} tiles, {time.time() - t0:.0f}s, "
                  f"{n_new} slides this run", flush=True)

    manifest = {
        "benchmark": str(bench_root.resolve()), "dataset": dataset, "labels": str(labels_path.resolve()),
        "labels_sha256": hash_file(labels_path), "n_slides": len(slide_ids),
        "backbone": "student_backbone", "model": "vit_base_patch16_224", "output": "CLS after final norm (768)",
        "preprocessing": "tggate-bench tiles (256 px, level 0); PIL Resize 224 + CenterCrop 224; ImageNet mean/std",
        "precision": "bf16 autocast on CUDA, saved as float32" if precision == "bf16" else "fp32",
        "pretraining_overlap": "likely: exp 0028 pretrained on TG-GATEs liver patches; slide-level overlap unchecked",
        "torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__,
        "models": [],
    }
    for e in epochs:
        key = model_key(e)
        paths = [run_dirs[e] / "_shards" / f"{sid}.npy" for sid in slide_ids]
        n_tiles = np.array([np.load(p, mmap_mode="r").shape[0] for p in paths])
        stack = np.zeros((len(paths), n_tiles.max(), 768), dtype=np.float32)
        means = np.zeros((len(paths), 768), dtype=np.float32)
        for k, p in enumerate(paths):
            emb = np.load(p)
            stack[k, :len(emb)] = emb
            means[k] = emb.mean(axis=0)
        npz = run_dirs[e] / f"{key}.npz"
        np.savez(npz, slide_ids=slide_ids.astype(str), tile_embeddings=stack, n_tiles=n_tiles,
                 mean_embeddings=means, model_key=key)
        del stack
        manifest["models"].append({"model": key, "epoch": e, "checkpoint": str(checkpoints[e].resolve()),
                                   "checkpoint_sha256": hash_file(checkpoints[e]), "npz": str(npz),
                                   "tiles_per_slide": [int(n_tiles.min()), int(n_tiles.max())]})
        (run_dirs[e] / "completion.json").write_text(json.dumps({"status": "completed", "model": key}, indent=2))
        print(f"wrote {npz}", flush=True)
    path = overlay / "outputs" / EMB_EXP / f"{dataset}__dino0028_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def eval_tasks(dataset, epochs):
    """(experiment, args) per tggate-bench run: anomaly score, then ridge per pooler."""
    tasks = []
    for e in epochs:
        tasks.append((ANOMALY_EXP, ["--dataset", dataset, "--model", model_key(e), "--variant", "study"]))
        tasks += [(POOLED_EXP, ["--dataset", dataset, "--model", model_key(e), "--pooler", p]) for p in POOLERS]
    return tasks


def run_bench_experiment(bench_root, overlay, exp, args):
    """Run tggate-bench's experiment.py as is, with the overlay as PROJECT_ROOT."""
    env = {**os.environ, "PROJECT_ROOT": str(Path(overlay).resolve()), "EXP_NAME": exp}
    env.pop("OUTPUT_ROOT", None)
    script = Path(bench_root) / "experiments" / exp / "experiment.py"
    subprocess.run([sys.executable, str(script), *args], env=env, check=True)


def _load_bench_module(bench_root, exp):
    path = Path(bench_root) / "experiments" / exp / "experiment.py"
    spec = importlib.util.spec_from_file_location(f"tggate_bench_{exp}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_metrics(dirs):
    return pd.concat([pd.read_csv(d / "metrics_by_study.csv", dtype={"exp_id": str}) for d in dirs],
                     ignore_index=True)


def summarize(bench_root, overlay, dataset, epochs, output, n_bootstrap=1000, seed=42):
    """Study-mean AUROC / Spearman across findings with study-level bootstrap
    CIs (tggate-bench Exp 0005's estimator), DINO epochs next to the existing
    encoders, plus paired study-level differences."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench_root, overlay, output = Path(bench_root), Path(overlay), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_mod = _load_bench_module(bench_root, SUMMARY_EXP)

    dino = [overlay / "outputs" / ANOMALY_EXP / f"{dataset}__{model_key(e)}" for e in epochs]
    dino += [overlay / "outputs" / POOLED_EXP / f"{dataset}__{model_key(e)}__{p}" for e in epochs for p in POOLERS]
    for d in dino:
        status = json.loads((d / "completion.json").read_text()).get("status") if (d / "completion.json").exists() else None
        if status != "completed":
            raise FileNotFoundError(f"Not completed: {d}")
    base = [bench_root / "outputs" / ANOMALY_EXP / f"{dataset}__{m}" for m in BASELINES if m != "imgstats"]
    base += [bench_root / "outputs" / POOLED_EXP / f"{dataset}__{m}__{p}" for m in BASELINES for p in POOLERS]
    base = [d for d in base if (d / "completion.json").exists()]

    by_study = _read_metrics(dino + base)
    keys = ["model", "pooler", "centering"]
    across = by_study.groupby(keys + ["exp_id"], as_index=False)[["auroc", "spearman"]].mean()
    across["finding"] = "ALL"
    studies = sorted(across["exp_id"].unique())
    boot = np.random.default_rng(seed).integers(0, len(studies), size=(n_bootstrap, len(studies)))
    table = pd.concat([summary_mod.bootstrap_study_mean(across, studies, boot, m) for m in ("auroc", "spearman")],
                      axis=1).reset_index().drop(columns="finding")
    table.to_csv(output / "summary_across_findings.csv", index=False)
    across.to_csv(output / "metrics_by_study_across_findings.csv", index=False)

    keys_dino = [model_key(e) for e in epochs]
    pairs = [(k, ref) for k in keys_dino for ref in ("resnet50-imagenet", "uni")]
    pairs += [(keys_dino[i], keys_dino[-1]) for i in range(len(keys_dino) - 1)]
    rows = []
    for (pooler, centering), grp in across.groupby(["pooler", "centering"]):
        wide = grp.pivot_table(index="exp_id", columns="model", values="auroc").reindex(studies)
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).to_numpy()
            samples = np.nanmean(d[boot], axis=1)
            rows.append({"pooler": pooler, "centering": centering, "model_a": a, "model_b": b,
                         "auroc_diff": np.nanmean(d), "lo": np.nanpercentile(samples, 2.5),
                         "hi": np.nanpercentile(samples, 97.5), "n_studies": int((~np.isnan(d)).sum())})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(output / "paired_auroc_differences.csv", index=False)

    _plot(table, epochs, output / "auroc_by_epoch.png", plt)
    return table, diffs


def _plot(table, epochs, path, plt):
    """Small multiples per (pooler, centering): DINO AUROC by epoch with its
    95% CI band; the other encoders as labelled gray reference lines."""
    panels = table[table["model"].str.startswith("dino0028")][["pooler", "centering"]].drop_duplicates()
    panels = panels.sort_values(["pooler", "centering"]).to_numpy().tolist()
    cols = 4
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False, sharey=True)
    for ax, (pooler, centering) in zip(axes.flat, panels):
        sub = table[(table["pooler"] == pooler) & (table["centering"] == centering)]
        dino = sub[sub["model"].str.startswith("dino0028")].copy()
        dino["epoch"] = dino["model"].str[-3:].astype(int)
        dino = dino.sort_values("epoch")
        ax.fill_between(dino["epoch"], dino["auroc_lo"], dino["auroc_hi"], color="#2a78d6", alpha=.18, lw=0)
        ax.plot(dino["epoch"], dino["auroc_study_mean"], color="#2a78d6", lw=2, marker="o", ms=5, label="DINO 0028")
        for _, r in sub[~sub["model"].str.startswith("dino0028")].iterrows():
            ax.axhline(r["auroc_study_mean"], color="#9a9a9a", lw=1, ls="--")
            ax.text(epochs[-1] + 8, r["auroc_study_mean"], r["model"], fontsize=7, color="#555555", va="center")
        ax.set(title=f"{pooler} / {centering}", xlabel="epoch", xlim=(0, epochs[-1] + 90))
        ax.grid(alpha=.25, lw=.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("study-mean AUROC (6 findings)")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
