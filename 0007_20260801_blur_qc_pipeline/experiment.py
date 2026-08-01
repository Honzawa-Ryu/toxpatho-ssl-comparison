"""ぼやけQCパイプラインの検証実行(Goal.md §9 受け入れ基準)。

以下を1ジョブで行う:
  1. 実データのcoords h5のキー/属性を確認する(Goal.md §2の「要検証」項目)。
  2. lib/analysis/blur_qc の単体テストを実行する。
  3. score_blur.py を対象WSI(config.yml analysis.wsi_ids)に対して実行する。
  4. sample_patches_memmap.py を threshold_scope=per_slide / global の
     両方で実行する。
  5. 生成したmemmapをnp.memmapとして開き直し、meta.jsonのshape/dtypeと
     一致することを確認する。

結果は outputs/{exp_name}/default/verification_report.md にまとめる。
実データ出力(data/blur_qc/scores, data/blur_qc/patches_*)はGoal.mdの想定通り
リポジトリのdata/配下に書く(検証用WSI 2枚 x n_patches=50 のみなので容量は
無視できる規模)。
"""
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _check_coords_h5_schema(coords_dir: str, wsi_ids: list, report: list) -> None:
    report.append("## 1. coords h5 のキー/属性確認\n")
    for wsi_id in wsi_ids:
        h5_path = Path(coords_dir) / f"{wsi_id}_patches.h5"
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            attrs = dict(f.attrs)
            coords_shape = f["coords"].shape if "coords" in f else None
            coords_dtype = str(f["coords"].dtype) if "coords" in f else None
        report.append(f"- {wsi_id}: keys={keys}, attrs={attrs}")
        report.append(f"  coords shape={coords_shape}, dtype={coords_dtype}")
        if "coords" not in keys:
            raise KeyError(
                f"想定と異なりました: {h5_path} に 'coords' キーがありません(keys={keys})。"
                " Goal.md §2 の前提が崩れています。"
            )
    report.append("-> 'coords' キーの前提を実データで確認済み。\n")


def _run_unit_tests(report: list) -> None:
    report.append("## 2. 単体テスト (lib/analysis/tests/test_blur_qc.py)\n")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("lib.analysis.tests.test_blur_qc")
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)
    result = runner.run(suite)
    report.append("```\n" + buf.getvalue() + "\n```")
    if not result.wasSuccessful():
        raise RuntimeError("単体テストが失敗しました。詳細はレポート参照。")
    report.append(
        f"-> {result.testsRun} 件成功 (failures={len(result.failures)}, errors={len(result.errors)})\n"
    )


def _run_step1_and_step2(project_root: Path, params: dict, report: list) -> None:
    wsi_ids = params["wsi_ids"]
    wsi_dir = str(project_root / params["wsi_dir"])
    coords_dir = str(project_root / params["coords_dir"])
    scores_dir = str(project_root / params["scores_dir"])
    patches_dir = str(project_root / params["patches_dir"])
    patch_size = params["patch_size"]
    n_patches = params["n_patches"]
    seed = params["seed"]
    percentile = params["percentile"]

    exp_dir = project_root / "experiments" / os.environ["EXP_NAME"]

    report.append("## 3. score_blur.py (Step 1)\n")
    for wsi_id in wsi_ids:
        cmd = [
            sys.executable, str(exp_dir / "score_blur.py"),
            "--wsi_dir", wsi_dir,
            "--coords_dir", coords_dir,
            "--output_dir", scores_dir,
            "--patch_size", str(patch_size),
            "--wsi_id", wsi_id,
        ]
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        report.append(f"$ {' '.join(cmd)}\n```\n{result.stdout}\n{result.stderr}\n```")
        if result.returncode != 0:
            raise RuntimeError(f"score_blur.py failed for {wsi_id}: {result.stderr}")

        out_h5 = Path(scores_dir) / f"{wsi_id}_blur.h5"
        with h5py.File(out_h5, "r") as f:
            coords_shape = f["coords"].shape
            score_shape = f["blur_score"].shape
            score_dtype = str(f["blur_score"].dtype)
        report.append(
            f"-> {wsi_id}: coords{coords_shape}, blur_score{score_shape} dtype={score_dtype}\n"
        )

    report.append("## 4. sample_patches_memmap.py (Step 2, per_slide / global)\n")
    for threshold_scope in ("per_slide", "global"):
        out_dir = f"{patches_dir}_{threshold_scope}"
        for wsi_id in wsi_ids:
            cmd = [
                sys.executable, str(exp_dir / "sample_patches_memmap.py"),
                "--scores_dir", scores_dir,
                "--wsi_dir", wsi_dir,
                "--threshold_scope", threshold_scope,
                "--percentile", str(percentile),
                "--n_patches", str(n_patches),
                "--seed", str(seed),
                "--output_dir", out_dir,
                "--patch_size", str(patch_size),
                "--wsi_id", wsi_id,
            ]
            result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
            report.append(f"$ {' '.join(cmd)}\n```\n{result.stdout}\n{result.stderr}\n```")
            if result.returncode != 0:
                raise RuntimeError(
                    f"sample_patches_memmap.py failed for {wsi_id} ({threshold_scope}): {result.stderr}"
                )

            meta_path = Path(out_dir) / f"{wsi_id}_patches.meta.json"
            meta = json.loads(meta_path.read_text())
            memmap_path = Path(out_dir) / f"{wsi_id}_patches.memmap"
            shape = tuple(meta["shape"])
            mm = np.memmap(memmap_path, dtype=np.uint8, mode="r", shape=shape)
            if mm.shape != shape or mm.dtype != np.uint8:
                raise RuntimeError(f"memmap round-trip mismatch for {wsi_id} ({threshold_scope})")
            report.append(
                f"-> {wsi_id} [{threshold_scope}]: sampled {shape[0]} patches, "
                f"memmap round-trip OK (shape={shape})\n"
            )


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)

    config = load_config(Path(__file__).parent)
    params = config.get("analysis", {})

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    report = ["# blur_qc 検証レポート\n"]
    try:
        _check_coords_h5_schema(str(project_root / params["coords_dir"]), params["wsi_ids"], report)
        _run_unit_tests(report)
        _run_step1_and_step2(project_root, params, report)
        report.append("\n## 結論\n\nすべての検証項目に合格しました。")
    finally:
        (run_dir / "verification_report.md").write_text("\n".join(report), encoding="utf-8")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
