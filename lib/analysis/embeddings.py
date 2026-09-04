# -*- coding: utf-8 -*-
"""
# 固定パッチ集合の埋め込み抽出

`scripts/analysis/extract_embeddings.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI と methods.yaml の解釈は呼び出し側に残してある。

パッチの読み出しは EXP8（0008）の共有 memmap 形式（`patches.memmap` + `index.csv`）。
旧 WebDataset(.tar shard) 版から `lib/trainer/data.py` と同じ形式に揃えたもので、
`data/shards/` は既に撤去済み。
"""
import os
import json

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

import lib.model.zoo as zoo
from lib.trainer.data import load_index_table, split_wsi_ids_by_fold

THUMB = 96  # thumbnail size (px) stored for prototype montages


def eval_transform(grayscale=False):
    steps = [transforms.Resize((224, 224))]
    if grayscale:
        # decolorize but keep 3 channels so ImageNet-pretrained stems still apply
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


def load_fixed_patches(data_dir, per_slide=10, fold_idx=0, num_folds=5,
                       patch_size=224, grayscale=False, filter_wsi=""):
    """Deterministically load per_slide patches from every held-out WSI.

    学習と同じ WSI 単位の fold 分割を使い、val fold（＝事前学習で重みを更新して
    いない WSI）だけから取る。同一チェックポイントの ep 推移を並べたときに
    「学習データを覚えただけ」を表現の改善と誤読しないため。

    `index.csv` 内の各 WSI の行は EXP8（0008）の採取時点で既に空間的にランダム
    化されているので、先頭 per_slide 行を取るだけで空間バイアスの無い決定的
    サンプルになる。RNG を使わないので seed 管理が要らず、全手法・全 epoch で
    自動的に同一集合・同一順序になる（この比較の前提）。

    per_slide: 1 WSI あたりの枚数。filter_wsi 指定時はその 1 枚からの総枚数。
    filter_wsi: if set, keep only patches from that WSI (single-slide analysis).
    grayscale: decolorize inputs (color-ablation control).
    """
    memmap_path = os.path.join(data_dir, "patches.memmap")
    if not os.path.exists(memmap_path):
        raise FileNotFoundError(f"patches.memmap が見つかりません: {memmap_path}")
    index_df = load_index_table(data_dir)

    # 途中で切れた memmap をそのまま読むと reshape は通ってしまい、中身だけが
    # ずれる（224px を 112px と誤ると行数がちょうど4倍になって成立する）。
    # index.csv の行数と patch_size から期待サイズを出して先に弾く。
    expected = len(index_df) * patch_size * patch_size * 3
    actual = os.path.getsize(memmap_path)
    if actual != expected:
        raise RuntimeError(
            f"patches.memmap のサイズが不整合: {actual} bytes。"
            f"index.csv {len(index_df)} 行 × {patch_size}px からの期待値は {expected} bytes。"
            f"転送または生成が途中で切れていないか確認すること。")

    # fold 分割は lib/trainer/data.py の関数をそのまま使う。wsi_id は str で
    # 読まれるので np.sort は辞書順になり（4桁 ID と 5桁 ID が混在している）、
    # ここを数値ソートで書き直すと学習時と違う val 集合になる。
    _, val_wsi_ids = split_wsi_ids_by_fold(index_df["wsi_id"].unique(), fold_idx, num_folds)
    df = index_df[index_df["wsi_id"].isin(val_wsi_ids)]
    if filter_wsi:
        df = df[df["wsi_id"] == str(filter_wsi)]
        if df.empty:
            raise RuntimeError(
                f"WSI {filter_wsi} は fold {fold_idx} の val 側にない"
                f"（val は {len(val_wsi_ids)} WSI）。")

    # sort=True で wsi_id 順を固定して各 WSI の先頭 per_slide 行を取り、読み出しは
    # row 昇順にする（150GB の memmap へのアクセスがほぼ順次になる）。
    sel = df.groupby("wsi_id", sort=True).head(per_slide).sort_values("row")
    print(f"[patches] fold {fold_idx}/{num_folds} val: "
          f"{sel['wsi_id'].nunique()} WSIs x {per_slide} = {len(sel)} patches"
          f"{' | grayscale' if grayscale else ''}"
          f"{f' | wsi={filter_wsi}' if filter_wsi else ''}")

    mm = np.memmap(memmap_path, dtype=np.uint8, mode="r").reshape(-1, patch_size, patch_size, 3)
    tf = eval_transform(grayscale)
    tensors, thumbs, meta = [], [], []
    for r in sel.itertuples(index=False):
        # memmap から明示的にコピーする（lib/trainer/data.py の Dataset と同じ扱い）。
        img = Image.fromarray(np.array(mm[r.row]))
        tensors.append(tf(img))
        thumbs.append(np.asarray(img.resize((THUMB, THUMB)), dtype=np.uint8))
        # スライドラベルのキー名は "wsi"。lib/analysis/batch_color.py と
        # compare.run(label_key=...) がこの名前で引くので "wsi_id" にしない。
        meta.append({"key": f"{r.wsi_id}_{r.row}", "wsi": r.wsi_id, "row": int(r.row),
                     "x": int(r.x), "y": int(r.y), "blur_score": float(r.blur_score)})
    print(f"[patches] loaded {len(tensors)} patches")
    if not tensors:
        raise RuntimeError("No patches matched (check --filter_wsi / fold_idx).")
    return torch.stack(tensors), np.stack(thumbs), meta


@torch.no_grad()
def embed(model, x, device, batch_size=256):
    outs = []
    model.eval()
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size].to(device)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model(xb)
        f = torch.flatten(f, start_dim=1).float().cpu()
        outs.append(f)
    return torch.cat(outs).numpy()


def run(methods_config, data_dir="data/ssl_patches", per_slide=10, fold_idx=0, num_folds=5,
        batch_size=256, output_dir="outputs/representation_analysis",
        grayscale=False, filter_wsi=""):
    """全手法で共通のパッチ集合を埋め込み、emb_{name}.npy として保存する。

    旧 `scripts/analysis/extract_embeddings.py` の main() 本体をそのまま関数化した
    もの（REFACTOR_PLAN.md §5-0「研究の実処理は lib/」/ Phase 4）。
    scripts 側は CLI シムとして残してある。

    checkpoint が見つからない手法はスキップするので、学習が全部終わる前でも回せる。
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(methods_config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    x, thumbs, meta = load_fixed_patches(
        data_dir, per_slide=per_slide, fold_idx=fold_idx, num_folds=num_folds,
        grayscale=grayscale, filter_wsi=filter_wsi)
    np.save(os.path.join(output_dir, "patches_thumbs.npy"), thumbs)
    with open(os.path.join(output_dir, "patches_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    done = {}
    for m in cfg["methods"]:
        name = m["name"]
        ssl_name = m["ssl_name"]
        is_foundation = ssl_name == "foundation"
        if is_foundation:
            # Pretrained external foundation model (e.g. UNI): no in-project
            # checkpoint; weights come from timm/HF cache, so no model_path to check.
            print(f"[embed] {name}  ({m['model_name']} / foundation)")
            model = zoo.prepare_foundation_eval(
                model_name=m["model_name"], DEVICE=device)
        else:
            mp = m["model_path"]
            if not os.path.exists(mp):
                print(f"[skip] {name}: checkpoint not found ({mp})")
                continue
            print(f"[embed] {name}  ({m['model_name']} / {ssl_name})")
            model = zoo.prepare_model_eval(
                model_name=m["model_name"], ssl_name=ssl_name,
                model_path=mp, pretrained=False, DEVICE=device)
        emb = embed(model, x, device, batch_size=batch_size)
        np.save(os.path.join(output_dir, f"emb_{name}.npy"), emb)
        print(f"        -> emb_{name}.npy  shape={emb.shape}")
        done[name] = list(emb.shape)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "extract_summary.json"), "w") as f:
        json.dump({"n_patches": int(x.size(0)),
                   "n_slides": len({m["wsi"] for m in meta}),
                   "per_slide": per_slide, "fold_idx": fold_idx, "num_folds": num_folds,
                   "embeddings": done}, f, indent=2)
    print(f"[done] embeddings for {list(done)} in {output_dir}")
    return done
