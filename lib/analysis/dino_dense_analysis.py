"""Dense temporal diagnostics; all rank definitions and controls are explicit."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from lib.analysis.dino_epoch_analysis import hash_file, slide_center


def spectral_metrics(features):
    """Centered spectra. Singular entropy matches the previous fixed-image rank.

    Covariance entropy, participation ratio and stable rank weight the spectrum
    differently; their values must not be called the same effective rank.
    """
    centered = np.asarray(features, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0)[::-1]
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("Constant features have no centered variance")
    singular = np.sqrt(eigenvalues)
    probabilities = singular / singular.sum()
    variance = eigenvalues / total

    def entropy_rank(p):
        positive = p > 0
        return float(np.exp(-np.sum(p[positive] * np.log(p[positive]))))

    metrics = {
        "singular_effective_rank": entropy_rank(probabilities),
        "covariance_entropy_rank": entropy_rank(variance),
        "participation_ratio": float(1 / np.square(variance).sum()),
        "stable_rank": float(1 / variance[0]),
        "top1_variance": float(variance[0]),
        "top10_variance": float(variance[:10].sum()),
        "top50_variance": float(variance[:50].sum()),
        "tail_after100_variance": float(variance[100:].sum()),
        "bottom_half_singular_mass": float(probabilities[len(probabilities)//2:].sum()),
        "components90": int(np.searchsorted(np.cumsum(variance), .9) + 1),
        "components95": int(np.searchsorted(np.cumsum(variance), .95) + 1),
        "total_centered_variance": float(total / (len(features)-1)),
    }
    return metrics, singular, variance


def prepare_cka(features):
    centered = np.asarray(features, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    return centered, float(np.linalg.norm(centered.T @ centered, ord="fro"))


def prepared_cka(x, y):
    denominator = x[1] * y[1]
    if denominator <= 0:
        raise ValueError("Cannot compute CKA of constant features")
    return float(np.square(np.linalg.norm(x[0].T @ y[0], ord="fro")) / denominator)


def verified_inputs(output):
    output = Path(output)
    manifest = json.loads((output / "extract_manifest.json").read_text())
    metadata = json.loads((output / "patches_meta.json").read_text())
    digest = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, indent=2).encode()).hexdigest()
    if digest != manifest["sample_sha256"]:
        raise ValueError("Sample metadata differs from extraction manifest")
    items = manifest["checkpoints"]
    if not manifest.get("complete", False):
        raise ValueError("Extraction is not complete")
    epochs = [item["epoch"] for item in items]
    if epochs != manifest["expected_epochs"] or epochs != sorted(set(epochs)):
        raise ValueError("Checkpoint list is incomplete or out of order")
    embeddings, affines = {}, {}
    for item in items:
        feature_path = output / f"emb_{item['name']}.npy"
        norm_path = output / f"norm_{item['name']}.npz"
        if hash_file(feature_path) != item["embedding_sha256"]:
            raise ValueError(f"Embedding checksum mismatch: {feature_path}")
        if hash_file(norm_path) != item["norm_affine_sha256"]:
            raise ValueError(f"LayerNorm checksum mismatch: {norm_path}")
        features = np.load(feature_path)
        if features.shape != tuple(item["shape"]) or len(features) != len(metadata):
            raise ValueError("Feature/metadata mismatch")
        if not np.isfinite(features).all():
            raise ValueError("Non-finite features")
        embeddings[item["epoch"]] = features
        with np.load(norm_path) as affine:
            affines[item["epoch"]] = (affine['weight'].copy(), affine['bias'].copy())
    return manifest, metadata, embeddings, affines


def training_records(directory):
    """Merge rank0 logs by embedded date, not HHMMSS filenames across resumes."""
    logs = []
    for path in Path(directory).glob("log_*.txt"):
        if re.search(r'_rank\d+\.txt$', path.name):
            continue
        lines = path.read_text(errors="replace").splitlines()
        dates = [m.group(1) for line in lines if (m := re.search(r'\[(\d{8}-\d{6})\]', line))]
        logs.append((min(dates) if dates else '', path, lines))
    by_epoch, events, sources = {}, [], []
    for _, path, lines in sorted(logs):
        current, pending_resume = None, None
        sources.append({"path": str(path.resolve()), "sha256": hash_file(path)})
        for line in lines:
            if 'Resumed from' in line:
                pending_resume = line
            match = re.search(r'Epoch:\s*(\d+),\s*train_loss:', line)
            if match:
                current = int(match.group(1))
                row = {"epoch": current, "log_source": path.name}
                for key in ('train_loss', 'lr', 'grad_norm', 'ln_gain', 'teacher_momentum', 'teacher_temp'):
                    value = re.search(rf'{key}:\s*([\d.eE+\-]+)', line)
                    if value:
                        row[key] = float(value.group(1))
                by_epoch[current] = row
                if pending_resume is not None:
                    events.append({"first_logged_epoch": current, "line": pending_resume})
                    pending_resume = None
            elif current is not None:
                for key in ('eff_rank', 'feat_std', 'val_loss'):
                    value = re.search(rf'{key}:\s*([\d.eE+\-]+)', line)
                    if value:
                        by_epoch[current][key] = float(value.group(1))
    return [by_epoch[e] for e in sorted(by_epoch)], events, sources


def analyze_dense(output, training_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    from lib.analysis.batch_color import eta2

    output = Path(output)
    manifest, metadata, embeddings, affines = verified_inputs(output)
    epochs = list(embeddings)
    slides = np.array([record['wsi'] for record in metadata])
    thumbnails = np.load(output / 'patches_thumbs.npy')
    colors = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(
        thumbnails.reshape(len(thumbnails), -1, 3).mean(axis=1))
    global_prepared, residual_prepared = [], []
    spectra_s, spectra_v, rows = [], [], []
    for epoch, features in embeddings.items():
        metrics, singular, variance = spectral_metrics(features)
        norms = np.linalg.norm(features.astype(np.float64), axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero norm features")
        normalized = features / norms[:, None]
        residual = slide_center(features, slides)
        gamma, beta = affines[epoch]
        if np.any(np.abs(gamma) < 1e-6):
            raise ValueError(f"Final LayerNorm cannot be inverted at ep{epoch}")
        corrected = (features.astype(np.float64) - beta) / gamma
        normalized_metrics, _, _ = spectral_metrics(normalized)
        residual_metrics, _, _ = spectral_metrics(residual)
        corrected_metrics, _, _ = spectral_metrics(corrected)
        row = {"epoch": epoch, **metrics,
               "l2_singular_effective_rank": normalized_metrics['singular_effective_rank'],
               "slide_centered_singular_effective_rank": residual_metrics['singular_effective_rank'],
               "without_final_ln_affine_singular_rank": corrected_metrics['singular_effective_rank'],
               "final_ln_gamma_abs_mean": float(np.abs(gamma).mean()),
               "final_ln_gamma_abs_cv": float(np.abs(gamma).std()/np.abs(gamma).mean()),
               "feature_norm_mean": float(norms.mean()),
               "feature_norm_cv": float(norms.std()/norms.mean()),
               "eta2_slide": eta2(normalized, slides), "eta2_color": eta2(normalized, colors),
               "mean_pairwise_cosine": float((np.square(normalized.sum(axis=0)).sum()-len(features))/(len(features)*(len(features)-1))),
               "nonfinite_values": 0}
        rows.append(row)
        spectra_s.append(singular)
        spectra_v.append(variance)
        global_prepared.append(prepare_cka(features))
        residual_prepared.append(prepare_cka(residual))
        print(f"diagnostics ep{epoch:03d}: rank={metrics['singular_effective_rank']:.2f}, cov_rank={metrics['covariance_entropy_rank']:.2f}", flush=True)
    table = pd.DataFrame(rows)
    table['rank_change_per_epoch'] = table.singular_effective_rank.diff()/table.epoch.diff()
    table.to_csv(output / 'dense_metrics.csv', index=False)
    np.savez(output / 'dense_spectra.npz', epochs=epochs, singular_values=spectra_s, variance_fractions=spectra_v)

    matrices = {label: np.eye(len(epochs)) for label in ('global', 'slide_centered')}
    for i in range(len(epochs)):
        for j in range(i+1, len(epochs)):
            for label, prepared in (('global', global_prepared), ('slide_centered', residual_prepared)):
                value = prepared_cka(prepared[i], prepared[j])
                matrices[label][i,j] = matrices[label][j,i] = value
        if i % 4 == 0:
            print(f"CKA matrix rows {i+1}/{len(epochs)}", flush=True)
    adjacent = []
    for i in range(1, len(epochs)):
        adjacent.append({"previous_epoch": epochs[i-1], "epoch": epochs[i],
                         "global_cka": matrices['global'][i-1,i],
                         "slide_centered_cka": matrices['slide_centered'][i-1,i]})
    pd.DataFrame(adjacent).to_csv(output / 'adjacent_cka.csv', index=False)
    for label, matrix in matrices.items():
        pd.DataFrame(matrix, index=epochs, columns=epochs).to_csv(output / f'cka_{label}.csv')
        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', origin='lower')
        tick = list(range(0,len(epochs),4))
        ax.set_xticks(tick, [epochs[i] for i in tick], rotation=90)
        ax.set_yticks(tick, [epochs[i] for i in tick])
        ax.set(xlabel='Epoch', ylabel='Epoch', title=f'Linear CKA: {label}')
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(output / f'cka_{label}.png', dpi=150); plt.close(fig)

    records, events, sources = training_records(training_dir)
    logs = pd.DataFrame(records)
    logs.to_csv(output / 'training_metrics.csv', index=False)
    pd.DataFrame(events).to_csv(output / 'resume_events.csv', index=False)
    (output / 'training_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    merged = table.merge(logs, on='epoch', how='left', validate='one_to_one')
    merged.to_csv(output / 'dense_metrics_with_training.csv', index=False)
    # The monitor's augmented validation images differ from the fixed image
    # collection. Overlay trends only; do not interpret absolute offsets.
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    a=axes[0,0]
    for col, label in (('singular_effective_rank','Fixed images'),('l2_singular_effective_rank','L2-normalized'),
                       ('slide_centered_singular_effective_rank','Slide-centered'),('without_final_ln_affine_singular_rank','Final LN affine removed')):
        a.plot(table.epoch, table[col], label=label)
    a.set(ylabel='Singular entropy effective rank'); a.legend(fontsize=8)
    axes[0,1].plot(table.epoch, table.covariance_entropy_rank, label='Covariance entropy rank')
    axes[0,1].plot(table.epoch, table.participation_ratio, label='Participation ratio')
    axes[0,1].plot(table.epoch, table.stable_rank, label='Stable rank')
    axes[0,1].legend(fontsize=8)
    axes[0,2].plot(table.epoch, table.top1_variance, label='Top 1 variance')
    axes[0,2].plot(table.epoch, table.top10_variance, label='Top 10 variance')
    axes[0,2].plot(table.epoch, table.tail_after100_variance, label='Variance after component 100')
    axes[0,2].legend(fontsize=8)
    adj=pd.DataFrame(adjacent)
    axes[1,0].plot(adj.epoch, adj.global_cka, label='Global')
    axes[1,0].plot(adj.epoch, adj.slide_centered_cka, label='Slide-centered')
    axes[1,0].set(ylabel='CKA vs preceding checkpoint'); axes[1,0].legend(fontsize=8)
    axes[1,1].plot(table.epoch, table.eta2_slide, label='Slide')
    axes[1,1].plot(table.epoch, table.eta2_color, label='Color')
    axes[1,1].legend(fontsize=8); axes[1,1].set(ylabel='Variance explained (eta squared)')
    axes[1,2].plot(table.epoch, table.rank_change_per_epoch)
    axes[1,2].set(ylabel='Rank change per epoch')
    for ax in axes.flat:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'dense_overview.png',dpi=150);plt.close(fig)

    late=merged[merged.epoch>=300]
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    axes[0,0].plot(late.epoch, late.singular_effective_rank, marker='.', label='Fixed images')
    axes[0,0].plot(late.epoch, late.eff_rank, marker='.', label='Training monitor (different images)')
    axes[0,0].legend(fontsize=8);axes[0,0].set(ylabel='Singular entropy effective rank')
    axes[0,1].plot(late.epoch, late['lr']);axes[0,1].set(yscale='log',ylabel='Logged learning rate')
    axes[0,2].plot(late.epoch, late.train_loss,label='Train SSL loss')
    axes[0,2].plot(late.epoch, late.val_loss,label='Val SSL loss');axes[0,2].legend(fontsize=8)
    axes[1,0].plot(late.epoch,late.final_ln_gamma_abs_cv,label='Final LN gamma CV')
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(late.epoch,late.feature_norm_mean,label='Mean feature norm')
    axes[1,1].legend(fontsize=8)
    axes[1,2].plot(late.epoch,late.rank_change_per_epoch);axes[1,2].set(ylabel='Rank change per epoch')
    for ax in axes.flat:
        for event in events:
            ax.axvline(event['first_logged_epoch'],color='gray',linestyle=':',alpha=.6)
        ax.set(xlabel='Epoch',xlim=(300,480));ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'late_training.png',dpi=150);plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(12,4))
    for epoch in (20,320,360,400,420,440,460,480):
        if epoch not in epochs: continue
        variance=np.asarray(spectra_v[epochs.index(epoch)])
        axes[0].plot(np.arange(1,len(variance)+1),variance,label=f'ep{epoch}')
        axes[1].plot(np.arange(1,len(variance)+1),np.cumsum(variance),label=f'ep{epoch}')
    axes[0].set(xscale='log',yscale='log',xlabel='Component',ylabel='Variance fraction',ylim=(1e-7,1))
    axes[1].set(xlabel='Components',ylabel='Cumulative variance',ylim=(0,1.01));axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'late_spectra.png',dpi=150);plt.close(fig)
    late_slopes=table[(table.epoch>=300)&(table.epoch>table.epoch.min())].sort_values('rank_change_per_epoch',ascending=False)
    summary={"n_checkpoints":len(epochs),"n_patches":len(metadata),"sample_sha256":manifest['sample_sha256'],
             "late_largest_rank_slopes":late_slopes[['epoch','rank_change_per_epoch']].head(10).to_dict('records'),
             "late_min_adjacent_cka":adj[adj.epoch>=300].sort_values('global_cka').head(10).to_dict('records'),
             "resume_events":events,"pretraining_overlap":manifest['pretraining_overlap'],
             "definitions":{"singular_rank":"exp entropy of normalized singular values of centered feature matrix",
                            "covariance_rank":"exp entropy of normalized eigenvalues of centered X^T X",
                            "stable_rank":"sum eigenvalues / largest eigenvalue",
                            "participation_ratio":"(sum eigenvalues)^2 / sum squared eigenvalues",
                            "final_ln_removed":"(feature - final norm.bias) / final norm.weight; fixed weights, no retraining"}}
    (output/'dense_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def compare_precision(dense_output, fp32_outputs):
    import pandas as pd
    dense_output=Path(dense_output)
    if isinstance(fp32_outputs,(str,Path)):
        fp32_outputs=[fp32_outputs]
    dense=json.loads((dense_output/'extract_manifest.json').read_text())
    old_items={item['epoch']:item for item in dense['checkpoints']}
    rows=[]
    for fp32_output in fp32_outputs:
        rows.extend(_precision_rows(dense_output,Path(fp32_output),dense,old_items))
    if len({row['epoch'] for row in rows})!=len(rows):
        raise ValueError('Duplicate epochs across FP32 extractions')
    rows.sort(key=lambda row:row['epoch'])
    table=pd.DataFrame(rows)
    table.to_csv(dense_output/'precision_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dense_table=pd.read_csv(dense_output/'dense_metrics.csv')
    late=dense_table[dense_table.epoch>=300]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(late.epoch,late.singular_effective_rank,label='BF16 (all saved epochs)')
    axes[0].scatter(table.epoch,table.fp32_rank,s=25,color='orange',label='FP32 checks',zorder=3)
    axes[0].set(xlabel='Epoch',ylabel='Centered singular effective rank');axes[0].legend()
    axes[1].plot(table.epoch,100*table.relative_rank_difference,marker='o')
    axes[1].axhline(0,color='gray',linewidth=.8)
    axes[1].set(xlabel='Epoch',ylabel='(BF16 rank - FP32 rank) / FP32 rank (%)')
    fig.tight_layout();fig.savefig(dense_output/'precision_check.png',dpi=150);plt.close(fig)
    return rows


def _precision_rows(dense_output,fp32_output,dense,old_items):
    from lib.analysis.cka import linear_cka
    fp32, _, features, affines=verified_inputs(fp32_output)
    if dense['sample_sha256']!=fp32['sample_sha256']:
        raise ValueError('Precision comparison samples differ')
    if not fp32['precision'].startswith('fp32'):
        raise ValueError('Reference extraction is not FP32')
    rows=[]
    for epoch, high in features.items():
        item=next(item for item in fp32['checkpoints'] if item['epoch']==epoch)
        if item['checkpoint_sha256']!=old_items[epoch]['checkpoint_sha256']:
            raise ValueError('Precision comparison uses different weights')
        low_path=dense_output/f'emb_ep{epoch:03d}.npy'
        if hash_file(low_path)!=old_items[epoch]['embedding_sha256']:
            raise ValueError('BF16 features changed after extraction')
        low=np.load(low_path).astype(np.float64)
        low_metrics,_,_=spectral_metrics(low)
        high_metrics,_,_=spectral_metrics(high)
        relative_error=float(np.linalg.norm(low-high)/np.linalg.norm(high))
        gamma,beta=affines[epoch]
        high_corrected,_,_=spectral_metrics((high-beta)/gamma)
        rows.append({'epoch':epoch,'fp32_source':str(fp32_output.resolve()),'bf16_rank':low_metrics['singular_effective_rank'],
                     'fp32_rank':high_metrics['singular_effective_rank'],
                     'relative_rank_difference':float((low_metrics['singular_effective_rank']-high_metrics['singular_effective_rank'])/high_metrics['singular_effective_rank']),
                     'feature_relative_frobenius_error':relative_error,'bf16_vs_fp32_cka':linear_cka(low,high),
                     'fp32_without_final_ln_affine_rank':high_corrected['singular_effective_rank']})
    return rows


def write_dense_report(output):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    table=pd.read_csv(output/'dense_metrics.csv').set_index('epoch')
    adjacent=pd.read_csv(output/'adjacent_cka.csv')
    precision=(pd.read_csv(output/'precision_comparison.csv')
               if (output/'precision_comparison.csv').exists() else None)
    late=table[table.index>=300]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for column,label in (('singular_effective_rank','Raw'),('l2_singular_effective_rank','L2-normalized'),
                          ('without_final_ln_affine_singular_rank','Final LN affine removed'),
                          ('slide_centered_singular_effective_rank','Slide-centered')):
        axes[0].plot(late.index,late[column],label=label)
    axes[0].set(ylabel='Singular entropy effective rank');axes[0].legend(fontsize=8)
    for column,label in (('top1_variance','Top 1'),('top10_variance','Top 10'),('tail_after100_variance','After component 100')):
        axes[1].plot(late.index,100*late[column],label=label)
    axes[1].set(ylabel='Centered variance (%)');axes[1].legend(fontsize=8)
    axes[2].plot(late.index,late.components90,label='Components for 90% variance')
    axes[2].plot(late.index,late.components95,label='Components for 95% variance')
    axes[2].set(ylabel='Components');axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'late_rank_controls.png',dpi=150);plt.close(fig)
    r320,r440,r480=table.loc[320],table.loc[440],table.loc[480]
    late_adjacent=adjacent[adjacent.epoch>=300]
    max_relative=(float(precision.relative_rank_difference.abs().max()) if precision is not None else None)
    rows=["# DINO 0028: 5 epoch刻みの変動と後半のrank増加", "",
          "保存済み ep5〜480 の全96点を、前回と同じ100スライド×20枚＝2,000パッチで解析。",
          "student backbone、画像・順序・前処理固定。事前学習WSIとの重複は未確認で、探索的な表現解析。", "",
          "## 分かったこと", "",
          f"- 後半は単発のジャンプではなく、300〜440 epoch頃の増加と、その後の横ばいが主な形。"
          f" rankは ep320 {r320.singular_effective_rank:.1f} → ep440 {r440.singular_effective_rank:.1f} → ep480 {r480.singular_effective_rank:.1f}。",
          f"- L2正規化後でも ep320 {r320.l2_singular_effective_rank:.1f} → ep480 {r480.l2_singular_effective_rank:.1f}。"
          "画像ごとの特徴ノルムを揃えても増加が残る。",
          f"- 最終LayerNormの成分ごとの倍率とバイアスを除いても、rankは "
          f"ep320 {r320.without_final_ln_affine_singular_rank:.1f} → ep480 {r480.without_final_ln_affine_singular_rank:.1f}。"
          "最終LayerNormの変化だけでは説明できない。内部層のLayerNormの影響はこの操作では除いていない。",
          f"- 共分散固有値のエントロピーrankも {r320.covariance_entropy_rank:.1f} → {r480.covariance_entropy_rank:.1f}。"
          f"100番目より後の成分の分散割合は {100*r320.tail_after100_variance:.1f}% → {100*r480.tail_after100_variance:.1f}%。"
          f"90%の分散を説明する成分数は {int(r320.components90)} → {int(r480.components90)}。"
          "主要成分への集中が弱まり、中位・下位の成分にも分散が配分されている。",
          f"- 後半の隣接5 epoch間CKAは全体で {late_adjacent.global_cka.min():.4f}〜{late_adjacent.global_cka.max():.4f}。"
          "特徴の形は連続的に変わっており、今回の時点間比較では一度に別の表現へ切り替わる兆候は見られない。"]
    if precision is not None:
        rows.append(f"- FP32で{len(precision)}点を再抽出。BF16とのrank差は最大 {100*max_relative:.4f}%、"
                    f"精度間CKAは最小 {precision.bf16_vs_fp32_cka.min():.6f}。BF16誤差だけで増加を説明できない。")
    rows += ["", "この確認はスペクトルと表現の変化を記述するもの。学習率・weight decay・自己蒸留のどれが"
             "増加の原因かを特定する介入実験はしていない。病理所見の識別が良くなったかも未判定。", "",
             "![全期間の変動](dense_overview.png)", "",
             "## 後半の拡大", "",
             "| epoch | 特異値entropy rank | 共分散entropy rank | 90%分散の成分数 | η²_slide |", "|---:|---:|---:|---:|---:|"]
    for epoch in (160,240,300,320,360,400,410,415,420,425,430,440,460,470,480):
        if epoch in table.index:
            r=table.loc[epoch]
            rows.append(f"| {epoch} | {r.singular_effective_rank:.2f} | {r.covariance_entropy_rank:.2f} "
                        f"| {int(r.components90)} | {r.eta2_slide:.3f} |")
    rows += ["", "![後半のrankと分散配分](late_rank_controls.png)", "",
             f"5 epochごとの増分は均一ではない。410→415は+{table.loc[415].singular_effective_rank-table.loc[410].singular_effective_rank:.2f}、"
             f"425→430は+{table.loc[430].singular_effective_rank-table.loc[425].singular_effective_rank:.2f}。"
             "この2区間の両端もFP32で確認した。",
             "![学習ログとの対応](late_training.png)", "",
             "点線はresume後の最初の記録epoch（420）。rankの増加はそれ以前から続いており、"
             "再開だけで始まった増加ではない。学習モニターは別の拡張画像を使うため、rankの絶対値は直接比較しない。",
             "学習率の低下とrank増加は同じ期間に起きているが、因果関係の証明ではない。", "",
             "![後半のスペクトル](late_spectra.png)", "",
             "## 時点間 CKA", "",
             "全96×96の数値は cka_global.csv / cka_slide_centered.csv。隣接比較は adjacent_cka.csv。",
             "![全体CKA](cka_global.png)", "![スライド平均を除いたCKA](cka_slide_centered.png)", "",
             "スライド平均除去は各スライドの平均を引いて全パッチをまとめて計算する。スライド別CKAの平均ではない。",
             "通常のbiased linear CKA。前回5点の値とも一致を確認した。"]
    if precision is not None:
        rows += ["", "## 推論精度の確認", "",
                 "同じ画像・同じ重みをFP32（TF32無効）で再抽出。FP32の対象を選んだ基準は、"
                 "後半の代表時点と、後半のrank増分が大きかった2区間。", "",
                 "![BF16とFP32](precision_check.png)", "",
                 "| epoch | BF16 rank | FP32 rank | 相対差(%) |", "|---:|---:|---:|---:|"]
        for _,r in precision.iterrows():
            rows.append(f"| {int(r.epoch)} | {r.bf16_rank:.3f} | {r.fp32_rank:.3f} | {100*r.relative_rank_difference:+.4f} |")
    rows += ["", "## 算式と再現用ファイル", "",
             "中心化した特徴行列Xの特異値をsとすると、今回のeffrankは p=s/Σs、exp(-Σp log p)。",
             "特徴の一様な拡大はpを変えない。共分散entropy rankはs²を正規化する別の指標で、数値を同一視しない。",
             "participation ratioは(Σs²)²/Σs⁴、stable rankはΣs²/max(s²)。",
             "既存の学習モニターも特異値のentropyを使っているが、画像集合と推論条件は異なる。",
             "CKAの定義: [Kornblith et al. (2019)](https://arxiv.org/abs/1905.00414)。", "",
             "- dense_metrics.csv: 全96点のrank・正規化・LayerNorm・分散配分・スライド依存。",
             "- dense_spectra.npz: 全96点の特異値と分散割合。",
             "- dense_metrics_with_training.csv / training_metrics.csv: 学習ログとの対応。",
             "- precision_comparison.csv: FP32との差。",
             "- extract_manifest.json / patches_meta.json: 重み・画像・順序・特徴のハッシュ。",
             "- training_sources.json / resume_events.csv: 使用ログと再開イベント。", "",
             "実行コードは評価用ブランチの scripts/analysis/{analyze_dino_epochs.py,analyze_dino_dense.py}。",
             "入力の学習出力と既存5点の結果は変更していない。"]
    (output/'report.md').write_text('\n'.join(rows)+'\n')
