"""
MAE ViT-B/16 の1GPUあたり最大batch_sizeを探る使い捨てスクリプト(run_slurm.shからは呼ばれない)。

lib/sslmodel/sslutils.py の MAE クラス(本番のモデル定義そのもの)を使い、
本物のWSIパッチの代わりに合成ランダム画像テンソルで forward/backward/optimizer.step
を数回試す(lib/trainer/loop.py の diagnose_gpu_bound と同じ発想)。
DDPは絡めない(VRAM上限は1GPUあたりの値で決まり、2GPU DDPでも各rankが同じ量を
使うだけなので、1GPUで探れば十分)。

実行: PROJECT_ROOT配下で `python experiments/0013_20260805_paper_mae_vitb16/vram_probe.py`
(.venv有効化 + PYTHONPATH設定済み前提)
"""
import sys

import torch

sys.path.insert(0, "/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison")

from lib.sslmodel.sslutils import MAE

CANDIDATES = [4096, 3072, 2560, 2048, 1536, 1280, 1024, 768, 512]
N_STEPS = 5  # OOMが出るかどうかの確認だけなので数stepで十分


def try_batch_size(bs: int, device: torch.device) -> tuple[bool, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ssl = MAE(DEVICE=device)
    model, criterion = ssl.prepare_model(head_size=768)  # ViT-B/16
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
    model.train()
    try:
        for _ in range(N_STEPS):
            x = torch.randn(bs, 3, 224, 224, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ssl.calc_loss(model, [x], criterion)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        return False, float("nan")
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    device = torch.device("cuda:0")
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"device: {torch.cuda.get_device_name(device)}  total_vram: {total_gb:.1f} GB")
    for bs in CANDIDATES:
        ok, peak_gb = try_batch_size(bs, device)
        if ok:
            print(f"batch_size={bs:>5}  OK   peak={peak_gb:.2f} GB  ({peak_gb / total_gb * 100:.1f}%)")
        else:
            print(f"batch_size={bs:>5}  OOM")


if __name__ == "__main__":
    main()
